"""
Database operations for UID Matcher Reflex App
Handles Snowflake and SurveyMonkey API connections
"""

import os
import pandas as pd
import requests
import logging
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.snowflake_engine = None
        self.surveymonkey_token = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            self.surveymonkey_token = os.getenv("SURVEYMONKEY_ACCESS_TOKEN")
            
            # Snowflake connection string
            sf_user = os.getenv("SNOWFLAKE_USER")
            sf_password = os.getenv("SNOWFLAKE_PASSWORD")
            sf_account = os.getenv("SNOWFLAKE_ACCOUNT")
            sf_database = os.getenv("SNOWFLAKE_DATABASE")
            sf_schema = os.getenv("SNOWFLAKE_SCHEMA")
            sf_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
            sf_role = os.getenv("SNOWFLAKE_ROLE")
            
            if all([sf_user, sf_password, sf_account, sf_database, sf_schema, sf_warehouse, sf_role]):
                connection_string = (
                    f"snowflake://{sf_user}:{sf_password}@{sf_account}/"
                    f"{sf_database}/{sf_schema}?warehouse={sf_warehouse}&role={sf_role}"
                )
                self.snowflake_engine = create_engine(connection_string)
                
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    def check_snowflake_connection(self) -> Tuple[bool, str]:
        """Check Snowflake connection status"""
        try:
            if not self.snowflake_engine:
                return False, "No Snowflake configuration found"
            
            with self.snowflake_engine.connect() as conn:
                result = conn.execute(text("SELECT CURRENT_VERSION()"))
                version = result.fetchone()[0]
                return True, f"Connected to Snowflake version {version}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def check_surveymonkey_connection(self) -> Tuple[bool, str]:
        """Check SurveyMonkey API connection"""
        try:
            if not self.surveymonkey_token:
                return False, "No SurveyMonkey token found"
            
            headers = {
                "Authorization": f"Bearer {self.surveymonkey_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.surveymonkey.com/v3/users/me",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                username = user_data.get("username", "Unknown")
                return True, f"Connected as {username}"
            elif response.status_code == 401:
                return False, "Authentication failed - invalid token"
            else:
                return False, f"API error: {response.status_code}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_surveys(self) -> List[Dict]:
        """Get all surveys from SurveyMonkey"""
        try:
            if not self.surveymonkey_token:
                return []
            
            url = "https://api.surveymonkey.com/v3/surveys"
            headers = {"Authorization": f"Bearer {self.surveymonkey_token}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                logger.error(f"Failed to get surveys: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting surveys: {e}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def get_survey_details(self, survey_id: str) -> Optional[Dict]:
        """Get detailed survey information"""
        try:
            if not self.surveymonkey_token:
                return None
            
            url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
            headers = {"Authorization": f"Bearer {self.surveymonkey_token}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting survey details for {survey_id}: {e}")
            return None
    
    def get_question_bank(self, limit: int = 10000, offset: int = 0) -> pd.DataFrame:
        """Get question bank from Snowflake"""
        try:
            if not self.snowflake_engine:
                return pd.DataFrame()
            
            query = """
                SELECT HEADING_0, MAX(UID) AS UID
                FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
                GROUP BY HEADING_0
                LIMIT :limit OFFSET :offset
            """
            
            with self.snowflake_engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
            
            result.columns = result.columns.str.lower()
            result = result.rename(columns={'heading_0': 'HEADING_0', 'uid': 'UID'})
            return result
            
        except Exception as e:
            logger.error(f"Failed to get question bank: {e}")
            return pd.DataFrame()
    
    def get_question_bank_with_authority(self) -> pd.DataFrame:
        """Get question bank with authority count"""
        try:
            if not self.snowflake_engine:
                return pd.DataFrame()
            
            query = """
            SELECT 
                HEADING_0, 
                UID, 
                COUNT(*) as AUTHORITY_COUNT
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
            AND TRIM(HEADING_0) != ''
            GROUP BY HEADING_0, UID
            ORDER BY UID, AUTHORITY_COUNT DESC
            """
            
            with self.snowflake_engine.connect() as conn:
                result = pd.read_sql(text(query), conn)
            
            result.columns = result.columns.str.upper()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get question bank with authority: {e}")
            return pd.DataFrame()
    
    def upload_dataframe_to_snowflake(self, df: pd.DataFrame, table_name: str) -> bool:
        """Upload DataFrame to Snowflake"""
        try:
            if not self.snowflake_engine or df.empty:
                return False
            
            df.to_sql(
                table_name,
                self.snowflake_engine,
                if_exists='replace',
                index=False,
                method='multi'
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to Snowflake: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 