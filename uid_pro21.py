import streamlit as st
import pandas as pd
import requests
import re
import logging
import json
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Enhanced Page Configuration
st.set_page_config(
    page_title="UID Matcher Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333;
        margin: 1rem 0;
    }
    .sidebar .stSelectbox label {
        color: #667eea;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MatchingThresholds:
    """Configuration class for matching thresholds"""
    TFIDF_HIGH_CONFIDENCE: float = 0.60
    TFIDF_LOW_CONFIDENCE: float = 0.50
    SEMANTIC_THRESHOLD: float = 0.60
    HEADING_TFIDF_THRESHOLD: float = 0.55
    HEADING_SEMANTIC_THRESHOLD: float = 0.65
    HEADING_LENGTH_THRESHOLD: int = 50

@dataclass
class AppConfig:
    """Application configuration"""
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    BATCH_SIZE: int = 1000
    DEFAULT_SYNONYM_MAP: Dict[str, str] = None
    HEADING_REFERENCES: List[str] = None
    
    def __post_init__(self):
        if self.DEFAULT_SYNONYM_MAP is None:
            self.DEFAULT_SYNONYM_MAP = {
                "please select": "what is",
                "sector you are from": "your sector",
                "identity type": "id type",
                "what type of": "type of",
                "are you": "do you",
            }
        
        if self.HEADING_REFERENCES is None:
            self.HEADING_REFERENCES = [
                "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
                "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
                "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
                "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
                "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
            ]

# Initialize configuration
config = AppConfig()
thresholds = MatchingThresholds()

class UIManager:
    """Manages UI components and styling"""
    
    @staticmethod
    def render_header():
        st.markdown("""
        <div class="main-header">
            <h1>🧠 UID Matcher Pro</h1>
            <p>Intelligent Question Matching & Survey Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric(title: str, value: str, color: str = "primary"):
        colors = {
            "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "success": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "warning": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
            "info": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
        }
        st.markdown(f"""
        <div style="background: {colors.get(color, colors['primary'])}; padding: 1rem; border-radius: 8px; color: white; text-align: center; margin: 0.5rem 0;">
            <h3 style="margin: 0; color: white;">{value}</h3>
            <p style="margin: 0; color: white; opacity: 0.9;">{title}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_success(message: str):
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Success:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_warning(message: str):
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ Warning:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

class DataManager:
    """Handles all data operations and caching"""
    
    @staticmethod
    @st.cache_resource
    def load_sentence_transformer():
        logger.info(f"Loading SentenceTransformer model: {config.MODEL_NAME}")
        try:
            return SentenceTransformer(config.MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

    @staticmethod
    @st.cache_resource
    def get_snowflake_engine():
        try:
            sf = st.secrets["snowflake"]
            logger.info(f"Connecting to Snowflake: user={sf.user}, account={sf.account}")
            engine = create_engine(
                f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
                f"?warehouse={sf.warehouse}&role={sf.role}"
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT CURRENT_VERSION()"))
            return engine
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            DataManager._handle_snowflake_error(e)
            raise

    @staticmethod
    def _handle_snowflake_error(error):
        error_str = str(error)
        if "250001" in error_str:
            UIManager.show_warning(
                "Snowflake account is locked. UID matching disabled. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
            )
        elif "invalid identifier" in error_str.lower():
            UIManager.show_warning(
                "Invalid Snowflake table schema. Contact your admin to verify table structure."
            )

    @staticmethod
    @st.cache_data
    def get_tfidf_vectors(df_reference: pd.DataFrame):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(df_reference["norm_text"])
        return vectorizer, vectors

    @staticmethod
    def run_snowflake_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        try:
            with DataManager.get_snowflake_engine().connect() as conn:
                return pd.read_sql(text(query), conn, params=params or {})
        except Exception as e:
            logger.error(f"Snowflake query failed: {e}")
            DataManager._handle_snowflake_error(e)
            raise

class TextProcessor:
    """Handles text normalization and processing"""
    
    @staticmethod
    def enhanced_normalize(text: str, synonym_map: Dict[str, str] = None) -> str:
        if synonym_map is None:
            synonym_map = config.DEFAULT_SYNONYM_MAP
        
        text = str(text).lower()
        text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses content
        text = re.sub(r'[^a-z0-9 ]', '', text)  # Keep only alphanumeric and spaces
        
        # Apply synonym mapping
        for phrase, replacement in synonym_map.items():
            text = text.replace(phrase, replacement)
        
        # Remove stop words
        return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

    @staticmethod
    def classify_question(text: str, heading_references: List[str] = None) -> str:
        if heading_references is None:
            heading_references = config.HEADING_REFERENCES
        
        # Length-based classification
        if len(text.split()) > thresholds.HEADING_LENGTH_THRESHOLD:
            return "Heading"
        
        # TF-IDF similarity check
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            all_texts = heading_references + [text]
            tfidf_vectors = vectorizer.fit_transform([TextProcessor.enhanced_normalize(t) for t in all_texts])
            similarity_scores = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
            max_tfidf_score = np.max(similarity_scores)
            
            # Semantic similarity check
            model = DataManager.load_sentence_transformer()
            emb_text = model.encode([text], convert_to_tensor=True)
            emb_refs = model.encode(heading_references, convert_to_tensor=True)
            semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
            max_semantic_score = np.max(semantic_scores.cpu().numpy())
            
            # Classification decision
            if (max_tfidf_score >= thresholds.HEADING_TFIDF_THRESHOLD or 
                max_semantic_score >= thresholds.HEADING_SEMANTIC_THRESHOLD):
                return "Heading"
                
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
        
        return "Main Question/Multiple Choice"

class SurveyMonkeyAPI:
    """Handles SurveyMonkey API operations"""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.base_url = "https://api.surveymonkey.com/v3"

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"SurveyMonkey API request failed: {e}")
            raise

    def get_surveys(self) -> List[Dict]:
        return self._make_request("GET", "surveys").get("data", [])

    def get_survey_details(self, survey_id: str) -> Dict:
        return self._make_request("GET", f"surveys/{survey_id}/details")

    def create_survey(self, survey_data: Dict) -> str:
        response = self._make_request("POST", "surveys", json=survey_data)
        return response.get("id")

    def extract_questions(self, survey_json: Dict) -> List[Dict]:
        questions = []
        global_position = 0
        
        for page in survey_json.get("pages", []):
            for question in page.get("questions", []):
                q_text = question.get("headings", [{}])[0].get("heading", "")
                if not q_text:
                    continue
                
                global_position += 1
                question_data = self._process_question(question, survey_json, global_position, q_text)
                questions.append(question_data)
                
                # Process choices
                choices = question.get("answers", {}).get("choices", [])
                questions.extend(self._process_choices(choices, question, survey_json, global_position, q_text))
        
        return questions

    def _process_question(self, question: Dict, survey_json: Dict, position: int, q_text: str) -> Dict:
        family = question.get("family", None)
        choices = question.get("answers", {}).get("choices", [])
        
        # Determine schema type
        schema_type = self._determine_schema_type(family, choices, q_text)
        question_category = TextProcessor.classify_question(q_text)
        
        return {
            "heading_0": q_text,
            "position": position,
            "is_choice": False,
            "parent_question": None,
            "question_uid": question.get("id"),
            "schema_type": schema_type,
            "mandatory": False,
            "mandatory_editable": True,
            "survey_id": survey_json.get("id", ""),
            "survey_title": survey_json.get("title", ""),
            "question_category": question_category
        }

    def _process_choices(self, choices: List[Dict], question: Dict, survey_json: Dict, position: int, q_text: str) -> List[Dict]:
        choice_list = []
        for choice in choices:
            choice_text = choice.get("text", "")
            if choice_text:
                choice_data = {
                    "heading_0": f"{q_text} - {choice_text}",
                    "position": position,
                    "is_choice": True,
                    "parent_question": q_text,
                    "question_uid": question.get("id"),
                    "schema_type": self._determine_schema_type(question.get("family"), choices, q_text),
                    "mandatory": False,
                    "mandatory_editable": False,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", ""),
                    "question_category": "Main Question/Multiple Choice"
                }
                choice_list.append(choice_data)
        return choice_list

    def _determine_schema_type(self, family: str, choices: List[Dict], q_text: str) -> str:
        if family == "single_choice":
            return "Single Choice"
        elif family == "multiple_choice":
            return "Multiple Choice"
        elif family == "open_ended":
            return "Open-Ended"
        elif family == "matrix":
            return "Matrix"
        else:
            if choices:
                if "select one" in q_text.lower() or len(choices) <= 2:
                    return "Single Choice"
                return "Multiple Choice"
            return "Open-Ended"

class MatchingEngine:
    """Handles UID matching operations"""
    
    @staticmethod
    def compute_tfidf_matches(df_reference: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        # Prepare data
        df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
        df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
        
        # Normalize text
        df_reference["norm_text"] = df_reference["heading_0"].apply(TextProcessor.enhanced_normalize)
        df_target["norm_text"] = df_target["heading_0"].apply(TextProcessor.enhanced_normalize)

        # Compute similarities
        vectorizer, ref_vectors = DataManager.get_tfidf_vectors(df_reference)
        target_vectors = vectorizer.transform(df_target["norm_text"])
        similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

        # Process matches
        matched_uids, matched_qs, scores, confs = [], [], [], []
        for sim_row in similarity_matrix:
            best_idx = sim_row.argmax()
            best_score = sim_row[best_idx]
            
            confidence, uid, question = MatchingEngine._evaluate_match(
                best_score, best_idx, df_reference
            )
            
            matched_uids.append(uid)
            matched_qs.append(question)
            scores.append(round(best_score, 4))
            confs.append(confidence)

        # Add results to dataframe
        df_target["Suggested_UID"] = matched_uids
        df_target["Matched_Question"] = matched_qs
        df_target["Similarity"] = scores
        df_target["Match_Confidence"] = confs
        
        return df_target

    @staticmethod
    def _evaluate_match(score: float, idx: int, df_reference: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
        if score >= thresholds.TFIDF_HIGH_CONFIDENCE:
            conf = "✅ High"
        elif score >= thresholds.TFIDF_LOW_CONFIDENCE:
            conf = "⚠️ Low"
        else:
            conf = "❌ No match"
            idx = None
        
        uid = df_reference.iloc[idx]["uid"] if idx is not None else None
        question = df_reference.iloc[idx]["heading_0"] if idx is not None else None
        
        return conf, uid, question

    @staticmethod
    def compute_semantic_matches(df_reference: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        try:
            model = DataManager.load_sentence_transformer()
            emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
            emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
            cosine_scores = util.cos_sim(emb_target, emb_ref)

            sem_matches, sem_scores = [], []
            for i in range(len(df_target)):
                best_idx = cosine_scores[i].argmax().item()
                score = cosine_scores[i][best_idx].item()
                
                if score >= thresholds.SEMANTIC_THRESHOLD:
                    sem_matches.append(df_reference.iloc[best_idx]["uid"])
                    sem_scores.append(round(score, 4))
                else:
                    sem_matches.append(None)
                    sem_scores.append(None)

            df_target["Semantic_UID"] = sem_matches
            df_target["Semantic_Similarity"] = sem_scores
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            st.error(f"Semantic matching failed: {e}")
        
        return df_target

    @staticmethod
    def finalize_matches(df_target: pd.DataFrame, df_reference: pd.DataFrame) -> pd.DataFrame:
        # Combine TF-IDF and semantic matches
        df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
        df_target["configured_final_UID"] = df_target["Final_UID"]
        df_target["Final_Question"] = df_target["Matched_Question"]
        df_target["Final_Match_Type"] = df_target.apply(MatchingEngine._assign_match_type, axis=1)
        
        # Prevent UID assignment for heading questions
        df_target.loc[df_target["question_category"] == "Heading", ["Final_UID", "configured_final_UID"]] = None
        
        # Create change UID mapping
        df_target["Change_UID"] = df_target["Final_UID"].apply(
            lambda x: MatchingEngine._create_uid_mapping(x, df_reference)
        )
        
        # Handle choice inheritance
        df_target = MatchingEngine._handle_choice_inheritance(df_target)
        
        # Add survey ID/title column
        if "survey_id" in df_target.columns and "survey_title" in df_target.columns:
            df_target["survey_id_title"] = df_target.apply(
                lambda x: f"{x['survey_id']} - {x['survey_title']}" 
                if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                axis=1
            )
        
        return df_target

    @staticmethod
    def _assign_match_type(row: pd.Series) -> str:
        if pd.notnull(row["Suggested_UID"]):
            return row["Match_Confidence"]
        return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

    @staticmethod
    def _create_uid_mapping(uid: Any, df_reference: pd.DataFrame) -> Optional[str]:
        if pd.notnull(uid) and uid in df_reference["uid"].values:
            question = df_reference[df_reference['uid'] == uid]['heading_0'].iloc[0]
            return f"{uid} - {question}"
        return None

    @staticmethod
    def _handle_choice_inheritance(df_target: pd.DataFrame) -> pd.DataFrame:
        # Inherit UIDs from parent questions for choices
        for idx, row in df_target.iterrows():
            if row["is_choice"] and pd.notnull(row["parent_question"]):
                parent_rows = df_target[df_target["heading_0"] == row["parent_question"]]
                if not parent_rows.empty:
                    parent_uid = parent_rows["Final_UID"].iloc[0]
                    df_target.at[idx, "Final_UID"] = parent_uid
                    df_target.at[idx, "configured_final_UID"] = parent_uid
        
        return df_target

    @staticmethod
    def detect_uid_conflicts(df_target: pd.DataFrame) -> pd.DataFrame:
        uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
        duplicate_uids = uid_conflicts[uid_conflicts > 1].index
        df_target["UID_Conflict"] = df_target["Final_UID"].apply(
            lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
        )
        return df_target

class MetricsCalculator:
    """Handles metrics calculations"""
    
    @staticmethod
    def calculate_matched_percentage(df_final: pd.DataFrame) -> float:
        if df_final is None or df_final.empty:
            return 0.0
        
        # Filter main questions only
        df_main = df_final[df_final["is_choice"] == False].copy()
        
        # Apply exclusion filters
        privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
        html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
        html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
        
        eligible_questions = df_main[privacy_filter & html_filter]
        
        if eligible_questions.empty:
            return 0.0
        
        matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
        percentage = (len(matched_questions) / len(eligible_questions)) * 100
        
        return round(percentage, 2)

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def initialize_session_state():
        defaults = {
            "page": "home",
            "df_target": None,
            "df_final": None,
            "uid_changes": {},
            "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
            "df_reference": None,
            "survey_template": None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

class SurveyCategorizer:
    """Handles survey categorization based on title keywords"""
    
    CATEGORY_KEYWORDS = {
        "Application": ["application", "apply", "applying"],
        "Pre programme": ["pre programme", "pre-programme", "pre program", "pre-program", "preparation", "preparatory"],
        "Enrollment": ["enrollment", "enrolment", "enroll", "registration", "register"],
        "Progress Review": ["progress", "review", "checkpoint", "mid-point", "interim", "monitoring"],
        "Impact": ["impact", "outcome", "result", "effect", "consequence"],
        "GROW": ["GROW"],  # Exact match for CAPS
        "Feedback": ["feedback", "opinion", "evaluation", "rating", "comment"],
        "Pulse": ["pulse", "check-in", "quick survey", "temperature check"]
    }
    
    @staticmethod
    def categorize_survey(title: str) -> str:
        """Categorize survey based on title keywords"""
        title_lower = title.lower()
        
        # Check for exact GROW match first (case sensitive)
        if "GROW" in title:
            return "GROW"
        
        # Check other categories
        for category, keywords in SurveyCategorizer.CATEGORY_KEYWORDS.items():
            if category == "GROW":  # Skip GROW in this loop as it's handled above
                continue
            
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    return category
        
        return "Other"
    
    @staticmethod
    def analyze_survey_categories(surveys_data: List[Dict]) -> pd.DataFrame:
        """Analyze and categorize surveys"""
        categorized_surveys = []
        
        for survey in surveys_data:
            title = survey.get("title", "")
            survey_id = survey.get("id", "")
            category = SurveyCategorizer.categorize_survey(title)
            
            categorized_surveys.append({
                "survey_id": survey_id,
                "survey_title": title,
                "category": category
            })
        
        return pd.DataFrame(categorized_surveys)
    
    @staticmethod
    def extract_unique_questions_by_category(api: SurveyMonkeyAPI, df_categorized: pd.DataFrame) -> pd.DataFrame:
        """Extract unique questions grouped by category"""
        all_questions = []
        
        for category in df_categorized['category'].unique():
            category_surveys = df_categorized[df_categorized['category'] == category]
            category_questions = set()  # Use set to track unique questions
            
            for _, survey_row in category_surveys.iterrows():
                survey_id = survey_row['survey_id']
                
                try:
                    survey_json = api.get_survey_details(survey_id)
                    questions = api.extract_questions(survey_json)
                    
                    for question in questions:
                        # Create a unique identifier for the question
                        question_text = question.get('heading_0', '')
                        if question_text and question_text not in category_questions:
                            category_questions.add(question_text)
                            
                            question_data = question.copy()
                            question_data['category'] = category
                            question_data['Final_UID'] = None
                            question_data['configured_final_UID'] = None
                            question_data['Change_UID'] = None
                            
                            all_questions.append(question_data)
                
                except Exception as e:
                    logger.error(f"Failed to process survey {survey_id}: {e}")
                    continue
        
        return pd.DataFrame(all_questions)

class QuestionAnalyzer:
    """Handles question bank analysis and UID optimization"""
    
    @staticmethod
    def analyze_uid_distribution():
        """Analyze UID distribution across questions"""
        query = """
        SELECT 
            HEADING_0,
            UID,
            COUNT(*) as UID_COUNT
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL
        GROUP BY HEADING_0, UID
        ORDER BY HEADING_0, UID_COUNT DESC
        """
        return DataManager.run_snowflake_query(query)
    
    @staticmethod
    def find_optimal_uid_assignments(df_uid_distribution):
        """Find optimal UID assignments based on highest counts"""
        # Group by question and find the UID with highest count for each question
        optimal_assignments = []
        
        for question, group in df_uid_distribution.groupby('HEADING_0'):
            # Sort by count descending and take the top UID
            top_assignment = group.loc[group['UID_COUNT'].idxmax()]
            
            # Get all UIDs for this question to identify conflicts
            all_uids = group.to_dict('records')
            
            optimal_assignments.append({
                'QUESTION': question,
                'OPTIMAL_UID': top_assignment['UID'],
                'OPTIMAL_COUNT': top_assignment['UID_COUNT'],
                'TOTAL_ASSIGNMENTS': len(all_uids),
                'ALL_UIDS': all_uids,
                'HAS_CONFLICT': len(all_uids) > 1
            })
        
        return pd.DataFrame(optimal_assignments)
    
    @staticmethod
    def identify_uid_conflicts(df_optimal):
        """Identify UIDs that are assigned to multiple questions"""
        uid_usage = {}
        
        for _, row in df_optimal.iterrows():
            uid = row['OPTIMAL_UID']
            if uid not in uid_usage:
                uid_usage[uid] = []
            uid_usage[uid].append({
                'question': row['QUESTION'],
                'count': row['OPTIMAL_COUNT']
            })
        
        # Find UIDs assigned to multiple questions
        conflicts = []
        for uid, assignments in uid_usage.items():
            if len(assignments) > 1:
                conflicts.append({
                    'UID': uid,
                    'QUESTIONS_COUNT': len(assignments),
                    'ASSIGNMENTS': assignments,
                    'TOTAL_USAGE': sum(a['count'] for a in assignments)
                })
        
        return pd.DataFrame(conflicts)
    
    @staticmethod
    def create_optimized_question_bank(df_optimal, df_conflicts):
        """Create final optimized question bank resolving conflicts"""
        optimized_bank = []
        conflict_uids = set(df_conflicts['UID'].tolist()) if not df_conflicts.empty else set()
        
        for _, row in df_optimal.iterrows():
            uid = row['OPTIMAL_UID']
            question = row['QUESTION']
            
            # Mark if this UID has conflicts
            has_conflict = uid in conflict_uids
            conflict_severity = "High" if has_conflict and row['TOTAL_ASSIGNMENTS'] > 3 else "Low" if has_conflict else "None"
            
            optimized_bank.append({
                'UID': uid,
                'QUESTION': question,
                'USAGE_COUNT': row['OPTIMAL_COUNT'],
                'ALTERNATIVE_UIDS': len(row['ALL_UIDS']) - 1,
                'CONFLICT_STATUS': conflict_severity,
                'CONFIDENCE_SCORE': QuestionAnalyzer._calculate_confidence_score(row)
            })
        
        return pd.DataFrame(optimized_bank)
    
    @staticmethod
    def _calculate_confidence_score(row):
        """Calculate confidence score for UID assignment"""
        total_count = sum(uid_info['UID_COUNT'] for uid_info in row['ALL_UIDS'])
        optimal_ratio = row['OPTIMAL_COUNT'] / total_count if total_count > 0 else 0
        
        # Score based on dominance of the optimal UID
        if optimal_ratio >= 0.8:
            return "High"
        elif optimal_ratio >= 0.6:
            return "Medium"
        else:
            return "Low"

class PageRenderer:
    """Renders different pages of the application"""
    
    def __init__(self):
        self.ui = UIManager()
        
    def render_home_page(self):
        self.ui.render_header()
        
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Survey Management")
            if st.button("🔍 View Surveys", use_container_width=True):
                st.session_state.page = "view_surveys"
                st.rerun()
            if st.button("⚙️ Configure Survey", use_container_width=True):
                st.session_state.page = "configure_survey"
                st.rerun()
            if st.button("➕ Create New Survey", use_container_width=True):
                st.session_state.page = "create_survey"
                st.rerun()
        
        with col2:
            st.markdown("#### 🗄️ Question Bank")
            if st.button("📖 View Question Bank", use_container_width=True):
                st.session_state.page = "view_question_bank"
                st.rerun()
            if st.button("🔄 Update Question Bank", use_container_width=True):
                st.session_state.page = "update_question_bank"
                st.rerun()
        
        with col3:
            st.markdown("#### 📈 Analytics")
            if st.button("🎯 Final Unique QuestionBank", use_container_width=True):
                st.session_state.page = "final_unique_bank"
                st.rerun()
            if st.button("📋 Questions per Category", use_container_width=True):
                st.session_state.page = "questions_category"
                st.rerun()
            UIManager.render_metric("Total Surveys", "Loading...", "info")
            UIManager.render_metric("Match Rate", "Loading...", "success")
    
    def render_navigation(self):
        """Render navigation sidebar"""
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            pages = {
                "🏠 Home": "home",
                "📊 View Surveys": "view_surveys", 
                "⚙️ Configure Survey": "configure_survey",
                "📖 Question Bank": "view_question_bank",
                "🔄 Update Bank": "update_question_bank",
                "➕ Create Survey": "create_survey",
                "🎯 Final QuestionBank": "final_unique_bank",
                "📋 Questions per Category": "questions_category"
            }
            
            for page_name, page_key in pages.items():
                if st.button(page_name, use_container_width=True):
                    st.session_state.page = page_key
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### ℹ️ Application Info")
            st.info("UID Matcher Pro v2.0\nIntelligent question matching powered by AI")

def main():
    """Main application function"""
    
    # Initialize session state
    SessionManager.initialize_session_state()
    
    # Check secrets configuration
    if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
        st.error("❌ Missing secrets configuration for Snowflake or SurveyMonkey.")
        st.stop()
    
    # Initialize page renderer
    renderer = PageRenderer()
    
    # Render navigation
    renderer.render_navigation()
    
    # Route to appropriate page
    if st.session_state.page == "home":
        renderer.render_home_page()
    
    elif st.session_state.page == "view_surveys":
        render_view_surveys_page()
    
    elif st.session_state.page == "configure_survey":
        render_configure_survey_page()
    
    elif st.session_state.page == "view_question_bank":
        render_question_bank_page()
    
    elif st.session_state.page == "update_question_bank":
        render_update_question_bank_page()
    
    elif st.session_state.page == "create_survey":
        render_create_survey_page()
    
    elif st.session_state.page == "final_unique_bank":
        render_final_unique_bank_page()
    
    elif st.session_state.page == "questions_category":
        render_questions_category_page()

def render_view_surveys_page():
    """Render the view surveys page"""
    UIManager.render_header()
    st.markdown("## 📊 Survey Management")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token")
        if not token:
            st.error("❌ SurveyMonkey token missing in configuration.")
            return
        
        api = SurveyMonkeyAPI(token)
        
        with st.spinner("🔄 Fetching surveys..."):
            surveys = api.get_surveys()
        
        if not surveys:
            st.warning("⚠️ No surveys found.")
            return
        
        # Survey selection interface
        st.markdown("### 🎯 Select Surveys")
        
        col1, col2 = st.columns(2)
        
        with col1:
            survey_choices = {s["title"]: s["id"] for s in surveys}
            selected_survey = st.selectbox(
                "Choose by Title",
                [""] + list(survey_choices.keys()),
                index=0
            )
        
        with col2:
            survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
            survey_id_title_choices.sort(key=lambda x: int(x.split(" - ")[0]), reverse=True)
            
            selected_survey_ids = st.multiselect(
                "Select by ID/Title",
                survey_id_title_choices,
                default=[],
                help="Select one or more surveys"
            )
        
        # Process selected surveys
        all_selected_ids = []
        if selected_survey:
            all_selected_ids.append(survey_choices[selected_survey])
        
        all_selected_ids.extend([s.split(" - ")[0] for s in selected_survey_ids])
        all_selected_ids = list(set(all_selected_ids))
        
        if all_selected_ids:
            combined_questions = []
            
            progress_bar = st.progress(0)
            for i, survey_id in enumerate(all_selected_ids):
                with st.spinner(f"📥 Fetching survey {survey_id}..."):
                    survey_json = api.get_survey_details(survey_id)
                    questions = api.extract_questions(survey_json)
                    combined_questions.extend(questions)
                progress_bar.progress((i + 1) / len(all_selected_ids))
            
            st.session_state.df_target = pd.DataFrame(combined_questions)
            
            if st.session_state.df_target.empty:
                st.warning("⚠️ No questions found in selected surveys.")
            else:
                # Display options
                col1, col2 = st.columns(2)
                with col1:
                    show_main_only = st.checkbox("📝 Show main questions only", value=False)
                with col2:
                    UIManager.render_metric("Total Questions", str(len(st.session_state.df_target)), "info")
                
                # Prepare display data
                display_df = st.session_state.df_target.copy()
                if show_main_only:
                    display_df = display_df[display_df["is_choice"] == False]
                
                display_df["survey_id_title"] = display_df.apply(
                    lambda x: f"{x['survey_id']} - {x['survey_title']}" 
                    if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                    axis=1
                )
                
                # Enhanced dataframe display
                st.markdown("### 📋 Survey Questions")
                st.dataframe(
                    display_df[[
                        "survey_id_title", "heading_0", "position", "is_choice", 
                        "parent_question", "schema_type", "question_category"
                    ]],
                    column_config={
                        "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
                        "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                        "position": st.column_config.NumberColumn("Pos", width="small"),
                        "is_choice": st.column_config.CheckboxColumn("Choice", width="small"),
                        "parent_question": st.column_config.TextColumn("Parent", width="medium"),
                        "schema_type": st.column_config.TextColumn("Type", width="small"),
                        "question_category": st.column_config.TextColumn("Category", width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("👆 Select a survey above to view its questions.")
    
    except Exception as e:
        logger.error(f"Survey viewing failed: {e}")
        st.error(f"❌ Error loading surveys: {e}")

def render_configure_survey_page():
    """Render the configure survey page"""
    UIManager.render_header()
    st.markdown("## ⚙️ Survey Configuration & UID Matching")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token")
        if not token:
            st.error("❌ SurveyMonkey token missing in configuration.")
            return
        
        api = SurveyMonkeyAPI(token)
        
        # Survey selection section
        with st.expander("📊 Select Surveys", expanded=True):
            surveys = api.get_surveys()
            
            col1, col2 = st.columns(2)
            with col1:
                survey_choices = {s["title"]: s["id"] for s in surveys}
                selected_survey = st.selectbox("Choose Survey", [""] + list(survey_choices.keys()))
            
            with col2:
                survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
                survey_id_title_choices.sort(key=lambda x: int(x.split(" - ")[0]), reverse=True)
                selected_survey_ids = st.multiselect("Select Multiple", survey_id_title_choices)
        
        # Process selections
        all_selected_ids = []
        if selected_survey:
            all_selected_ids.append(survey_choices[selected_survey])
        all_selected_ids.extend([s.split(" - ")[0] for s in selected_survey_ids])
        all_selected_ids = list(set(all_selected_ids))
        
        if not all_selected_ids:
            st.info("👆 Please select at least one survey to begin configuration.")
            return
        
        # Main tabs for configuration workflow
        tab1, tab2, tab3 = st.tabs(["📝 Questions & Settings", "🎯 UID Matching", "📊 Final Configuration"])
        
        with tab1:
            render_questions_settings_tab(api, all_selected_ids)
        
        with tab2:
            render_uid_matching_tab()
        
        with tab3:
            render_final_configuration_tab()
    
    except Exception as e:
        logger.error(f"Configure survey failed: {e}")
        st.error(f"❌ Configuration error: {e}")

def render_questions_settings_tab(api: SurveyMonkeyAPI, survey_ids: List[str]):
    """Render questions and settings tab"""
    if not survey_ids:
        return
    
    # Fetch and process questions
    combined_questions = []
    progress_bar = st.progress(0)
    
    for i, survey_id in enumerate(survey_ids):
        with st.spinner(f"📥 Processing survey {survey_id}..."):
            survey_json = api.get_survey_details(survey_id)
            questions = api.extract_questions(survey_json)
            combined_questions.extend(questions)
        progress_bar.progress((i + 1) / len(survey_ids))
    
    st.session_state.df_target = pd.DataFrame(combined_questions)
    
    if st.session_state.df_target.empty:
        st.warning("⚠️ No questions found in selected surveys.")
        return
    
    # Initialize UID matching
    try:
        with st.spinner("🔍 Initializing UID matching..."):
            st.session_state.df_reference = DataManager.run_snowflake_query("""
                SELECT HEADING_0, MAX(UID) AS UID
                FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                WHERE UID IS NOT NULL
                GROUP BY HEADING_0
                LIMIT 10000
            """)
            
            # Run matching pipeline
            df_temp = MatchingEngine.compute_tfidf_matches(
                st.session_state.df_reference, 
                st.session_state.df_target
            )
            df_temp = MatchingEngine.compute_semantic_matches(st.session_state.df_reference, df_temp)
            st.session_state.df_final = MatchingEngine.finalize_matches(df_temp, st.session_state.df_reference)
            st.session_state.df_final = MatchingEngine.detect_uid_conflicts(st.session_state.df_final)
            
    except Exception as e:
        UIManager.show_warning("Snowflake connection failed. UID matching disabled.")
        st.session_state.df_reference = None
        st.session_state.df_final = st.session_state.df_target.copy()
        # Add empty columns for consistency
        for col in ["Final_UID", "configured_final_UID", "Change_UID"]:
            st.session_state.df_final[col] = None
    
    # Display questions with mandatory settings
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📝 Question Configuration")
    with col2:
        show_main_only = st.checkbox("📋 Main questions only", value=False)
    
    display_df = st.session_state.df_target.copy()
    if show_main_only:
        display_df = display_df[display_df["is_choice"] == False]
    
    display_df["survey_id_title"] = display_df.apply(
        lambda x: f"{x['survey_id']} - {x['survey_title']}" 
        if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
        axis=1
    )
    
    # Editable dataframe for mandatory settings
    edited_df = st.data_editor(
        display_df,
        column_config={
            "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
            "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
            "mandatory": st.column_config.CheckboxColumn("Required", help="Mark as mandatory"),
            "position": st.column_config.NumberColumn("Position", width="small"),
            "schema_type": st.column_config.TextColumn("Type", width="small"),
            "question_category": st.column_config.TextColumn("Category", width="small")
        },
        disabled=["survey_id_title", "heading_0", "position", "schema_type", "question_category"],
        hide_index=True,
        use_container_width=True
    )
    
    # Update session state with mandatory changes
    if not edited_df.empty:
        for idx, row in edited_df.iterrows():
            if idx < len(st.session_state.df_target):
                st.session_state.df_target.at[idx, "mandatory"] = row["mandatory"]

def render_uid_matching_tab():
    """Render UID matching and configuration tab"""
    if st.session_state.df_final is None:
        st.info("📝 Please configure questions in the first tab.")
        return
    
    # Metrics display
    matched_percentage = MetricsCalculator.calculate_matched_percentage(st.session_state.df_final)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        UIManager.render_metric("Match Rate", f"{matched_percentage}%", "success")
    with col2:
        total_questions = len(st.session_state.df_final[st.session_state.df_final["is_choice"] == False])
        UIManager.render_metric("Total Questions", str(total_questions), "info")
    with col3:
        matched_count = len(st.session_state.df_final[
            (st.session_state.df_final["is_choice"] == False) & 
            (st.session_state.df_final["Final_UID"].notna())
        ])
        UIManager.render_metric("Matched", str(matched_count), "primary")
    
    # Filters and search
    col1, col2 = st.columns(2)
    with col1:
        show_main_only = st.checkbox("📋 Main questions only", value=False, key="uid_main_only")
        match_filter = st.selectbox(
            "🔍 Filter by status",
            ["All", "Matched", "Not Matched", "High Confidence", "Low Confidence"]
        )
    
    with col2:
        search_query = st.text_input("🔎 Search questions", placeholder="Type to filter...")
    
    # Apply filters
    result_df = st.session_state.df_final.copy()
    
    if show_main_only:
        result_df = result_df[result_df["is_choice"] == False]
    
    if search_query:
        result_df = result_df[result_df["heading_0"].str.contains(search_query, case=False, na=False)]
    
    if match_filter != "All":
        filter_map = {
            "Matched": result_df["Final_UID"].notna(),
            "Not Matched": result_df["Final_UID"].isna(),
            "High Confidence": result_df["Match_Confidence"] == "✅ High",
            "Low Confidence": result_df["Match_Confidence"] == "⚠️ Low"
        }
        result_df = result_df[filter_map[match_filter]]
    
    # UID options for manual assignment
    uid_options = [None]
    if st.session_state.df_reference is not None:
        uid_options.extend([
            f"{row['UID']} - {row['HEADING_0']}" 
            for _, row in st.session_state.df_reference.iterrows()
        ])
    
    # Display editable matching results
    st.markdown("### 🎯 UID Matching Results")
    
    if result_df.empty:
        st.info("No questions match the current filters.")
    else:
        display_columns = [
            "heading_0", "Final_UID", "Match_Confidence", 
            "Similarity", "schema_type", "Change_UID"
        ]
        display_columns = [col for col in display_columns if col in result_df.columns]
        
        edited_matching_df = st.data_editor(
            result_df[display_columns],
            column_config={
                "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                "Match_Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Similarity": st.column_config.NumberColumn("Score", width="small", format="%.3f"),
                "schema_type": st.column_config.TextColumn("Type", width="small"),
                "Change_UID": st.column_config.SelectboxColumn(
                    "Reassign UID",
                    options=uid_options,
                    help="Select a different UID"
                )
            },
            disabled=["heading_0", "Final_UID", "Match_Confidence", "Similarity", "schema_type"],
            hide_index=True,
            use_container_width=True
        )
        
        # Process UID changes
        for idx, row in edited_matching_df.iterrows():
            if pd.notnull(row.get("Change_UID")):
                new_uid = row["Change_UID"].split(" - ")[0] if " - " in row["Change_UID"] else None
                if new_uid:
                    original_idx = result_df.index[idx]
                    st.session_state.df_final.at[original_idx, "Final_UID"] = new_uid
                    st.session_state.df_final.at[original_idx, "configured_final_UID"] = new_uid
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📝 Submit New Question")
        st.markdown("[Google Form](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    
    with col2:
        st.markdown("#### 🆔 Submit New UID")
        st.markdown("[Google Form](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")
    
    with col3:
        if st.button("🔄 Refresh Matching", use_container_width=True):
            st.rerun()

def render_final_configuration_tab():
    """Render final configuration and export tab"""
    if st.session_state.df_final is None:
        st.info("📝 Please complete UID matching in the previous tab.")
        return
    
    # Final metrics
    matched_percentage = MetricsCalculator.calculate_matched_percentage(st.session_state.df_final)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        UIManager.render_metric("Final Match Rate", f"{matched_percentage}%", "success")
    with col2:
        total_items = len(st.session_state.df_final)
        UIManager.render_metric("Total Items", str(total_items), "info")
    with col3:
        main_questions = len(st.session_state.df_final[st.session_state.df_final["is_choice"] == False])
        UIManager.render_metric("Main Questions", str(main_questions), "primary")
    with col4:
        choices = len(st.session_state.df_final[st.session_state.df_final["is_choice"] == True])
        UIManager.render_metric("Choices", str(choices), "warning")
    
    # Configuration preview
    st.markdown("### 📊 Final Configuration Preview")
    
    show_main_only = st.checkbox("📋 Show main questions only", value=True, key="final_main_only")
    
    config_df = st.session_state.df_final.copy()
    if show_main_only:
        config_df = config_df[config_df["is_choice"] == False]
    
    # Add survey info column
    if "survey_id" in config_df.columns and "survey_title" in config_df.columns:
        config_df["survey_info"] = config_df.apply(
            lambda x: f"{x['survey_id']} - {x['survey_title']}" 
            if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
            axis=1
        )
    
    display_columns = [
        "survey_info", "heading_0", "configured_final_UID", 
        "position", "mandatory", "schema_type", "question_category"
    ]
    display_columns = [col for col in display_columns if col in config_df.columns]
    
    st.dataframe(
        config_df[display_columns],
        column_config={
            "survey_info": st.column_config.TextColumn("Survey", width="medium"),
            "heading_0": st.column_config.TextColumn("Question", width="large"),
            "configured_final_UID": st.column_config.TextColumn("UID", width="medium"),
            "position": st.column_config.NumberColumn("Pos", width="small"),
            "mandatory": st.column_config.CheckboxColumn("Required", width="small"),
            "schema_type": st.column_config.TextColumn("Type", width="small"),
            "question_category": st.column_config.TextColumn("Category", width="small")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export data
        export_columns = [
            "survey_id", "survey_title", "heading_0", "configured_final_UID", 
            "position", "is_choice", "parent_question", "question_uid", 
            "schema_type", "mandatory", "question_category"
        ]
        export_columns = [col for col in export_columns if col in st.session_state.df_final.columns]
        export_df = st.session_state.df_final[export_columns].copy()
        export_df = export_df.rename(columns={"configured_final_UID": "uid"})
        
        # Download button
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"survey_configuration_{uuid4().hex[:8]}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Snowflake upload
        if st.button("🚀 Upload to Snowflake", use_container_width=True):
            try:
                with st.spinner("📤 Uploading to Snowflake..."):
                    engine = DataManager.get_snowflake_engine()
                    with engine.connect() as conn:
                        export_df.to_sql(
                            'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                            conn,
                            schema='DBT_SURVEY_MONKEY',
                            if_exists='append',
                            index=False
                        )
                UIManager.show_success("Data successfully uploaded to Snowflake!")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

def render_question_bank_page():
    """Render question bank page"""
    UIManager.render_header()
    st.markdown("## 📖 Question Bank")
    
    try:
        with st.spinner("📥 Loading question bank..."):
            df_reference = DataManager.run_snowflake_query("""
                SELECT HEADING_0, UID
                FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                WHERE UID IS NOT NULL
                ORDER BY UID
            """)
        
        if df_reference.empty:
            st.warning("⚠️ No data found in question bank.")
            return
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search questions", placeholder="Enter search term...")
        with col2:
            UIManager.render_metric("Total Questions", str(len(df_reference)), "info")
        
        # Apply search filter
        if search_term:
            df_reference = df_reference[
                df_reference["HEADING_0"].str.contains(search_term, case=False, na=False)
            ]
        
        # Display question bank
        st.dataframe(
            df_reference,
            column_config={
                "HEADING_0": st.column_config.TextColumn("Question Text", width="large"),
                "UID": st.column_config.TextColumn("UID", width="medium")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        if st.button("📥 Download Question Bank", use_container_width=True):
            csv_data = df_reference.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"question_bank_{uuid4().hex[:8]}.csv",
                "text/csv"
            )
    
    except Exception as e:
        logger.error(f"Question bank loading failed: {e}")
        st.error(f"❌ Error loading question bank: {e}")

def render_update_question_bank_page():
    """Render update question bank page"""
    UIManager.render_header()
    st.markdown("## 🔄 Update Question Bank")
    
    try:
        with st.spinner("📥 Loading data..."):
            # Get reference questions (with UIDs)
            df_reference = DataManager.run_snowflake_query("""
                SELECT HEADING_0, MAX(UID) AS UID
                FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                WHERE UID IS NOT NULL
                GROUP BY HEADING_0
                LIMIT 10000
            """)
            
            # Get target questions (without UIDs)
            df_target = DataManager.run_snowflake_query("""
                SELECT DISTINCT HEADING_0
                FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
            """)
        
        if df_reference.empty or df_target.empty:
            st.warning("⚠️ No data available for matching.")
            return
        
        # Run matching
        with st.spinner("🎯 Running UID matching..."):
            df_matched = MatchingEngine.compute_tfidf_matches(df_reference, df_target)
            df_matched = MatchingEngine.compute_semantic_matches(df_reference, df_matched)
            df_final = MatchingEngine.finalize_matches(df_matched, df_reference)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            UIManager.render_metric("Questions to Match", str(len(df_target)), "info")
        with col2:
            matched_count = len(df_final[df_final["Final_UID"].notna()])
            UIManager.render_metric("Successfully Matched", str(matched_count), "success")
        with col3:
            match_rate = (matched_count / len(df_target) * 100) if len(df_target) > 0 else 0
            UIManager.render_metric("Match Rate", f"{match_rate:.1f}%", "primary")
        
        # Filter options
        confidence_filter = st.multiselect(
            "🔍 Filter by match confidence",
            ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
            default=["✅ High", "⚠️ Low", "🧠 Semantic"]
        )
        
        filtered_df = df_final[df_final["Final_Match_Type"].isin(confidence_filter)]
        
        # Display results
        st.markdown("### 📊 Matching Results")
        st.dataframe(
            filtered_df[[
                "heading_0", "Final_UID", "Final_Match_Type", 
                "Similarity", "Semantic_Similarity"
            ]],
            column_config={
                "heading_0": st.column_config.TextColumn("Question", width="large"),
                "Final_UID": st.column_config.TextColumn("Matched UID", width="medium"),
                "Final_Match_Type": st.column_config.TextColumn("Match Type", width="small"),
                "Similarity": st.column_config.NumberColumn("TF-IDF Score", width="small", format="%.3f"),
                "Semantic_Similarity": st.column_config.NumberColumn("Semantic Score", width="small", format="%.3f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv_data,
                f"uid_matching_results_{uuid4().hex[:8]}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🚀 Apply Matches to Snowflake", use_container_width=True):
                st.info("🔧 This feature will be implemented to update the database with new UID matches.")
    
    except Exception as e:
        logger.error(f"Question bank update failed: {e}")
        st.error(f"❌ Error updating question bank: {e}")

def render_create_survey_page():
    """Render create survey page"""
    UIManager.render_header()
    st.markdown("## ➕ Create New Survey")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token")
        if not token:
            st.error("❌ SurveyMonkey token missing in configuration.")
            return
        
        api = SurveyMonkeyAPI(token)
        
        # Survey creation form
        with st.form("create_survey_form"):
            st.markdown("### 📝 Survey Details")
            
            col1, col2 = st.columns(2)
            with col1:
                survey_title = st.text_input("Survey Title *", placeholder="Enter survey title")
                survey_language = st.selectbox("Language", ["en", "es", "fr", "de"])
            
            with col2:
                survey_description = st.text_area("Description", placeholder="Optional survey description")
                num_pages = st.number_input("Number of Pages", min_value=1, max_value=10, value=1)
            
            # Dynamic page and question builder
            pages_data = []
            for page_num in range(num_pages):
                st.markdown(f"### 📄 Page {page_num + 1}")
                
                col1, col2 = st.columns(2)
                with col1:
                    page_title = st.text_input(f"Page Title", value=f"Page {page_num + 1}", key=f"page_title_{page_num}")
                with col2:
                    num_questions = st.number_input(
                        f"Questions on Page {page_num + 1}", 
                        min_value=1, max_value=10, value=1, 
                        key=f"num_questions_{page_num}"
                    )
                
                page_description = st.text_area(f"Page Description", key=f"page_desc_{page_num}")
                
                questions_data = []
                for q_num in range(num_questions):
                    with st.expander(f"❓ Question {q_num + 1}"):
                        question_text = st.text_input("Question Text *", key=f"q_text_{page_num}_{q_num}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            question_type = st.selectbox(
                                "Question Type",
                                ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                                key=f"q_type_{page_num}_{q_num}"
                            )
                        with col2:
                            is_required = st.checkbox("Required", key=f"q_required_{page_num}_{q_num}")
                        
                        # Question-type specific options
                        choices_data = []
                        
                        if question_type in ["Single Choice", "Multiple Choice"]:
                            num_choices = st.number_input(
                                "Number of Choices", 
                                min_value=2, max_value=10, value=3,
                                key=f"num_choices_{page_num}_{q_num}"
                            )
                            
                            for choice_num in range(num_choices):
                                choice_text = st.text_input(
                                    f"Choice {choice_num + 1}",
                                    key=f"choice_{page_num}_{q_num}_{choice_num}"
                                )
                                if choice_text:
                                    choices_data.append({"text": choice_text, "position": choice_num + 1})
                        
                        elif question_type == "Matrix":
                            col1, col2 = st.columns(2)
                            with col1:
                                num_rows = st.number_input(
                                    "Number of Rows", 
                                    min_value=2, max_value=10, value=3,
                                    key=f"num_rows_{page_num}_{q_num}"
                                )
                            with col2:
                                num_rating_choices = st.number_input(
                                    "Rating Scale Size", 
                                    min_value=3, max_value=10, value=5,
                                    key=f"num_rating_{page_num}_{q_num}"
                                )
                            
                            rows_data = []
                            for row_num in range(num_rows):
                                row_text = st.text_input(
                                    f"Row {row_num + 1}",
                                    key=f"row_{page_num}_{q_num}_{row_num}"
                                )
                                if row_text:
                                    rows_data.append({"text": row_text, "position": row_num + 1})
                            
                            for choice_num in range(num_rating_choices):
                                choice_text = st.text_input(
                                    f"Rating {choice_num + 1}",
                                    key=f"rating_{page_num}_{q_num}_{choice_num}"
                                )
                                if choice_text:
                                    choices_data.append({"text": choice_text, "position": choice_num + 1})
                        
                        if question_text:
                            question_data = {
                                "heading": question_text,
                                "family": question_type.lower().replace(" ", "_"),
                                "subtype": "vertical" if question_type != "Open-Ended" else "essay",
                                "position": q_num + 1,
                                "is_required": is_required,
                                "choices": choices_data if question_type != "Open-Ended" else None
                            }
                            
                            if question_type == "Matrix" and 'rows_data' in locals():
                                question_data["rows"] = rows_data
                            
                            questions_data.append(question_data)
                
                if questions_data:
                    pages_data.append({
                        "title": page_title,
                        "description": page_description,
                        "questions": questions_data
                    })
            
            # Survey settings
            st.markdown("### ⚙️ Survey Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_progress_bar = st.checkbox("Progress Bar", value=True)
            with col2:
                hide_asterisks = st.checkbox("Hide Required Asterisks", value=False)
            with col3:
                one_question_at_a_time = st.checkbox("One Question per Page", value=False)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create Survey", use_container_width=True)
            
            if submitted:
                if not survey_title or not pages_data:
                    st.error("❌ Survey title and at least one page with questions are required.")
                else:
                    # Create survey template
                    survey_template = {
                        "title": survey_title,
                        "nickname": survey_title,
                        "language": survey_language,
                        "pages": pages_data,
                        "settings": {
                            "progress_bar": show_progress_bar,
                            "hide_asterisks": hide_asterisks,
                            "one_question_at_a_time": one_question_at_a_time
                        }
                    }
                    
                    try:
                        with st.spinner("🔨 Creating survey..."):
                            # Create survey
                            survey_id = api.create_survey({
                                "title": survey_template["title"],
                                "nickname": survey_template.get("nickname", survey_template["title"]),
                                "language": survey_template.get("language", "en")
                            })
                            
                            # Create pages and questions
                            for page_template in survey_template["pages"]:
                                page_id = api._make_request("POST", f"surveys/{survey_id}/pages", json={
                                    "title": page_template.get("title", ""),
                                    "description": page_template.get("description", "")
                                }).get("id")
                                
                                for question_template in page_template["questions"]:
                                    payload = {
                                        "family": question_template["family"],
                                        "subtype": question_template["subtype"],
                                        "headings": [{"heading": question_template["heading"]}],
                                        "position": question_template["position"],
                                        "required": question_template.get("is_required", False)
                                    }
                                    
                                    if question_template.get("choices"):
                                        payload["answers"] = {"choices": question_template["choices"]}
                                    
                                    if question_template["family"] == "matrix" and question_template.get("rows"):
                                        payload["answers"] = {
                                            "rows": question_template["rows"],
                                            "choices": question_template["choices"]
                                        }
                                    
                                    api._make_request("POST", f"surveys/{survey_id}/pages/{page_id}/questions", json=payload)
                        
                        UIManager.show_success(f"Survey created successfully! Survey ID: {survey_id}")
                        
                        # Show preview
                        st.markdown("### 📋 Survey Preview")
                        st.json(survey_template)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to create survey: {e}")
    
    except Exception as e:
        logger.error(f"Survey creation failed: {e}")
        st.error(f"❌ Error in survey creation: {e}")

def render_final_unique_bank_page():
    """Render final unique question bank analysis page"""
    UIManager.render_header()
    st.markdown("## 🎯 Final Unique QuestionBank")
    st.markdown("### 🔍 UID Distribution Analysis & Optimization")
    
    try:
        # Load and analyze data
        with st.spinner("📊 Analyzing UID distribution..."):
            df_uid_distribution = QuestionAnalyzer.analyze_uid_distribution()
            
            if df_uid_distribution.empty:
                st.warning("⚠️ No UID distribution data found.")
                return
            
            # Find optimal assignments
            df_optimal = QuestionAnalyzer.find_optimal_uid_assignments(df_uid_distribution)
            df_conflicts = QuestionAnalyzer.identify_uid_conflicts(df_optimal)
            df_optimized_bank = QuestionAnalyzer.create_optimized_question_bank(df_optimal, df_conflicts)
        
        # Main dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_questions = len(df_optimal)
            UIManager.render_metric("Total Questions", str(total_questions), "info")
        
        with col2:
            questions_with_conflicts = len(df_optimal[df_optimal['HAS_CONFLICT'] == True])
            UIManager.render_metric("Questions with Conflicts", str(questions_with_conflicts), "warning")
        
        with col3:
            unique_uids = len(df_optimal['OPTIMAL_UID'].unique())
            UIManager.render_metric("Unique UIDs", str(unique_uids), "primary")
        
        with col4:
            conflict_rate = (questions_with_conflicts / total_questions * 100) if total_questions > 0 else 0
            UIManager.render_metric("Conflict Rate", f"{conflict_rate:.1f}%", "warning")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 UID Distribution", 
            "🎯 Optimal Assignments", 
            "⚠️ Conflict Analysis", 
            "📖 Optimized Question Bank"
        ])
        
        with tab1:
            render_uid_distribution_tab(df_uid_distribution, df_optimal)
        
        with tab2:
            render_optimal_assignments_tab(df_optimal)
        
        with tab3:
            render_conflict_analysis_tab(df_conflicts, df_optimal)
        
        with tab4:
            render_optimized_bank_tab(df_optimized_bank)
    
    except Exception as e:
        logger.error(f"Final unique bank analysis failed: {e}")
        st.error(f"❌ Error in analysis: {e}")

def render_questions_category_page():
    """Render questions per survey category page"""
    UIManager.render_header()
    st.markdown("## 📋 Questions per Survey Category")
    st.markdown("### 🏷️ Categorized Survey Analysis & UID Assignment")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token")
        if not token:
            st.error("❌ SurveyMonkey token missing in configuration.")
            return
        
        api = SurveyMonkeyAPI(token)
        
        # Load surveys and categorize
        with st.spinner("📊 Loading and categorizing surveys..."):
            surveys = api.get_surveys()
            if not surveys:
                st.warning("⚠️ No surveys found.")
                return
            
            df_categorized = SurveyCategorizer.analyze_survey_categories(surveys)
        
        # Display category distribution
        st.markdown("### 📈 Survey Category Distribution")
        
        category_counts = df_categorized['category'].value_counts()
        
        # Create columns for category metrics
        categories = list(SurveyCategorizer.CATEGORY_KEYWORDS.keys()) + ["Other"]
        cols = st.columns(min(len(categories), 4))
        
        for i, category in enumerate(categories):
            col_idx = i % len(cols)
            with cols[col_idx]:
                count = category_counts.get(category, 0)
                color = "success" if count > 0 else "info"
                UIManager.render_metric(category, str(count), color)
        
        # Category selection and analysis
        st.markdown("### 🎯 Category Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_categories = st.multiselect(
                "Select categories to analyze",
                options=category_counts.index.tolist(),
                default=category_counts.index.tolist()[:3] if len(category_counts) > 0 else []
            )
        
        with col2:
            show_unique_only = st.checkbox("Show unique questions only", value=True)
        
        if not selected_categories:
            st.info("👆 Please select at least one category to analyze.")
            return
        
        # Filter categorized surveys
        filtered_surveys = df_categorized[df_categorized['category'].isin(selected_categories)]
        
        # Extract questions for selected categories
        with st.spinner("🔄 Extracting questions from selected categories..."):
            df_category_questions = SurveyCategorizer.extract_unique_questions_by_category(api, filtered_surveys)
        
        if df_category_questions.empty:
            st.warning("⚠️ No questions found in selected categories.")
            return
        
        # Load reference data for UID matching
        try:
            with st.spinner("🔍 Loading UID reference data..."):
                st.session_state.df_reference = DataManager.run_snowflake_query("""
                    SELECT HEADING_0, MAX(UID) AS UID
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    WHERE UID IS NOT NULL
                    GROUP BY HEADING_0
                    LIMIT 10000
                """)
                
                # Run UID matching
                if not st.session_state.df_reference.empty:
                    df_temp = MatchingEngine.compute_tfidf_matches(
                        st.session_state.df_reference, 
                        df_category_questions
                    )
                    df_temp = MatchingEngine.compute_semantic_matches(st.session_state.df_reference, df_temp)
                    df_category_questions = MatchingEngine.finalize_matches(df_temp, st.session_state.df_reference)
                
        except Exception as e:
            UIManager.show_warning("Snowflake connection failed. UID matching disabled.")
            st.session_state.df_reference = None
        
        # Store in session state
        st.session_state.df_category_questions = df_category_questions
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Category Overview",
            "📝 Questions & UID Assignment", 
            "📋 Final Configuration"
        ])
        
        with tab1:
            render_category_overview_tab(df_categorized, df_category_questions, selected_categories)
        
        with tab2:
            render_category_uid_assignment_tab(df_category_questions, show_unique_only)
        
        with tab3:
            render_category_final_config_tab(df_category_questions)
    
    except Exception as e:
        logger.error(f"Questions category analysis failed: {e}")
        st.error(f"❌ Error in category analysis: {e}")

# Helper functions for Final Unique QuestionBank tabs
def render_uid_distribution_tab(df_uid_distribution, df_optimal):
    """Render UID distribution analysis tab"""
    st.markdown("### 📊 UID Distribution Analysis")
    
    # Search and filter options
    col1, col2 = st.columns(2)
    with col1:
        search_question = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
    with col2:
        min_count = st.number_input("Minimum UID count", min_value=1, value=1)
    
    # Apply filters
    filtered_df = df_uid_distribution.copy()
    if search_question:
        filtered_df = filtered_df[filtered_df['HEADING_0'].str.contains(search_question, case=False, na=False)]
    filtered_df = filtered_df[filtered_df['UID_COUNT'] >= min_count]
    
    # Summary statistics
    st.markdown("#### 📈 Distribution Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_count = filtered_df['UID_COUNT'].mean() if not filtered_df.empty else 0
        UIManager.render_metric("Average UID Count", f"{avg_count:.1f}", "info")
    
    with col2:
        max_count = filtered_df['UID_COUNT'].max() if not filtered_df.empty else 0
        UIManager.render_metric("Max UID Count", str(max_count), "success")
    
    with col3:
        total_records = len(filtered_df)
        UIManager.render_metric("Total Records", str(total_records), "primary")
    
    # Detailed distribution table
    st.markdown("#### 📋 Detailed UID Distribution")
    
    if filtered_df.empty:
        st.info("No data matches the current filters.")
    else:
        # Add ranking and optimal UID indicator
        display_df = filtered_df.copy()
        
        # Mark optimal UIDs
        optimal_uid_map = dict(zip(df_optimal['QUESTION'], df_optimal['OPTIMAL_UID']))
        display_df['IS_OPTIMAL'] = display_df.apply(
            lambda row: "✅ Optimal" if optimal_uid_map.get(row['HEADING_0']) == row['UID'] else "❌ Alternative",
            axis=1
        )
        
        # Sort by question and count
        display_df = display_df.sort_values(['HEADING_0', 'UID_COUNT'], ascending=[True, False])
        
        st.dataframe(
            display_df,
            column_config={
                "HEADING_0": st.column_config.TextColumn("Question", width="large"),
                "UID": st.column_config.TextColumn("UID", width="medium"),
                "UID_COUNT": st.column_config.NumberColumn("Usage Count", width="small"),
                "IS_OPTIMAL": st.column_config.TextColumn("Status", width="small")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Distribution Data",
            csv_data,
            f"uid_distribution_{uuid4().hex[:8]}.csv",
            "text/csv"
        )

def render_optimal_assignments_tab(df_optimal):
    """Render optimal UID assignments tab"""
    st.markdown("### 🎯 Optimal UID Assignments")
    st.markdown("*Based on highest usage count per question*")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        conflict_filter = st.selectbox(
            "Filter by conflict status",
            ["All", "With Conflicts", "No Conflicts"]
        )
    with col2:
        search_optimal = st.text_input("🔍 Search questions", placeholder="Filter optimal assignments...")
    
    # Apply filters
    filtered_optimal = df_optimal.copy()
    
    if conflict_filter == "With Conflicts":
        filtered_optimal = filtered_optimal[filtered_optimal['HAS_CONFLICT'] == True]
    elif conflict_filter == "No Conflicts":
        filtered_optimal = filtered_optimal[filtered_optimal['HAS_CONFLICT'] == False]
    
    if search_optimal:
        filtered_optimal = filtered_optimal[filtered_optimal['QUESTION'].str.contains(search_optimal, case=False, na=False)]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clean_assignments = len(filtered_optimal[filtered_optimal['HAS_CONFLICT'] == False])
        UIManager.render_metric("Clean Assignments", str(clean_assignments), "success")
    
    with col2:
        conflicted_assignments = len(filtered_optimal[filtered_optimal['HAS_CONFLICT'] == True])
        UIManager.render_metric("Conflicted Assignments", str(conflicted_assignments), "warning")
    
    with col3:
        avg_alternatives = filtered_optimal['TOTAL_ASSIGNMENTS'].mean() - 1 if not filtered_optimal.empty else 0
        UIManager.render_metric("Avg Alternatives", f"{avg_alternatives:.1f}", "info")
    
    # Display optimal assignments
    st.markdown("#### 📊 Optimal Assignment Details")
    
    if filtered_optimal.empty:
        st.info("No assignments match the current filters.")
    else:
        # Prepare display data
        display_df = filtered_optimal.copy()
        display_df['CONFLICT_INDICATOR'] = display_df['HAS_CONFLICT'].apply(
            lambda x: "⚠️ Has Conflicts" if x else "✅ Clean"
        )
        
        # Calculate dominance percentage
        display_df['DOMINANCE_PCT'] = display_df.apply(
            lambda row: f"{(row['OPTIMAL_COUNT'] / sum(uid['UID_COUNT'] for uid in row['ALL_UIDS']) * 100):.1f}%",
            axis=1
        )
        
        st.dataframe(
            display_df[['QUESTION', 'OPTIMAL_UID', 'OPTIMAL_COUNT', 'TOTAL_ASSIGNMENTS', 'DOMINANCE_PCT', 'CONFLICT_INDICATOR']],
            column_config={
                "QUESTION": st.column_config.TextColumn("Question", width="large"),
                "OPTIMAL_UID": st.column_config.TextColumn("Optimal UID", width="medium"),
                "OPTIMAL_COUNT": st.column_config.NumberColumn("Usage Count", width="small"),
                "TOTAL_ASSIGNMENTS": st.column_config.NumberColumn("Total UIDs", width="small"),
                "DOMINANCE_PCT": st.column_config.TextColumn("Dominance %", width="small"),
                "CONFLICT_INDICATOR": st.column_config.TextColumn("Status", width="medium")
            },
            use_container_width=True,
            hide_index=True
        )

def render_conflict_analysis_tab(df_conflicts, df_optimal):
    """Render conflict analysis dashboard"""
    st.markdown("### ⚠️ UID Conflict Analysis")
    st.markdown("*UIDs assigned to multiple questions*")
    
    if df_conflicts.empty:
        UIManager.show_success("🎉 No UID conflicts found! All UIDs have unique question assignments.")
        return
    
    # Conflict severity analysis
    high_severity_conflicts = len(df_conflicts[df_conflicts['QUESTIONS_COUNT'] >= 3])
    medium_severity_conflicts = len(df_conflicts[df_conflicts['QUESTIONS_COUNT'] == 2])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        UIManager.render_metric("Total Conflicted UIDs", str(len(df_conflicts)), "warning")
    
    with col2:
        UIManager.render_metric("High Severity (3+ questions)", str(high_severity_conflicts), "warning")
    
    with col3:
        UIManager.render_metric("Medium Severity (2 questions)", str(medium_severity_conflicts), "info")
    
    # Add these functions to the end of your uid_pro21.py file (before the if __name__ == "__main__":)

        with st.expander(f"{severity_color} UID {uid} - {questions_count} questions ({total_usage} total usage)"):
            assignments_df = pd.DataFrame(conflict['ASSIGNMENTS'])
            assignments_df = assignments_df.sort_values('count', ascending=False)
            
            # Show which question should get this UID (highest count)
            recommended_question = assignments_df.iloc[0]['question']
            recommended_count = assignments_df.iloc[0]['count']
            
            st.markdown(f"**Recommended Assignment:** {recommended_question} ({recommended_count} uses)")
            
            # Show all assignments
            st.markdown("**All Current Assignments:**")
            for _, assignment in assignments_df.iterrows():
                percentage = (assignment['count'] / total_usage * 100)
                icon = "👑" if assignment['question'] == recommended_question else "🔄"
                st.write(f"{icon} {assignment['question']} - {assignment['count']} uses ({percentage:.1f}%)")
    
    # Resolution recommendations
    st.markdown("#### 💡 Resolution Recommendations")
    
    resolution_data = []
    for _, conflict in df_conflicts.iterrows():
        uid = conflict['UID']
        assignments = pd.DataFrame(conflict['ASSIGNMENTS']).sort_values('count', ascending=False)
        
        recommended = assignments.iloc[0]
        alternatives = assignments.iloc[1:]
        
        resolution_data.append({
            'UID': uid,
            'RECOMMENDED_QUESTION': recommended['question'],
            'RECOMMENDED_COUNT': recommended['count'],
            'ALTERNATIVES_NEEDED': len(alternatives),
            'SEVERITY': 'High' if len(assignments) >= 3 else 'Medium'
        })
    
    resolution_df = pd.DataFrame(resolution_data)
    
    st.dataframe(
        resolution_df,
        column_config={
            "UID": st.column_config.TextColumn("Conflicted UID", width="medium"),
            "RECOMMENDED_QUESTION": st.column_config.TextColumn("Keep for Question", width="large"),
            "RECOMMENDED_COUNT": st.column_config.NumberColumn("Usage Count", width="small"),
            "ALTERNATIVES_NEEDED": st.column_config.NumberColumn("Need New UIDs", width="small"),
            "SEVERITY": st.column_config.TextColumn("Severity", width="small")
        },
        use_container_width=True,
        hide_index=True
    )

def render_optimized_bank_tab(df_optimized_bank):
    """Render the final optimized question bank"""
    st.markdown("### 📖 Optimized Question Bank")
    st.markdown("*Final recommended UID assignments with conflict resolution*")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = len(df_optimized_bank)
        UIManager.render_metric("Total Questions", str(total_questions), "info")
    
    with col2:
        high_confidence = len(df_optimized_bank[df_optimized_bank['CONFIDENCE_SCORE'] == 'High'])
        UIManager.render_metric("High Confidence", str(high_confidence), "success")
    
    with col3:
        no_conflicts = len(df_optimized_bank[df_optimized_bank['CONFLICT_STATUS'] == 'None'])
        UIManager.render_metric("No Conflicts", str(no_conflicts), "success")
    
    with col4:
        avg_usage = df_optimized_bank['USAGE_COUNT'].mean() if not df_optimized_bank.empty else 0
        UIManager.render_metric("Avg Usage", f"{avg_usage:.0f}", "primary")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_filter = st.selectbox(
            "Filter by confidence",
            ["All", "High", "Medium", "Low"]
        )
    
    with col2:
        conflict_filter = st.selectbox(
            "Filter by conflicts",
            ["All", "None", "Low", "High"]
        )
    
    with col3:
        search_bank = st.text_input("🔍 Search questions", placeholder="Filter question bank...")
    
    # Apply filters
    filtered_bank = df_optimized_bank.copy()
    
    if confidence_filter != "All":
        filtered_bank = filtered_bank[filtered_bank['CONFIDENCE_SCORE'] == confidence_filter]
    
    if conflict_filter != "All":
        filtered_bank = filtered_bank[filtered_bank['CONFLICT_STATUS'] == conflict_filter]
    
    if search_bank:
        filtered_bank = filtered_bank[filtered_bank['QUESTION'].str.contains(search_bank, case=False, na=False)]
    
    # Display optimized question bank
    st.markdown("#### 📊 Final Question Bank")
    
    if filtered_bank.empty:
        st.info("No questions match the current filters.")
    else:
        # Add status indicators
        display_df = filtered_bank.copy()
        display_df['STATUS_INDICATOR'] = display_df.apply(
            lambda row: f"{row['CONFIDENCE_SCORE']} Confidence | {row['CONFLICT_STATUS']} Conflict",
            axis=1
        )
        
        # Color code based on status
        def get_status_color(conflict_status, confidence):
            if conflict_status == "None" and confidence == "High":
                return "✅ Excellent"
            elif conflict_status == "Low" and confidence in ["High", "Medium"]:
                return "⚠️ Good"
            elif conflict_status == "High" or confidence == "Low":
                return "🔴 Needs Review"
            else:
                return "🟡 Acceptable"
        
        display_df['OVERALL_STATUS'] = display_df.apply(
            lambda row: get_status_color(row['CONFLICT_STATUS'], row['CONFIDENCE_SCORE']),
            axis=1
        )
        
        st.dataframe(
            display_df[['UID', 'QUESTION', 'USAGE_COUNT', 'ALTERNATIVE_UIDS', 'CONFIDENCE_SCORE', 'CONFLICT_STATUS', 'OVERALL_STATUS']],
            column_config={
                "UID": st.column_config.TextColumn("UID", width="medium"),
                "QUESTION": st.column_config.TextColumn("Question", width="large"),
                "USAGE_COUNT": st.column_config.NumberColumn("Usage Count", width="small"),
                "ALTERNATIVE_UIDS": st.column_config.NumberColumn("Alt UIDs", width="small"),
                "CONFIDENCE_SCORE": st.column_config.TextColumn("Confidence", width="small"),
                "CONFLICT_STATUS": st.column_config.TextColumn("Conflict Level", width="small"),
                "OVERALL_STATUS": st.column_config.TextColumn("Overall Status", width="medium")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Optimized Bank",
                csv_data,
                f"optimized_question_bank_{uuid4().hex[:8]}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🚀 Deploy to Production", use_container_width=True):
                st.info("🔧 This feature will implement the optimized UID assignments in the production database.")

# Helper functions for Questions per Category tabs
def render_category_overview_tab(df_categorized, df_category_questions, selected_categories):
    """Render category overview tab"""
    st.markdown("### 📊 Category Analysis Overview")
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏷️ Survey Distribution by Category")
        category_summary = df_categorized['category'].value_counts().reset_index()
        category_summary.columns = ['Category', 'Survey Count']
        
        st.dataframe(
            category_summary,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Survey Count": st.column_config.NumberColumn("Surveys", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📝 Questions by Selected Categories")
        if not df_category_questions.empty:
            questions_by_category = df_category_questions.groupby('category').agg({
                'heading_0': 'count',
                'is_choice': lambda x: (x == False).sum()  # Count main questions
            }).reset_index()
            questions_by_category.columns = ['Category', 'Total Items', 'Main Questions']
            
            st.dataframe(
                questions_by_category,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Total Items": st.column_config.NumberColumn("Total Items", width="small"),
                    "Main Questions": st.column_config.NumberColumn("Main Questions", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
    
    # Detailed surveys by category
    st.markdown("#### 🔍 Detailed Survey Breakdown")
    
    for category in selected_categories:
        with st.expander(f"📂 {category} Surveys"):
            category_surveys = df_categorized[df_categorized['category'] == category]
            
            if category_surveys.empty:
                st.info(f"No surveys found in {category} category.")
            else:
                st.dataframe(
                    category_surveys[['survey_id', 'survey_title']],
                    column_config={
                        "survey_id": st.column_config.TextColumn("Survey ID", width="medium"),
                        "survey_title": st.column_config.TextColumn("Survey Title", width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )

def render_category_uid_assignment_tab(df_category_questions, show_unique_only):
    """Render UID assignment tab for categorized questions"""
    st.markdown("### 📝 Questions & UID Assignment")
    
    if df_category_questions.empty:
        st.info("No questions available for UID assignment.")
        return
    
    # Calculate matching metrics
    matched_percentage = MetricsCalculator.calculate_matched_percentage(df_category_questions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = len(df_category_questions[df_category_questions['is_choice'] == False])
        UIManager.render_metric("Main Questions", str(total_questions), "info")
    
    with col2:
        total_choices = len(df_category_questions[df_category_questions['is_choice'] == True])
        UIManager.render_metric("Choices", str(total_choices), "primary")
    
    with col3:
        UIManager.render_metric("Match Rate", f"{matched_percentage}%", "success")
    
    with col4:
        unique_categories = len(df_category_questions['category'].unique())
        UIManager.render_metric("Categories", str(unique_categories), "warning")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox(
            "Filter by category",
            ["All"] + list(df_category_questions['category'].unique())
        )
    
    with col2:
        match_filter = st.selectbox(
            "Filter by match status",
            ["All", "Matched", "Not Matched", "High Confidence", "Low Confidence"]
        )
    
    with col3:
        search_question = st.text_input("🔍 Search questions", placeholder="Type to filter...")
    
    # Apply filters
    display_df = df_category_questions.copy()
    
    if show_unique_only:
        display_df = display_df[display_df['is_choice'] == False]
    
    if category_filter != "All":
        display_df = display_df[display_df['category'] == category_filter]
    
    if search_question:
        display_df = display_df[display_df['heading_0'].str.contains(search_question, case=False, na=False)]
    
    if match_filter != "All":
        filter_map = {
            "Matched": display_df["Final_UID"].notna(),
            "Not Matched": display_df["Final_UID"].isna(),
            "High Confidence": display_df.get("Match_Confidence", pd.Series()) == "✅ High",
            "Low Confidence": display_df.get("Match_Confidence", pd.Series()) == "⚠️ Low"
        }
        if match_filter in filter_map:
            display_df = display_df[filter_map[match_filter]]
    
    # UID options for manual assignment
    uid_options = [None]
    if st.session_state.df_reference is not None:
        uid_options.extend([
            f"{row['UID']} - {row['HEADING_0']}" 
            for _, row in st.session_state.df_reference.iterrows()
        ])
    
    # Display editable questions
    st.markdown("#### 🎯 Question UID Assignment")
    
    if display_df.empty:
        st.info("No questions match the current filters.")
    else:
        # Prepare columns for display
        display_columns = [
            "category", "heading_0", "Final_UID", "schema_type", 
            "mandatory", "question_category", "Change_UID"
        ]
        
        # Add confidence column if available
        if "Match_Confidence" in display_df.columns:
            display_columns.insert(3, "Match_Confidence")
        
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        edited_df = st.data_editor(
            display_df[display_columns],
            column_config={
                "category": st.column_config.TextColumn("Category", width="small"),
                "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                "Match_Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "schema_type": st.column_config.TextColumn("Type", width="small"),
                "mandatory": st.column_config.CheckboxColumn("Required", width="small"),
                "question_category": st.column_config.TextColumn("Q Category", width="small"),
                "Change_UID": st.column_config.SelectboxColumn(
                    "Assign UID",
                    options=uid_options,
                    help="Select or change UID assignment"
                )
            },
            disabled=["category", "heading_0", "Final_UID", "Match_Confidence", "schema_type", "question_category"],
            hide_index=True,
            use_container_width=True
        )
        
        # Process UID changes
        for idx, row in edited_df.iterrows():
            if pd.notnull(row.get("Change_UID")):
                new_uid = row["Change_UID"].split(" - ")[0] if " - " in row["Change_UID"] else None
                if new_uid:
                    original_idx = display_df.index[idx]
                    st.session_state.df_category_questions.at[original_idx, "Final_UID"] = new_uid
                    st.session_state.df_category_questions.at[original_idx, "configured_final_UID"] = new_uid
            
            # Update mandatory status
            if "mandatory" in row and idx < len(display_df):
                original_idx = display_df.index[idx]
                st.session_state.df_category_questions.at[original_idx, "mandatory"] = row["mandatory"]

def render_category_final_config_tab(df_category_questions):
    """Render final configuration tab for categorized questions"""
    st.markdown("### 📋 Final Category Configuration")
    
    if df_category_questions.empty:
        st.info("No configuration data available.")
        return
    
    # Final metrics
    matched_percentage = MetricsCalculator.calculate_matched_percentage(df_category_questions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        UIManager.render_metric("Final Match Rate", f"{matched_percentage}%", "success")
    
    with col2:
        total_items = len(df_category_questions)
        UIManager.render_metric("Total Items", str(total_items), "info")
    
    with col3:
        configured_items = len(df_category_questions[df_category_questions["Final_UID"].notna()])
        UIManager.render_metric("Configured Items", str(configured_items), "primary")
    
    with col4:
        categories_count = len(df_category_questions['category'].unique())
        UIManager.render_metric("Categories", str(categories_count), "warning")
    
    # Configuration preview by category
    st.markdown("#### 📊 Configuration by Category")
    
    for category in df_category_questions['category'].unique():
        with st.expander(f"📂 {category} Configuration"):
            category_data = df_category_questions[df_category_questions['category'] == category]
            
            # Show summary
            total_in_category = len(category_data)
            configured_in_category = len(category_data[category_data["Final_UID"].notna()])
            main_questions = len(category_data[category_data["is_choice"] == False])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", total_in_category)
            with col2:
                st.metric("Configured", configured_in_category)
            with col3:
                st.metric("Main Questions", main_questions)
            
            # Show detailed data
            display_columns = [
                "heading_0", "Final_UID", "is_choice", "mandatory", 
                "schema_type", "question_category"
            ]
            display_columns = [col for col in display_columns if col in category_data.columns]
            
            st.dataframe(
                category_data[display_columns],
                column_config={
                    "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                    "Final_UID": st.column_config.TextColumn("UID", width="medium"),
                    "is_choice": st.column_config.CheckboxColumn("Choice", width="small"),
                    "mandatory": st.column_config.CheckboxColumn("Required", width="small"),
                    "schema_type": st.column_config.TextColumn("Type", width="small"),
                    "question_category": st.column_config.TextColumn("Category", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
    
    # Export options
    st.markdown("#### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export data
        export_columns = [
            "category", "heading_0", "Final_UID", "is_choice", "mandatory",
            "schema_type", "question_category", "position"
        ]
        export_columns = [col for col in export_columns if col in df_category_questions.columns]
        export_df = df_category_questions[export_columns].copy()
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "📥 Download Category Configuration",
            csv_data,
            f"category_questions_config_{uuid4().hex[:8]}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🚀 Deploy Configuration", use_container_width=True):
            st.info("🔧 This feature will deploy the category-based question configuration to production.")

# Run the application
if __name__ == "__main__":
    main()





