"""
Configuration settings for UID Matcher Reflex App
"""

import os
from typing import Dict, Any

# ============= ENVIRONMENT VARIABLES =============

def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    return {
        # SurveyMonkey Configuration
        "surveymonkey": {
            "access_token": os.getenv("SURVEYMONKEY_ACCESS_TOKEN"),
        },
        
        # Snowflake Configuration
        "snowflake": {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        },
        
        # App Configuration
        "app": {
            "environment": os.getenv("APP_ENV", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        }
    }

# ============= EXAMPLE ENVIRONMENT VARIABLES =============
"""
Copy these to your .env file or set as environment variables:

# SurveyMonkey API Configuration
SURVEYMONKEY_ACCESS_TOKEN=your_surveymonkey_token_here

# Snowflake Configuration
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role

# Optional: App Configuration
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false
""" 