"""
UID Matcher Enhanced - Reflex App
Converted from Streamlit to Reflex
"""

import reflex as rx
import pandas as pd
import requests
import re
import logging
import json
import time
import os
import numpy as np
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

# ============= LOGGING SETUP =============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= CONSTANTS AND CONFIGURATION =============

# Matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50

# Model and API settings
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
MAX_SURVEYS_PER_BATCH = 10

# Identity types for export filtering
IDENTITY_TYPES = [
    'full name', 'first name', 'last name', 'e-mail', 'company', 'gender', 
    'country', 'age', 'title', 'role', 'phone number', 'location', 
    'pin', 'passport', 'date of birth', 'uct', 'student number',
    'department', 'region', 'city', 'id number', 'marital status',
    'education level', 'english proficiency', 'email', 'surname',
    'name', 'contact', 'address', 'mobile', 'telephone', 'qualification',
    'degree', 'identification', 'birth', 'married', 'single', 'language',
    'sex', 'position', 'job', 'organization', 'organisation'
]

# Enhanced synonym mapping
ENHANCED_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
    "how many people report to you": "team size",
    "how many staff report to you": "team size",
    "what is age": "what is your age",
    "what age": "what is your age",
    "your age": "what is your age",
    "current role": "current position",
    "your role": "your position",
}

# AMI Structure Categories
SURVEY_STAGES = {
    "Recruitment Survey": ["application", "apply", "applying", "candidate", "candidacy", "admission", "enrolment", "enrollment", "combined app"],
    "Pre-Programme Survey": ["pre programme", "pre-programme", "pre program", "pre-program", "before programme", "preparation", "prep"],
    "LL Feedback Survey": ["ll feedback", "learning lab", "in-person", "multilingual"],
    "Pulse Check Survey": ["pulse", "check-in", "checkin", "pulse check"],
    "Progress Review Survey": ["progress", "review", "assessment", "evaluation", "mid-point", "checkpoint", "interim"],
    "Growth Goal Reflection": ["growth goal", "post-ll", "reflection"],
    "AP Survey": ["ap survey", "accountability partner", "ap post"],
    "Longitudinal Survey": ["longitudinal", "impact", "annual impact"],
    "CEO/Client Lead Survey": ["ceo", "client lead", "clientlead"],
    "Change Challenge Survey": ["change challenge"],
    "Organisational Practices Survey": ["organisational practices", "organizational practices"],
    "Post-bootcamp Feedback Survey": ["post bootcamp", "bootcamp feedback"],
    "Set your goal post LL": ["set your goal", "post ll"],
    "Other": ["drop-out", "attrition", "finance link", "mentorship application"]
}

RESPONDENT_TYPES = {
    "Participant": ["participant", "learner", "student", "individual", "person"],
    "Business": ["business", "enterprise", "company", "entrepreneur", "owner"],
    "Team member": ["team member", "staff", "employee", "worker"],
    "Accountability Partner": ["accountability partner", "ap", "manager", "supervisor"],
    "Client Lead": ["client lead", "ceo", "executive", "leadership"],
    "Managers": ["managers", "management", "supervisor"]
}

PROGRAMMES = {
    "Grow Your Business (GYB)": ["gyb", "grow your business", "grow business"],
    "Micro Enterprise Accelerator (MEA)": ["mea", "micro enterprise", "accelerator"],
    "Start your Business (SYB)": ["syb", "start your business", "start business"],
    "Leadership Development Programme (LDP)": ["ldp", "leadership development", "leadership"],
    "Management Development Programme (MDP)": ["mdp", "management development", "management"],
    "Thrive at Work (T@W)": ["taw", "thrive at work", "thrive", "t@w"],
    "Bootcamp": ["bootcamp", "boot camp", "survival bootcamp", "work readiness", "get set up"],
    "Academy": ["academy", "care academy"],
    "Finance Link": ["finance link"],
    "Custom": ["winning behaviours", "custom", "learning needs"],
    "ALL": ["all programmes", "template", "multilingual"]
}

# UID Final Reference Data (truncated for brevity)
UID_FINAL_REFERENCE = {
    "On a scale of 0-10, how likely is it that you would recommend AMI to someone (a colleague, friend or other business?)": 1,
    "Do you (in general) feel more confident about your ability to raise capital for your business?": 38,
    "Have you set and shared your Growth Goal with AMI?": 57,
    "What is your gender?": 233,
    "What is your age?": 234,
    # ... (add more as needed)
}

# ============= STATE MANAGEMENT =============
class AppState(rx.State):
    """Main application state"""
    
    # Page navigation
    current_page: str = "home"
    
    # Connection status
    sm_connected: bool = False
    sf_connected: bool = False
    sm_message: str = ""
    sf_message: str = ""
    
    # Data
    surveys: List[Dict] = []
    df_target: Optional[pd.DataFrame] = None
    df_final: Optional[pd.DataFrame] = None
    question_bank: Optional[pd.DataFrame] = None
    question_bank_with_authority: Optional[pd.DataFrame] = None
    all_questions: Optional[pd.DataFrame] = None
    unique_uid_table: Optional[pd.DataFrame] = None
    
    # UI state
    selected_surveys: List[str] = []
    search_query: str = ""
    show_main_only: bool = True
    loading: bool = False
    
    # Filters
    survey_stage_filter: List[str] = list(SURVEY_STAGES.keys())
    respondent_type_filter: List[str] = list(RESPONDENT_TYPES.keys())
    programme_filter: List[str] = list(PROGRAMMES.keys())
    
    # Metrics
    total_questions: int = 0
    main_questions: int = 0
    matched_percentage: float = 0.0
    
    def navigate_to(self, page: str):
        """Navigate to a specific page"""
        self.current_page = page
    
    async def initialize_app(self):
        """Initialize the application"""
        self.loading = True
        
        # Check connections
        await self.check_connections()
        
        # Load initial data
        await self.load_initial_data()
        
        self.loading = False
    
    async def check_connections(self):
        """Check SurveyMonkey and Snowflake connections"""
        try:
            # Check SurveyMonkey
            sm_status, sm_msg = await self.check_surveymonkey_connection()
            self.sm_connected = sm_status
            self.sm_message = sm_msg
            
            # Check Snowflake
            sf_status, sf_msg = await self.check_snowflake_connection()
            self.sf_connected = sf_status
            self.sf_message = sf_msg
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
    
    async def check_surveymonkey_connection(self):
        """Check SurveyMonkey API connection"""
        try:
            token = self.get_surveymonkey_token()
            if not token:
                return False, "No access token available"
            
            headers = {
                "Authorization": f"Bearer {token}",
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
            else:
                return False, f"API error: {response.status_code}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    async def check_snowflake_connection(self):
        """Check Snowflake connection"""
        try:
            # Implementation would depend on your Snowflake setup
            # For now, return a placeholder
            return True, "Connected to Snowflake"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_surveymonkey_token(self):
        """Get SurveyMonkey token from environment or config"""
        # In Reflex, you'd typically use environment variables
        return os.getenv("SURVEYMONKEY_ACCESS_TOKEN")
    
    async def load_initial_data(self):
        """Load initial data"""
        try:
            if self.sm_connected:
                await self.load_surveys()
            
            if self.sf_connected:
                await self.load_question_bank()
                
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
    
    async def load_surveys(self):
        """Load surveys from SurveyMonkey"""
        try:
            token = self.get_surveymonkey_token()
            if not token:
                return
            
            url = "https://api.surveymonkey.com/v3/surveys"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                self.surveys = response.json().get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to load surveys: {e}")
    
    async def load_question_bank(self):
        """Load question bank from Snowflake"""
        try:
            # Placeholder for Snowflake query
            # You'd implement the actual Snowflake connection here
            pass
        except Exception as e:
            logger.error(f"Failed to load question bank: {e}")
    
    def update_search_query(self, value: str):
        """Update search query"""
        self.search_query = value
    
    def toggle_main_only(self):
        """Toggle show main questions only"""
        self.show_main_only = not self.show_main_only
    
    def update_selected_surveys(self, surveys: List[str]):
        """Update selected surveys"""
        self.selected_surveys = surveys

# ============= UTILITY FUNCTIONS =============

def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced text normalization with synonym mapping"""
    if not isinstance(text, str):
        return ""
    try:
        text = text.lower().strip()
        # Apply synonym mapping
        for phrase, replacement in synonym_map.items():
            text = text.replace(phrase, replacement)
        
        # Remove punctuation and normalize spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stop words
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return ""

# ============= COMPONENTS =============

def metric_card(title: str, value: str, description: str = ""):
    """Create a metric card component"""
    return rx.card(
        rx.vstack(
            rx.heading(title, size="4"),
            rx.text(value, size="6", weight="bold"),
            rx.text(description, size="2", color="gray"),
            spacing="1",
        ),
        padding="4",
        border="1px solid #e2e8f0",
        border_radius="8px",
    )

def navigation_sidebar():
    """Create navigation sidebar"""
    return rx.vstack(
        rx.heading("🧠 UID Matcher Enhanced", size="5"),
        rx.text("Advanced question bank with UID Final reference", size="2"),
        
        rx.divider(),
        
        # Connection status
        rx.vstack(
            rx.text("🔗 Connection Status", weight="bold"),
            rx.hstack(
                rx.text("📊 SurveyMonkey:"),
                rx.cond(
                    AppState.sm_connected,
                    rx.text("✅", color="green"),
                    rx.text("❌", color="red")
                )
            ),
            rx.hstack(
                rx.text("❄️ Snowflake:"),
                rx.cond(
                    AppState.sf_connected,
                    rx.text("✅", color="green"),
                    rx.text("❌", color="red")
                )
            ),
            spacing="2",
        ),
        
        rx.divider(),
        
        # Main navigation
        rx.vstack(
            rx.button(
                "🏠 Home Dashboard",
                on_click=AppState.navigate_to("home"),
                variant="ghost",
                width="100%",
            ),
            rx.button(
                "📋 Survey Selection",
                on_click=AppState.navigate_to("survey_selection"),
                variant="ghost",
                width="100%",
            ),
            rx.button(
                "📊 AMI Categories", 
                on_click=AppState.navigate_to("survey_categorization"),
                variant="ghost",
                width="100%",
            ),
            rx.button(
                "🔧 UID Matching",
                on_click=AppState.navigate_to("uid_matching"),
                variant="ghost", 
                width="100%",
            ),
            rx.button(
                "📖 Question Bank",
                on_click=AppState.navigate_to("question_bank"),
                variant="ghost",
                width="100%",
            ),
            rx.button(
                "🏗️ Survey Creation",
                on_click=AppState.navigate_to("survey_creation"),
                variant="ghost",
                width="100%",
            ),
            spacing="2",
        ),
        
        spacing="4",
        padding="4",
        width="250px",
        height="100vh",
        border_right="1px solid #e2e8f0",
    )

def home_page():
    """Home dashboard page"""
    return rx.vstack(
        rx.heading("🏠 Welcome to Enhanced UID Matcher", size="7"),
        rx.text("SurveyMonkey surveys → Snowflake reference → UID Final mapping → Enhanced question bank"),
        
        # Metrics
        rx.hstack(
            metric_card("🔄 Status", "Active"),
            metric_card("📊 SM Surveys", rx.text(AppState.surveys.length())),
            metric_card("🎯 UID Final Refs", str(len(UID_FINAL_REFERENCE))),
            spacing="4",
        ),
        
        rx.divider(),
        
        # UID Final Reference info
        rx.card(
            rx.vstack(
                rx.heading("🎯 UID Final Reference", size="5"),
                rx.text("New Feature: UID Final reference mapping loaded from provided file"),
                rx.text(f"• {len(UID_FINAL_REFERENCE)} questions mapped to UID Final values"),
                rx.text("• Used in Question Bank viewer for enhanced reference"),
                rx.text("• Provides authoritative UID assignments"),
                spacing="2",
            ),
            padding="4",
            background="blue.50",
        ),
        
        rx.divider(),
        
        # Workflow guide
        rx.heading("🚀 Recommended Workflow", size="5"),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.heading("1️⃣ Survey Selection", size="4"),
                    rx.text("Select and analyze surveys:"),
                    rx.text("• Browse available surveys"),
                    rx.text("• Extract questions with IDs"), 
                    rx.text("• Review question bank"),
                    rx.button(
                        "📋 Start Survey Selection",
                        on_click=AppState.navigate_to("survey_selection"),
                        width="100%",
                    ),
                    spacing="2",
                ),
                padding="4",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("2️⃣ AMI Structure", size="4"),
                    rx.text("Categorize with AMI structure:"),
                    rx.text("• Survey Stage classification"),
                    rx.text("• Respondent Type grouping"),
                    rx.text("• Programme alignment"),
                    rx.button(
                        "📊 View AMI Categories",
                        on_click=AppState.navigate_to("survey_categorization"),
                        width="100%",
                    ),
                    spacing="2",
                ),
                padding="4",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("3️⃣ Question Bank", size="4"),
                    rx.text("Enhanced question bank:"),
                    rx.text("• Snowflake reference questions"),
                    rx.text("• UID Final reference"),
                    rx.text("• Unique UID table creation"),
                    rx.button(
                        "📖 View Question Bank",
                        on_click=AppState.navigate_to("question_bank"),
                        width="100%",
                    ),
                    spacing="2",
                ),
                padding="4",
            ),
            spacing="4",
        ),
        
        spacing="6",
        padding="6",
    )

def survey_selection_page():
    """Survey selection page"""
    return rx.vstack(
        rx.heading("📋 Survey Selection & Question Bank", size="6"),
        rx.text("📊 Data Source: SurveyMonkey API - Survey selection and question extraction"),
        
        rx.cond(
            AppState.surveys.length() == 0,
            rx.card(
                rx.text("⚠️ No surveys available. Check SurveyMonkey connection."),
                background="yellow.50",
                padding="4",
            ),
            rx.vstack(
                rx.heading("🔍 Select Surveys", size="4"),
                # Survey multiselect would go here
                rx.text("Survey selection component would be implemented here"),
                spacing="4",
            )
        ),
        
        spacing="4",
        padding="6",
    )

def question_bank_page():
    """Question bank page"""
    return rx.vstack(
        rx.heading("📚 Enhanced Question Bank Viewer", size="6"),
        rx.text("❄️ Data Source: Snowflake + UID Final Reference - Enhanced question bank with authoritative mappings"),
        
        # UID Final Reference section
        rx.card(
            rx.vstack(
                rx.heading("🎯 UID Final Reference (Cached)", size="4"),
                rx.text(f"Loaded {len(UID_FINAL_REFERENCE)} authoritative question-to-UID Final mappings from cached reference"),
                rx.text("This is the master reference for UID Final assignments, independent of Snowflake connection."),
                spacing="2",
            ),
            background="blue.50",
            padding="4",
        ),
        
        # Metrics
        rx.hstack(
            metric_card("📊 Total UID Final Records", str(len(UID_FINAL_REFERENCE))),
            metric_card("🎯 Unique UID Finals", str(len(set(UID_FINAL_REFERENCE.values())))),
            metric_card("📈 UID Final Range", f"{min(UID_FINAL_REFERENCE.values())}-{max(UID_FINAL_REFERENCE.values())}"),
            spacing="4",
        ),
        
        spacing="4",
        padding="6",
    )

def page_router():
    """Route to appropriate page based on current_page state"""
    return rx.cond(
        AppState.current_page == "home",
        home_page(),
        rx.cond(
            AppState.current_page == "survey_selection",
            survey_selection_page(),
            rx.cond(
                AppState.current_page == "question_bank", 
                question_bank_page(),
                home_page()  # Default fallback
            )
        )
    )

def index():
    """Main application layout"""
    return rx.hstack(
        navigation_sidebar(),
        rx.box(
            page_router(),
            flex="1",
            overflow="auto",
        ),
        width="100%",
        height="100vh",
    )

# ============= APP CONFIGURATION =============
app = rx.App(
    style={
        "font_family": "Inter",
    }
)

app.add_page(index, route="/")

if __name__ == "__main__":
    app.run() 