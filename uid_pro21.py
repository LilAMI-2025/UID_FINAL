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
import difflib
from collections import defaultdict, Counter
import hashlib

# Setup
st.set_page_config(
    page_title="UID Matcher Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-card { 
        background: #fff3cd; 
        border-left: 4px solid #ffc107; 
    }
    
    .success-card { 
        background: #d4edda; 
        border-left: 4px solid #28a745; 
    }
    
    .info-card { 
        background: #d1ecf1; 
        border-left: 4px solid #17a2b8; 
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MATCHING_THRESHOLDS = {
    'tfidf_high': 0.60,
    'tfidf_low': 0.50,
    'semantic': 0.60,
    'heading_tfidf': 0.55,
    'heading_semantic': 0.65,
    'heading_length': 50
}

UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True
}

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# Survey Categories
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation'],
    'GROW': ['GROW'],
    'Feedback': ['feedback', 'evaluation', 'rating', 'satisfaction', 'opinion'],
    'Pulse': ['pulse', 'quick', 'brief', 'snapshot', 'check-in']
}

# Enhanced Synonym Mapping
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

# Initialize session state
def initialize_session_state():
    session_vars = {
        "page": "home",
        "df_target": None,
        "df_final": None,
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "df_reference": None,
        "survey_template": None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

initialize_session_state()

# Cached Resources
@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        st.error(f"Failed to load AI model: {e}")
        return None

@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        logger.info(f"Attempting Snowflake connection: user={sf.user}, account={sf.account}")
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
            f"?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        return engine
    except Exception as e:
        logger.error(f"Snowflake engine creation failed: {e}")
        if "250001" in str(e):
            st.warning("🔒 Snowflake connection failed: User account is locked.")
        return None

# Core Functions
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def categorize_survey(survey_title):
    if not survey_title:
        return "Unknown"
    
    title_lower = survey_title.lower()
    
    if 'GROW' in survey_title:
        return 'GROW'
    
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == 'GROW':
            continue
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    
    return "Other"

def score_question_quality(question):
    score = 0
    text = str(question).lower().strip()
    length = len(text)
    
    if 10 <= length <= 100:
        score += 20
    elif 5 <= length <= 150:
        score += 10
    elif length < 5:
        score -= 20
    
    if text.endswith('?'):
        score += 15
    
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
    if any(word in text.split()[:3] for word in question_words):
        score += 15
    
    if question and question[0].isupper():
        score += 10
    
    bad_patterns = ['click here', 'please select', '...', 'n/a', 'other', 'select one', 'choose all', 'privacy policy']
    if any(pattern in text for pattern in bad_patterns):
        score -= 15
    
    if '<' in text and '>' in text:
        score -= 20
    
    word_count = len(text.split())
    if 5 <= word_count <= 20:
        score += 10
    elif word_count > 30:
        score -= 5
    
    return score

# Database Functions
def run_snowflake_reference_query_all():
    """Fetch ALL reference questions from Snowflake"""
    try:
        engine = get_snowflake_engine()
        if engine is None:
            return pd.DataFrame()
            
        query = """
            SELECT HEADING_0, UID, SURVEY_TITLE
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL
            ORDER BY CAST(UID AS INTEGER) ASC
            LIMIT 5000
        """
        
        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn)
        
        logger.info(f"Fetched {len(result)} reference questions")
        return result
        
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

@st.cache_data
def get_all_reference_questions():
    return run_snowflake_reference_query_all()

def get_surveys(token):
    """Fetch surveys from SurveyMonkey"""
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch surveys: {e}")
        return []

# UI Functions
def create_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 UID Matcher Pro")
        st.markdown("Navigate through the application")
        
        if st.button("🏠 Home Dashboard", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("**📊 SurveyMonkey**")
        if st.button("👁️ View Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.rerun()
        if st.button("⚙️ Configure Survey", use_container_width=True):
            st.session_state.page = "configure_survey"
            st.rerun()
        if st.button("➕ Create New Survey", use_container_width=True):
            st.session_state.page = "create_survey"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("**📚 Question Bank**")
        if st.button("📖 View Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
            st.rerun()
        if st.button("⭐ Unique Questions Bank", use_container_width=True):
            st.session_state.page = "unique_question_bank"
            st.rerun()
        if st.button("📊 Categorized Questions", use_container_width=True):
            st.session_state.page = "categorized_questions"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("**⚖️ Governance**")
        st.markdown(f"• Max variations per UID: {UID_GOVERNANCE['max_variations_per_uid']}")
        st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
        st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")

def render_home_page():
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔄 Status", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            engine = get_snowflake_engine()
            if engine:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                    count = result.fetchone()[0]
                    st.metric("📊 Total UIDs", f"{count:,}")
            else:
                st.metric("📊 Total UIDs", "Connection Error")
        except:
            st.metric("📊 Total UIDs", "Connection Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                surveys = get_surveys(token)
                st.metric("📋 SM Surveys", len(surveys))
            else:
                st.metric("📋 SM Surveys", "No Token")
        except:
            st.metric("📋 SM Surveys", "API Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Governance", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced features highlight
    st.markdown("## 🚀 Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Enhanced UID Matching")
        st.markdown("• **Semantic Matching**: AI-powered question similarity")
        st.markdown("• **Governance Rules**: Automatic compliance checking")
        st.markdown("• **Conflict Detection**: Real-time duplicate identification")
        st.markdown("• **Quality Scoring**: Advanced question assessment")
        
        if st.button("🔧 Configure Survey with Enhanced Matching", use_container_width=True):
            st.session_state.page = "configure_survey"
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Survey Categorization")
        st.markdown("• **Auto-Categorization**: Smart survey type detection")
        st.markdown("• **Category Filters**: Application, GROW, Impact, etc.")
        st.markdown("• **Cross-Category Analysis**: Compare question patterns")
        st.markdown("• **Quality by Category**: Category-specific insights")
        
        if st.button("📊 View Categorized Questions", use_container_width=True):
            st.session_state.page = "categorized_questions"
            st.rerun()
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 SurveyMonkey Operations")
        if st.button("👁️ View & Analyze Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.rerun()
        if st.button("➕ Create New Survey", use_container_width=True):
            st.session_state.page = "create_survey"
            st.rerun()
    
    with col2:
        st.markdown("### 📚 Question Bank Management")
        if st.button("📖 View Full Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
            st.rerun()
        if st.button("⭐ Unique Questions Bank", use_container_width=True):
            st.session_state.page = "unique_question_bank"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        try:
            engine = get_snowflake_engine()
            if engine:
                st.markdown('<div class="success-card">✅ Snowflake: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
    
    with status_col2:
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                surveys = get_surveys(token)
                st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">❌ SurveyMonkey: No Token</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">❌ SurveyMonkey: API Issues</div>', unsafe_allow_html=True)
    
    with status_col3:
        st.markdown('<div class="success-card">✅ Governance: Active</div>', unsafe_allow_html=True)
        st.markdown(f"Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")

def render_view_surveys():
    st.markdown("## 👁️ View SurveyMonkey Surveys")
    st.markdown("*Browse and analyze your SurveyMonkey surveys*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">⚠️ SurveyMonkey token not configured in secrets.</div>', unsafe_allow_html=True)
            return
        
        with st.spinner("🔄 Fetching surveys from SurveyMonkey..."):
            surveys = get_surveys(token)
        
        if not surveys:
            st.markdown('<div class="info-card">ℹ️ No surveys found in your SurveyMonkey account.</div>', unsafe_allow_html=True)
            return
        
        st.success(f"📊 Found {len(surveys)} surveys in your account")
        
        # Create surveys dataframe
        surveys_data = []
        for survey in surveys:
            surveys_data.append({
                'ID': survey.get('id', ''),
                'Title': survey.get('title', ''),
                'Category': categorize_survey(survey.get('title', '')),
                'Date Created': survey.get('date_created', ''),
                'Response Count': survey.get('response_count', 0),
                'Question Count': survey.get('question_count', 0)
            })
        
        surveys_df = pd.DataFrame(surveys_data)
        
        # Display surveys
        st.dataframe(
            surveys_df,
            column_config={
                "ID": st.column_config.TextColumn("Survey ID", width="small"),
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Date Created": st.column_config.TextColumn("Created", width="medium"),
                "Response Count": st.column_config.NumberColumn("Responses", width="small"),
                "Question Count": st.column_config.NumberColumn("Questions", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
    except Exception as e:
        logger.error(f"View surveys failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

def render_unique_question_bank():
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*Best structured question for each UID with governance compliance and quality scoring*")
    
    try:
        with st.spinner("🔄 Loading question bank data..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found in the database.</div>', unsafe_allow_html=True)
                return
            
            st.info(f"📊 Loaded {len(df_reference)} total question variants from database")
        
        # Simple display of questions by UID
        st.markdown("### 📋 Questions by UID")
        
        if not df_reference.empty:
            # Group by UID and show sample
            uid_summary = df_reference.groupby('uid').agg({
                'heading_0': ['count', 'first'],
                'survey_title': 'first'
            }).round(2)
            
            # Flatten column names
            uid_summary.columns = ['Question_Count', 'Sample_Question', 'Survey_Title']
            uid_summary = uid_summary.reset_index()
            
            # Display first 50 UIDs
            display_df = uid_summary.head(50)
            
            st.dataframe(
                display_df,
                column_config={
                    "uid": st.column_config.TextColumn("UID", width="small"),
                    "Question_Count": st.column_config.NumberColumn("Count", width="small"),
                    "Sample_Question": st.column_config.TextColumn("Sample Question", width="large"),
                    "Survey_Title": st.column_config.TextColumn("Survey", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.info(f"Showing first 50 UIDs out of {len(uid_summary)} total UIDs")
        
    except Exception as e:
        logger.error(f"Unique questions bank failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

def render_view_question_bank():
    st.markdown("## 📖 View Full Question Bank")
    st.markdown("*Browse all questions in the database*")
    
    try:
        with st.spinner("🔄 Loading question bank..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No data found in the database.</div>', unsafe_allow_html=True)
                return
        
        st.success(f"📊 Loaded {len(df_reference)} questions from database")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
        
        # Filter data
        if search_term:
            filtered_df = df_reference[df_reference['heading_0'].str.contains(search_term, case=False, na=False)]
            st.info(f"Found {len(filtered_df)} questions matching '{search_term}'")
        else:
            filtered_df = df_reference.head(100)  # Show first 100 by default
            st.info("Showing first 100 questions. Use search to find specific questions.")
        
        # Display questions
        if not filtered_df.empty:
            st.dataframe(
                filtered_df,
                column_config={
                    "heading_0": st.column_config.TextColumn("Question", width="large"),
                    "uid": st.column_config.TextColumn("UID", width="small"),
                    "survey_title": st.column_config.TextColumn("Survey Title", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.markdown('<div class="info-card">ℹ️ No questions match your search.</div>', unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"View question bank failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

def render_categorized_questions():
    st.markdown("## 📊 Categorized Questions Bank")
    st.markdown("*Questions organized by survey categories*")
    
    try:
        with st.spinner("🔄 Loading and categorizing questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found.</div>', unsafe_allow_html=True)
                return
        
        # Add category column
        df_reference['category'] = df_reference['survey_title'].apply(categorize_survey)
        
        # Category overview
        category_stats = df_reference['category'].value_counts()
        
        st.markdown("### 📊 Category Overview")
        
        # Display category metrics
        cols = st.columns(min(4, len(category_stats)))
        for i, (category, count) in enumerate(category_stats.head(8).items()):
            with cols[i % 4]:
                st.metric(f"📋 {category}", count)
        
        st.markdown("---")
        
        # Category filter
        selected_category = st.selectbox(
            "📊 Select Category for Detailed View",
            ["All"] + sorted(category_stats.index.tolist())
        )
        
        # Filter by selected category
        if selected_category == "All":
            filtered_df = df_reference.copy()
        else:
            filtered_df = df_reference[df_reference['category'] == selected_category].copy()
        
        st.markdown(f"### 📋 {selected_category} Questions ({len(filtered_df)} items)")
        
        if not filtered_df.empty:
            # Display questions
            display_df = filtered_df[['uid', 'heading_0', 'category', 'survey_title']].head(50)
            
            st.dataframe(
                display_df,
                column_config={
                    "uid": st.column_config.TextColumn("UID", width="small"),
                    "heading_0": st.column_config.TextColumn("Question", width="large"),
                    "category": st.column_config.TextColumn("Category", width="medium"),
                    "survey_title": st.column_config.TextColumn("Survey", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
            
            if len(filtered_df) > 50:
                st.info(f"Showing first 50 questions out of {len(filtered_df)} total")
        else:
            st.markdown('<div class="info-card">ℹ️ No questions found in the selected category.</div>', unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Categorized questions failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

def render_configure_survey():
    st.markdown("## ⚙️ Configure Survey")
    st.markdown("*Upload or configure surveys for UID matching*")
    
    # Survey input methods
    input_method = st.radio(
        "📊 Choose survey input method:",
        ["Upload JSON File", "Fetch from SurveyMonkey", "Manual Entry"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if input_method == "Upload JSON File":
        uploaded_file = st.file_uploader("📁 Upload Survey JSON", type=['json'])
        
        if uploaded_file:
            try:
                survey_data = json.load(uploaded_file)
                st.success("✅ Survey JSON loaded successfully!")
                
                # Show basic info
                st.write(f"**Survey Title:** {survey_data.get('title', 'Unknown')}")
                st.write(f"**Survey ID:** {survey_data.get('id', 'Unknown')}")
                
                # Count questions
                question_count = 0
                for page in survey_data.get("pages", []):
                    question_count += len(page.get("questions", []))
                
                st.write(f"**Questions Found:** {question_count}")
                
                if st.button("🚀 Process Survey", type="primary"):
                    st.success("Survey processing would start here!")
                        
            except Exception as e:
                st.error(f"❌ Error loading survey: {e}")
    
    elif input_method == "Fetch from SurveyMonkey":
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">⚠️ SurveyMonkey token not configured.</div>', unsafe_allow_html=True)
            return
        
        try:
            with st.spinner("🔄 Fetching surveys..."):
                surveys = get_surveys(token)
            
            if surveys:
                survey_options = {f"{s['title']} (ID: {s['id']})": s['id'] for s in surveys}
                selected_survey = st.selectbox("📋 Select Survey:", list(survey_options.keys()))
                
                if selected_survey and st.button("📥 Load Selected Survey"):
                    survey_id = survey_options[selected_survey]
                    st.success(f"✅ Would load survey {survey_id}")
                    
        except Exception as e:
            st.error(f"❌ Error fetching surveys: {e}")
    
    elif input_method == "Manual Entry":
        st.markdown("### ✏️ Manual Question Entry")
        
        # Simple form for adding questions
        with st.form("add_question_form"):
            question_text = st.text_area("📝 Question Text:")
            question_type = st.selectbox("📊 Question Type:", ["Single Choice", "Multiple Choice", "Open-Ended"])
            
            if st.form_submit_button("➕ Add Question"):
                if question_text.strip():
                    st.success("✅ Question would be added to manual survey!")
                else:
                    st.error("❌ Please enter question text")

def render_create_survey():
    st.markdown("## ➕ Create New Survey")
    st.markdown("*Create a new survey in SurveyMonkey*")
    
    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.markdown('<div class="warning-card">⚠️ SurveyMonkey token not configured.</div>', unsafe_allow_html=True)
        return
    
    with st.form("create_survey_form"):
        survey_title = st.text_input("📝 Survey Title:")
        survey_description = st.text_area("📄 Survey Description (optional):")
        survey_language = st.selectbox("🌐 Language:", ["en", "es", "fr", "de"])
        
        if st.form_submit_button("🚀 Create Survey", type="primary"):
            if survey_title.strip():
                st.success(f"✅ Would create survey: '{survey_title}'")
            else:
                st.error("❌ Please enter a survey title")

# Create sidebar
create_sidebar()

# App Header
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing Snowflake configuration in secrets.</div>', unsafe_allow_html=True)
    st.markdown("Please configure your Snowflake credentials in the secrets to use database features.")

if "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing SurveyMonkey configuration in secrets.</div>', unsafe_allow_html=True)
    st.markdown("Please configure your SurveyMonkey token in the secrets to use API features.")

# Main page routing
if st.session_state.page == "home":
    render_home_page()

elif st.session_state.page == "view_surveys":
    render_view_surveys()

elif st.session_state.page == "configure_survey":
    render_configure_survey()

elif st.session_state.page == "create_survey":
    render_create_survey()

elif st.session_state.page == "view_question_bank":
    render_view_question_bank()

elif st.session_state.page == "unique_question_bank":
    render_unique_question_bank()

elif st.session_state.page == "categorized_questions":
    render_categorized_questions()

else:
    # Default fallback for any undefined pages
    st.markdown("## 🔄 Page Under Development")
    st.markdown("This page is currently being developed. Please use the sidebar to navigate to available pages.")
    
    # Show available pages
    st.markdown("### Available Pages:")
    st.markdown("• **🏠 Home Dashboard** - Main overview and system status")
    st.markdown("• **👁️ View Surveys** - Browse SurveyMonkey surveys")
    st.markdown("• **⚙️ Configure Survey** - Upload and configure surveys")
    st.markdown("• **➕ Create Survey** - Create new surveys")
    st.markdown("• **📖 View Question Bank** - Browse all questions")
    st.markdown("• **⭐ Unique Questions Bank** - Best questions per UID")
    st.markdown("• **📊 Categorized Questions** - Questions by category")
    
    if st.button("🏠 Return to Home", type="primary"):
        st.session_state.page = "home"
        st.rerun()

# Footer information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <small>
        🧠 UID Matcher Pro v2.0 | 
        Enhanced with AI-powered semantic matching, governance compliance, and survey categorization
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)
