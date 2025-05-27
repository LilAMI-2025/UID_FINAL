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
    
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True
}

# Survey Categories based on titles
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation'],
    'GROW': ['GROW'],  # Exact match for CAPS
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

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "df_target": None,
        "df_final": None,
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "df_reference": None,
        "survey_template": None,
        "snowflake_initialized": False,
        "surveymonkey_initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# ============= UTILITY FUNCTIONS =============

@st.cache_resource
def load_sentence_transformer():
    """Load SentenceTransformer model (cached)"""
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced text normalization"""
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    # Apply synonym mapping
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def categorize_survey(survey_title):
    """Categorize survey based on title keywords"""
    if not survey_title:
        return "Unknown"
    
    title_lower = survey_title.lower()
    
    # Check GROW first (exact match for CAPS)
    if 'GROW' in survey_title:
        return 'GROW'
    
    # Check other categories
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == 'GROW':
            continue
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    
    return "Other"

def classify_question(text, heading_references=HEADING_REFERENCES):
    """Classify question as Heading or Main Question"""
    if len(text.split()) > HEADING_LENGTH_THRESHOLD:
        return "Heading"
    
    # TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        all_texts = heading_references + [text]
        tfidf_vectors = vectorizer.fit_transform([enhanced_normalize(t) for t in all_texts])
        similarity_scores = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
        max_tfidf_score = np.max(similarity_scores)
        
        # Semantic similarity
        model = load_sentence_transformer()
        emb_text = model.encode([text], convert_to_tensor=True)
        emb_refs = model.encode(heading_references, convert_to_tensor=True)
        semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
        max_semantic_score = np.max(semantic_scores.cpu().numpy())
        
        if max_tfidf_score >= HEADING_TFIDF_THRESHOLD or max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
            return "Heading"
    except Exception as e:
        logger.error(f"Question classification failed: {e}")
    
    return "Main Question/Multiple Choice"

def score_question_quality(question):
    """Score question quality"""
    score = 0
    text = str(question).lower().strip()
    
    # Length scoring
    length = len(text)
    if 10 <= length <= 100:
        score += 20
    elif 5 <= length <= 150:
        score += 10
    elif length < 5:
        score -= 20
    
    # Question format scoring
    if text.endswith('?'):
        score += 15
    
    # English question words
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
    if any(word in text.split()[:3] for word in question_words):
        score += 15
    
    # Proper capitalization
    if question and question[0].isupper():
        score += 10
    
    # Avoid artifacts
    bad_patterns = ['click here', 'please select', '...', 'n/a', 'other', 'select one', 'choose all', 'privacy policy']
    if any(pattern in text for pattern in bad_patterns):
        score -= 15
    
    # Avoid HTML
    if '<' in text and '>' in text:
        score -= 20
    
    return score

def get_best_question_for_uid(questions_list):
    """Get the best quality question from a list"""
    if not questions_list:
        return None
    
    scored_questions = [(q, score_question_quality(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

def create_unique_questions_bank(df_reference):
    """Create unique questions bank with best question per UID"""
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_reference)} reference questions for unique bank")
    
    unique_questions = []
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        uid_questions = group['heading_0'].tolist()
        best_question = get_best_question_for_uid(uid_questions)
        
        # Get survey titles for categorization
        survey_titles = group.get('survey_title', pd.Series()).dropna().unique()
        categories = [categorize_survey(title) for title in survey_titles]
        primary_category = categories[0] if len(set(categories)) == 1 else "Mixed"
        
        if best_question:
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split()),
                'survey_category': primary_category,
                'survey_titles': ', '.join(survey_titles) if len(survey_titles) > 0 else 'Unknown',
                'quality_score': score_question_quality(best_question),
                'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid']
            })
    
    unique_df = pd.DataFrame(unique_questions)
    
    # Sort by UID
    if not unique_df.empty:
        try:
            unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
            unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
            unique_df = unique_df.drop('uid_numeric', axis=1)
        except:
            unique_df = unique_df.sort_values('uid')
    
    return unique_df

# ============= SURVEYMONKEY FUNCTIONS =============

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_surveys(token):
    """Get surveys from SurveyMonkey API"""
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch surveys: {e}")
        raise

def get_survey_details(survey_id, token):
    """Get detailed survey information"""
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
        raise

def create_survey(token, survey_template):
    """Create new survey via SurveyMonkey API"""
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": survey_template["title"],
            "nickname": survey_template.get("nickname", survey_template["title"]),
            "language": survey_template.get("language", "en")
        })
        response.raise_for_status()
        return response.json().get("id")
    except requests.RequestException as e:
        logger.error(f"Failed to create survey: {e}")
        raise

def extract_questions(survey_json):
    """Extract questions from survey JSON"""
    questions = []
    global_position = 0
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            
            # Determine schema type
            if family == "single_choice":
                schema_type = "Single Choice"
            elif family == "multiple_choice":
                schema_type = "Multiple Choice"
            elif family == "open_ended":
                schema_type = "Open-Ended"
            elif family == "matrix":
                schema_type = "Matrix"
            else:
                choices = question.get("answers", {}).get("choices", [])
                schema_type = "Multiple Choice" if choices else "Open-Ended"
                if choices and ("select one" in q_text.lower() or len(choices) <= 2):
                    schema_type = "Single Choice"
            
            question_category = classify_question(q_text)
            
            if q_text:
                global_position += 1
                questions.append({
                    "heading_0": q_text,
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_uid": q_id,
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", ""),
                    "question_category": question_category
                })
                
                # Add choices
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "heading_0": f"{q_text} - {choice_text}",
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_uid": q_id,
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False,
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", ""),
                            "question_category": "Main Question/Multiple Choice"
                        })
    return questions

# ============= SNOWFLAKE FUNCTIONS =============

@st.cache_resource
def get_snowflake_engine():
    """Initialize Snowflake connection (cached)"""
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
            st.warning(
                "🔒 Snowflake connection failed: User account is locked. "
                "UID matching is disabled, but you can use SurveyMonkey features. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
            )
        raise

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_all_reference_questions():
    """Fetch ALL reference questions from Snowflake with pagination"""
    all_data = []
    limit = 10000
    offset = 0
    
    while True:
        query = """
            SELECT HEADING_0, UID, SURVEY_TITLE
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL
            ORDER BY CAST(UID AS INTEGER) ASC
            LIMIT :limit OFFSET :offset
        """
        try:
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
            
            if result.empty:
                break
                
            all_data.append(result)
            offset += limit
            
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            if "250001" in str(e):
                st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

def run_snowflake_target_query():
    """Get target questions without UIDs"""
    query = """
        SELECT DISTINCT HEADING_0, SURVEY_TITLE
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NULL AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
        ORDER BY HEADING_0
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        return result
    except Exception as e:
        logger.error(f"Snowflake target query failed: {e}")
        raise

# ============= CONNECTION CHECKS =============

def check_surveymonkey_connection():
    """Check SurveyMonkey API connection"""
    try:
        token = st.secrets.get("surveymonkey", {}).get("token") or st.secrets.get("surveymonkey", {}).get("access_token")
        if not token:
            return False, "No SurveyMonkey token found"
        
        surveys = get_surveys(token)
        return True, f"Connected - {len(surveys)} surveys found"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def check_snowflake_connection():
    """Check Snowflake connection"""
    try:
        get_snowflake_engine()
        return True, "Connected successfully"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# ============= SIDEBAR NAVIGATION =============

with st.sidebar:
    st.markdown("### 🧠 UID Matcher Pro")
    st.markdown("Navigate through the application")
    
    # Connection status
    sm_status, sm_msg = check_surveymonkey_connection()
    sf_status, sf_msg = check_snowflake_connection()
    
    st.markdown("**🔗 Connection Status**")
    st.write(f"📊 SurveyMonkey: {'✅' if sm_status else '❌'}")
    st.write(f"❄️ Snowflake: {'✅' if sf_status else '❌'}")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # SurveyMonkey section (Priority 1)
    st.markdown("**📊 SurveyMonkey (Step 1)**")
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.session_state.surveymonkey_initialized = True
        st.rerun()
    if st.button("➕ Create New Survey", use_container_width=True):
        st.session_state.page = "create_survey"
        st.session_state.surveymonkey_initialized = True
        st.rerun()
    
    st.markdown("---")
    
    # Configuration section (Step 2)
    st.markdown("**⚙️ Configuration (Step 2)**")
    if st.button("⚙️ Configure Survey", use_container_width=True):
        if not st.session_state.surveymonkey_initialized:
            st.warning("⚠️ Please initialize SurveyMonkey first by viewing surveys")
        else:
            st.session_state.page = "configure_survey"
            st.rerun()
    
    st.markdown("---")
    
    # Question Bank section (Requires Snowflake)
    st.markdown("**📚 Question Bank (Snowflake)**")
    if st.button("📖 View Question Bank", use_container_width=True):
        st.session_state.page = "view_question_bank"
        st.session_state.snowflake_initialized = True
        st.rerun()
    if st.button("⭐ Unique Questions Bank", use_container_width=True):
        st.session_state.page = "unique_question_bank"
        st.session_state.snowflake_initialized = True
        st.rerun()
    if st.button("📊 Categorized Questions", use_container_width=True):
        st.session_state.page = "categorized_questions"
        st.session_state.snowflake_initialized = True
        st.rerun()
    if st.button("🧹 Data Quality Management", use_container_width=True):
        st.session_state.page = "data_quality"
        st.session_state.snowflake_initialized = True
        st.rerun()
    
    st.markdown("---")
    
    # Governance info
    st.markdown("**⚖️ Governance**")
    st.markdown(f"• Max variations per UID: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")

# ============= MAIN APP HEADER =============

st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets and "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake and SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# ============= PAGE ROUTING =============

if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔄 Status", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if sf_status:
            try:
                with get_snowflake_engine().connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                    count = result.fetchone()[0]
                    st.metric("📊 Total UIDs", f"{count:,}")
            except:
                st.metric("📊 Total UIDs", "Error")
        else:
            st.metric("📊 Total UIDs", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if sm_status:
            try:
                token = st.secrets.get("surveymonkey", {}).get("token") or st.secrets.get("surveymonkey", {}).get("access_token")
                surveys = get_surveys(token)
                st.metric("📋 SM Surveys", len(surveys))
            except:
                st.metric("📋 SM Surveys", "API Error")
        else:
            st.metric("📋 SM Surveys", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Governance", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow guide
    st.markdown("## 🚀 Recommended Workflow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Step 1: SurveyMonkey Operations")
        st.markdown("Start here to avoid function collisions:")
        st.markdown("• **View Surveys** - Initialize SurveyMonkey connection")
        st.markdown("• **Create Surveys** - Build new surveys")
        st.markdown("• **Extract Questions** - Get survey data")
        
        if st.button("🔧 Start with SurveyMonkey", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.session_state.surveymonkey_initialized = True
            st.rerun()
    
    with col2:
        st.markdown("### ❄️ Step 2: Snowflake Operations")
        st.markdown("Use after SurveyMonkey initialization:")
        st.markdown("• **View Question Bank** - Browse all questions")
        st.markdown("• **Unique Questions** - Best question per UID")
        st.markdown("• **Data Quality** - Clean and optimize")
        
        if st.button("🎯 Proceed to Question Bank", use_container_width=True):
            if not sf_status:
                st.error("❌ Snowflake connection required")
            else:
                st.session_state.page = "view_question_bank"
                st.session_state.snowflake_initialized = True
                st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if sm_status:
            st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-card">❌ SurveyMonkey: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Details: {sm_msg}")
    
    with status_col2:
        if sf_status:
            st.markdown('<div class="success-card">✅ Snowflake: Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Details: {sf_msg}")

elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ SurveyMonkey Survey Viewer")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token") or st.secrets.get("surveymonkey", {}).get("access_token")
        if not token:
            st.error("❌ SurveyMonkey token not found in secrets")
            st.stop()
        
        with st.spinner("🔄 Loading surveys from SurveyMonkey..."):
            surveys = get_surveys(token)
        
        if not surveys:
            st.warning("⚠️ No surveys found in your SurveyMonkey account")
            st.stop()
        
        st.success(f"✅ Loaded {len(surveys)} surveys from SurveyMonkey")
        
        # Survey selection
        survey_options = {f"{s['title']} (ID: {s['id']})": s['id'] for s in surveys}
        selected_survey_display = st.selectbox("Select a survey to analyze:", list(survey_options.keys()))
        selected_survey_id = survey_options[selected_survey_display]
        
        if st.button("🔍 Analyze Selected Survey", type="primary"):
            with st.spinner("🔄 Fetching survey details and extracting questions..."):
                try:
                    # Get survey details
                    survey_details = get_survey_details(selected_survey_id, token)
                    
                    # Extract questions
                    questions = extract_questions(survey_details)
                    df_questions = pd.DataFrame(questions)
                    
                    if df_questions.empty:
                        st.warning("⚠️ No questions found in this survey")
                    else:
                        st.success(f"✅ Extracted {len(df_questions)} questions from survey")
                        
                        # Store in session state for configuration
                        st.session_state.df_target = df_questions
                        st.session_state.survey_template = survey_details
                        
                        # Display results
                        st.markdown("### 📊 Survey Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Questions", len(df_questions))
                        with col2:
                            main_questions = len(df_questions[df_questions['is_choice'] == False])
                            st.metric("Main Questions", main_questions)
                        with col3:
                            choices = len(df_questions[df_questions['is_choice'] == True])
                            st.metric("Choice Options", choices)
                        
                        # Question categorization
                        if 'question_category' in df_questions.columns:
                            st.markdown("### 📋 Question Categories")
                            category_counts = df_questions['question_category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        # Survey category
                        survey_title = survey_details.get('title', 'Unknown')
                        survey_category = categorize_survey(survey_title)
                        st.markdown(f"**Survey Category:** {survey_category}")
                        
                        # Sample questions
                        st.markdown("### 📝 Sample Questions")
                        sample_questions = df_questions[df_questions['is_choice'] == False].head(5)
                        for idx, row in sample_questions.iterrows():
                            st.write(f"**{row['position']}.** {row['heading_0']}")
                            st.write(f"   *Type: {row['schema_type']} | Category: {row['question_category']}*")
                        
                        # Next steps
                        st.markdown("### ➡️ Next Steps")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("⚙️ Configure UID Matching", use_container_width=True):
                                st.session_state.page = "configure_survey"
                                st.rerun()
                        
                        with col2:
                            # Export functionality
                            csv_export = df_questions.to_csv(index=False)
                            st.download_button(
                                "📥 Download Questions CSV",
                                csv_export,
                                f"survey_questions_{selected_survey_id}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"❌ Error analyzing survey: {str(e)}")
                    logger.error(f"Survey analysis error: {e}")
        
        # Survey list display
        st.markdown("### 📋 All Available Surveys")
        surveys_df = pd.DataFrame(surveys)
        if not surveys_df.empty:
            # Add survey categories
            surveys_df['category'] = surveys_df['title'].apply(categorize_survey)
            st.dataframe(surveys_df[['title', 'id', 'category']], use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Failed to load surveys: {str(e)}")
        logger.error(f"Survey loading error: {e}")

elif st.session_state.page == "create_survey":
    st.markdown("## ➕ Create New SurveyMonkey Survey")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token") or st.secrets.get("surveymonkey", {}).get("access_token")
        if not token:
            st.error("❌ SurveyMonkey token not found in secrets")
            st.stop()
        
        st.markdown("### 📝 Survey Configuration")
        
        # Survey basic info
        col1, col2 = st.columns(2)
        
        with col1:
            survey_title = st.text_input("Survey Title*", placeholder="Enter survey title")
            survey_nickname = st.text_input("Survey Nickname", placeholder="Optional nickname")
        
        with col2:
            survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
            survey_category = st.selectbox("Survey Category", list(SURVEY_CATEGORIES.keys()))
        
        # Survey description
        survey_description = st.text_area("Survey Description", placeholder="Optional description")
        
        if survey_title and st.button("🚀 Create Survey", type="primary"):
            with st.spinner("🔄 Creating survey..."):
                try:
                    survey_template = {
                        "title": survey_title,
                        "nickname": survey_nickname or survey_title,
                        "language": survey_language,
                        "description": survey_description,
                        "category": survey_category
                    }
                    
                    new_survey_id = create_survey(token, survey_template)
                    
                    if new_survey_id:
                        st.success(f"✅ Survey created successfully!")
                        st.info(f"**Survey ID:** {new_survey_id}")
                        st.info(f"**Category:** {survey_category}")
                        
                        # Store in session state
                        st.session_state.survey_template = survey_template
                        st.session_state.survey_template['id'] = new_survey_id
                        
                        # Next steps
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👁️ View Created Survey", use_container_width=True):
                                st.session_state.page = "view_surveys"
                                st.rerun()
                        with col2:
                            if st.button("⚙️ Configure Questions", use_container_width=True):
                                st.session_state.page = "configure_survey"
                                st.rerun()
                    else:
                        st.error("❌ Failed to create survey - no ID returned")
                
                except Exception as e:
                    st.error(f"❌ Failed to create survey: {str(e)}")
                    logger.error(f"Survey creation error: {e}")
    
    except Exception as e:
        st.error(f"❌ Error in survey creation page: {str(e)}")
        logger.error(f"Create survey page error: {e}")

elif st.session_state.page == "view_question_bank":
    st.markdown("## 📖 Complete Question Bank")
    
    if not sf_status:
        st.error("❌ Snowflake connection required for Question Bank features")
        st.info("Please check your Snowflake connection in the sidebar")
        st.stop()
    
    try:
        with st.spinner("🔄 Loading complete question bank from Snowflake..."):
            df_reference = get_all_reference_questions()
        
        if df_reference.empty:
            st.warning("⚠️ No reference data found in the database")
            st.stop()
        
        st.success(f"✅ Loaded {len(df_reference):,} questions from database")
        
        # Store in session state
        st.session_state.df_reference = df_reference
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Questions", f"{len(df_reference):,}")
        with col2:
            unique_uids = df_reference['uid'].nunique()
            st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        with col3:
            avg_per_uid = len(df_reference) / unique_uids if unique_uids > 0 else 0
            st.metric("📈 Avg per UID", f"{avg_per_uid:.1f}")
        with col4:
            # Add survey categories
            df_reference['survey_category'] = df_reference['survey_title'].apply(categorize_survey)
            categories = df_reference['survey_category'].nunique()
            st.metric("📋 Categories", categories)
        
        # Display sample data
        st.markdown("### 📝 Question Bank Sample")
        display_columns = ['uid', 'heading_0', 'survey_title', 'survey_category']
        st.dataframe(df_reference[display_columns].head(100), use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_export = df_reference.to_csv(index=False)
            st.download_button(
                "📥 Download Question Bank",
                csv_export,
                f"question_bank_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("⭐ Create Unique Questions Bank", use_container_width=True):
                st.session_state.page = "unique_question_bank"
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Failed to load question bank: {str(e)}")
        logger.error(f"Question bank loading error: {e}")

elif st.session_state.page == "unique_question_bank":
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*Best structured question for each UID with governance compliance and quality scoring*")
    
    if not sf_status:
        st.error("❌ Snowflake connection required for Question Bank features")
        st.stop()
    
    try:
        with st.spinner("🔄 Loading question bank and creating unique questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.warning("⚠️ No reference data found in the database")
                st.stop()
            
            # Create unique questions bank
            unique_questions_df = create_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.warning("⚠️ No unique questions could be created")
            st.stop()
        
        st.success(f"✅ Created unique bank with {len(unique_questions_df):,} UIDs from {len(df_reference):,} total questions")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🆔 Unique UIDs", f"{len(unique_questions_df):,}")
        with col2:
            avg_quality = unique_questions_df['quality_score'].mean()
            st.metric("⭐ Avg Quality", f"{avg_quality:.1f}")
        with col3:
            compliant_count = sum(unique_questions_df['governance_compliant'])
            compliance_rate = (compliant_count / len(unique_questions_df)) * 100
            st.metric("⚖️ Governance", f"{compliance_rate:.1f}%")
        with col4:
            categories = unique_questions_df['survey_category'].nunique()
            st.metric("📋 Categories", categories)
        
        # Display results
        st.markdown("### 📝 Unique Questions Bank")
        display_cols = ['uid', 'best_question', 'survey_category', 'total_variants', 'quality_score']
        st.dataframe(unique_questions_df[display_cols], use_container_width=True, height=400)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_export = unique_questions_df.to_csv(index=False)
            st.download_button(
                "📥 Download Unique Questions",
                csv_export,
                f"unique_questions_bank_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🧹 Data Quality Analysis", use_container_width=True):
                st.session_state.page = "data_quality"
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Failed to create unique questions bank: {str(e)}")
        logger.error(f"Unique questions bank error: {e}")

elif st.session_state.page == "categorized_questions":
    st.markdown("## 📊 Questions by Survey Category")
    
    if not sf_status:
        st.error("❌ Snowflake connection required for categorized analysis")
        st.stop()
    
    try:
        with st.spinner("🔄 Loading and categorizing questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.warning("⚠️ No reference data found")
                st.stop()
            
            # Add categories
            df_reference['survey_category'] = df_reference['survey_title'].apply(categorize_survey)
        
        st.success(f"✅ Categorized {len(df_reference):,} questions")
        
        # Category overview
        st.markdown("### 📊 Category Overview")
        category_stats = df_reference.groupby('survey_category').agg({
            'heading_0': 'count',
            'uid': 'nunique',
            'survey_title': 'nunique'
        }).rename(columns={
            'heading_0': 'Total Questions',
            'uid': 'Unique UIDs',
            'survey_title': 'Surveys'
        })
        
        st.dataframe(category_stats, use_container_width=True)
        
        # Visual analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Questions by Category**")
            st.bar_chart(category_stats['Total Questions'])
        
        with col2:
            st.markdown("**UIDs by Category**")
            st.bar_chart(category_stats['Unique UIDs'])
    
    except Exception as e:
        st.error(f"❌ Failed to categorize questions: {str(e)}")
        logger.error(f"Categorization error: {e}")

elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Configure Survey with UID Assignment")
    
    # Check prerequisites
    if not st.session_state.surveymonkey_initialized:
        st.warning("⚠️ Please initialize SurveyMonkey first by viewing surveys")
        if st.button("👁️ Go to View Surveys"):
            st.session_state.page = "view_surveys"
            st.rerun()
        st.stop()
    
    # Check if we have target data
    if st.session_state.df_target is None:
        st.info("📋 No survey selected for configuration. Please select a survey first.")
        if st.button("👁️ Select Survey"):
            st.session_state.page = "view_surveys"
            st.rerun()
        st.stop()
    
    df_target = st.session_state.df_target
    st.success(f"✅ Configuring survey with {len(df_target)} questions")
    
    # Question configuration
    st.markdown("### 📝 Question Configuration")
    
    # Display questions
    st.dataframe(df_target[['position', 'heading_0', 'schema_type', 'question_category']], 
                use_container_width=True, height=400)
    
    # Export options
    st.markdown("### 📥 Export & Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_export = df_target.to_csv(index=False)
        st.download_button(
            "📥 Download Configuration",
            csv_export,
            f"survey_config_{uuid4()}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Refresh Survey Data", use_container_width=True):
            st.session_state.df_target = None
            st.session_state.page = "view_surveys"
            st.rerun()

elif st.session_state.page == "data_quality":
    st.markdown("## 🧹 Data Quality Management")
    
    if not sf_status:
        st.error("❌ Snowflake connection required for data quality analysis")
        st.stop()
    
    try:
        with st.spinner("🔄 Loading data for quality analysis..."):
            df_reference = get_all_reference_questions()
        
        if df_reference.empty:
            st.warning("⚠️ No reference data found")
            st.stop()
        
        st.success(f"✅ Loaded {len(df_reference):,} questions for quality analysis")
        
        # Quality overview
        st.markdown("### 📊 Data Quality Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Questions", f"{len(df_reference):,}")
        
        with col2:
            unique_uids = df_reference['uid'].nunique()
            st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        
        with col3:
            avg_per_uid = len(df_reference) / unique_uids if unique_uids > 0 else 0
            st.metric("📈 Avg per UID", f"{avg_per_uid:.1f}")
        
        with col4:
            # Check governance compliance
            uid_counts = df_reference['uid'].value_counts()
            violations = sum(uid_counts > UID_GOVERNANCE['max_variations_per_uid'])
            compliance_rate = ((len(uid_counts) - violations) / len(uid_counts)) * 100 if len(uid_counts) > 0 else 100
            st.metric("⚖️ Compliance", f"{compliance_rate:.1f}%")
        
        # Quality insights
        st.markdown("### 📈 Quality Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Question Length Distribution:**")
            question_lengths = df_reference['heading_0'].str.len()
            length_bins = pd.cut(question_lengths, bins=[0, 10, 50, 100, 200, float('inf')], 
                               labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
            length_counts = length_bins.value_counts()
            st.bar_chart(length_counts)
        
        with col2:
            st.markdown("**Questions by Survey Category:**")
            df_reference['survey_category'] = df_reference['survey_title'].apply(categorize_survey)
            category_counts = df_reference['survey_category'].value_counts()
            st.bar_chart(category_counts)
    
    except Exception as e:
        st.error(f"❌ Data quality analysis failed: {str(e)}")
        logger.error(f"Data quality error: {e}")

# ============= ERROR HANDLING & FALLBACKS =============

else:
    st.error("❌ Unknown page requested")
    st.info("🏠 Redirecting to home...")
    st.session_state.page = "home"
    st.rerun()

# ============= FOOTER =============

st.markdown("---")

# Footer
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

with footer_col2:
    st.markdown("**📊 Current Session**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Init: {'✅' if st.session_state.surveymonkey_initialized else '❌'}")

with footer_col3:
    st.markdown("**⚖️ Governance Rules**")
    st.write(f"Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.write(f"Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")

# ============= END OF SCRIPT =============
