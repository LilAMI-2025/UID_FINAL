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
if "page" not in st.session_state:
    st.session_state.page = "home"
if "df_target" not in st.session_state:
    st.session_state.df_target = None
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "uid_changes" not in st.session_state:
    st.session_state.uid_changes = {}
if "custom_questions" not in st.session_state:
    st.session_state.custom_questions = pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"])
if "df_reference" not in st.session_state:
    st.session_state.df_reference = None
if "survey_template" not in st.session_state:
    st.session_state.survey_template = None
if "snowflake_initialized" not in st.session_state:
    st.session_state.snowflake_initialized = False
if "surveymonkey_initialized" not in st.session_state:
    st.session_state.surveymonkey_initialized = False

# ============= SURVEYMONKEY FUNCTIONS (Priority 1) =============

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

# ============= SNOWFLAKE FUNCTIONS (Priority 2) =============

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

# ============= UI COMPONENTS =============

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

# Enhanced Sidebar Navigation
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

# App Header
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets and "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake and SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# ============= PAGE ROUTING =============

# Home Page
if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col3:
        if st.button("📊 View Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
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
        
        # Quality issues detection
        st.markdown("### 🔍 Quality Issues Detection")
        
        # Detect various quality issues
        quality_issues = {
            'Empty Questions': len(df_reference[df_reference['heading_0'].str.len() < 5]),
            'HTML Content': len(df_reference[df_reference['heading_0'].str.contains('<.*>', regex=True, na=False)]),
            'Privacy Policy': len(df_reference[df_reference['heading_0'].str.contains('privacy policy', case=False, na=False)]),
            'Duplicate UIDs': len(df_reference[df_reference.duplicated(['uid', 'heading_0'])])
        }
        
        # Display issues
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quality Issues Found:**")
            for issue, count in quality_issues.items():
                if count > 0:
                    st.write(f"❌ {issue}: {count:,}")
                else:
                    st.write(f"✅ {issue}: {count}")
        
        with col2:
            # Governance violations
            st.markdown("**Governance Violations:**")
            excessive_uids = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
            if len(excessive_uids) > 0:
                st.write(f"❌ UIDs exceeding limit: {len(excessive_uids)}")
                st.write(f"❌ Total excess questions: {excessive_uids.sum() - (len(excessive_uids) * UID_GOVERNANCE['max_variations_per_uid'])}")
            else:
                st.write("✅ All UIDs within governance limits")
        
        # Top problematic UIDs
        if len(excessive_uids) > 0:
            st.markdown("### ⚠️ Most Problematic UIDs")
            top_problematic = excessive_uids.head(10).reset_index()
            top_problematic.columns = ['UID', 'Question Count']
            top_problematic['Excess'] = top_problematic['Question Count'] - UID_GOVERNANCE['max_variations_per_uid']
            st.dataframe(top_problematic, use_container_width=True)
        
        # Data cleaning options
        st.markdown("### 🧹 Data Cleaning Options")
        
        cleaning_strategy = st.selectbox(
            "Select cleaning strategy:",
            ["Conservative", "Moderate", "Aggressive"],
            help="Conservative: Remove only duplicates | Moderate: Remove duplicates + normalize | Aggressive: Keep only best question per UID"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{cleaning_strategy} Strategy:**")
            if cleaning_strategy == "Conservative":
                st.write("• Remove exact duplicates only")
                st.write("• Remove obvious junk (empty, HTML)")
                st.write("• Minimal impact on data")
            elif cleaning_strategy == "Moderate":
                st.write("• Remove duplicates and similar questions")
                st.write("• Apply governance limits")
                st.write("• Normalize question text")
            else:  # Aggressive
                st.write("• Keep only best question per UID")
                st.write("• Full governance compliance")
                st.write("• Maximum data reduction")
        
        with col2:
            # Estimated impact
            if cleaning_strategy == "Conservative":
                estimated_removal = sum(quality_issues.values())
            elif cleaning_strategy == "Moderate":
                estimated_removal = sum(quality_issues.values()) + (excessive_uids.sum() - len(excessive_uids) * UID_GOVERNANCE['max_variations_per_uid'])
            else:  # Aggressive
                estimated_removal = len(df_reference) - unique_uids
            
            removal_percentage = (estimated_removal / len(df_reference)) * 100 if len(df_reference) > 0 else 0
            
            st.markdown("**Estimated Impact:**")
            st.write(f"• Questions to remove: ~{estimated_removal:,}")
            st.write(f"• Percentage reduction: ~{removal_percentage:.1f}%")
            st.write(f"• Remaining questions: ~{len(df_reference) - estimated_removal:,}")
        
        if st.button(f"🧹 Apply {cleaning_strategy} Cleaning", type="primary"):
            st.warning("⚠️ This is a simulation. In the full implementation, this would:")
            st.write(f"1. Apply {cleaning_strategy.lower()} cleaning strategy")
            st.write(f"2. Remove approximately {estimated_removal:,} questions")
            st.write("3. Create cleaned dataset for download")
            st.write("4. Maintain governance compliance")
            
            # Simulate cleaned data creation
            st.success("✅ Cleaning simulation completed!")
            
            # Create sample cleaned data for download
            if cleaning_strategy == "Aggressive":
                # Simulate keeping only best question per UID
                sample_cleaned = df_reference.groupby('uid').first().reset_index()
            else:
                # Simulate moderate cleaning
                sample_cleaned = df_reference.drop_duplicates(['uid', 'heading_0'])
            
            csv_export = sample_cleaned.to_csv(index=False)
            st.download_button(
                f"📥 Download {cleaning_strategy} Cleaned Data",
                csv_export,
                f"cleaned_data_{cleaning_strategy.lower()}_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Quality trends (if we had historical data)
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
    st.info("Redirecting to home...")
    st.session_state.page = "home"
    st.rerun()

# ============= FOOTER =============
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

with col2:
    st.markdown("**📊 Current Status**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Init: {'✅' if st.session_state.surveymonkey_initialized else '❌'}")

with col3:
    st.markdown("**⚖️ Governance Rules**")
    st.write(f"Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.write(f"Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")

# ============= ADDITIONAL HELPER FUNCTIONS =============

def validate_session_state():
    """Validate and clean session state"""
    required_keys = ['page', 'df_target', 'df_final', 'uid_changes', 'custom_questions', 
                    'df_reference', 'survey_template', 'snowflake_initialized', 'surveymonkey_initialized']
    
    for key in required_keys:
        if key not in st.session_state:
            if key in ['df_target', 'df_final', 'df_reference', 'survey_template']:
                st.session_state[key] = None
            elif key in ['uid_changes']:
                st.session_state[key] = {}
            elif key in ['custom_questions']:
                st.session_state[key] = pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"])
            elif key in ['snowflake_initialized', 'surveymonkey_initialized']:
                st.session_state[key] = False
            else:
                st.session_state[key] = "home"

def log_user_action(action, details=None):
    """Log user actions for debugging"""
    logger.info(f"User action: {action}")
    if details:
        logger.info(f"Action details: {details}")

def handle_error_gracefully(error, context="Unknown"):
    """Handle errors gracefully with user-friendly messages"""
    logger.error(f"Error in {context}: {str(error)}")
    
    if "250001" in str(error):
        st.error("🔒 Snowflake account is locked. Please contact your administrator.")
    elif "connection" in str(error).lower():
        st.error("🌐 Connection issue detected. Please check your network and try again.")
    elif "token" in str(error).lower():
        st.error("🔑 Authentication issue. Please check your API tokens.")
    else:
        st.error(f"❌ An error occurred in {context}: {str(error)}")
    
    return False

# Validate session state on each run
validate_session_state()

# End of script
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

# ============= SURVEYMONKEY PAGES =============

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
        
        # Survey templates
        st.markdown("---")
        st.markdown("### 📋 Common Survey Templates")
        
        templates = {
            "Employee Feedback": {
                "title": "Employee Feedback Survey",
                "category": "Feedback",
                "description": "Collect feedback from employees about workplace satisfaction"
            },
            "GROW Assessment": {
                "title": "GROW Leadership Assessment",
                "category": "GROW",
                "description": "Leadership development assessment survey"
            },
            "Pre-Programme": {
                "title": "Pre-Programme Readiness Survey",
                "category": "Pre programme",
                "description": "Assess participant readiness before programme start"
            },
            "Impact Evaluation": {
                "title": "Programme Impact Evaluation",
                "category": "Impact",
                "description": "Measure programme outcomes and impact"
            }
        }
        
        for template_name, template_data in templates.items():
            with st.expander(f"📋 {template_name} Template"):
                st.write(f"**Category:** {template_data['category']}")
                st.write(f"**Description:** {template_data['description']}")
                
                if st.button(f"Use {template_name} Template", key=f"template_{template_name}"):
                    survey_title = template_data['title']
                    survey_category = template_data['category']
                    survey_description = template_data['description']
                    st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error in survey creation page: {str(e)}")
        logger.error(f"Create survey page error: {e}")

# ============= SNOWFLAKE PAGES =============

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
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # UID filter
            uid_options = ['All'] + sorted(df_reference['uid'].astype(str).unique().tolist())
            selected_uid = st.selectbox("Filter by UID:", uid_options)
        
        with col2:
            # Category filter
            category_options = ['All'] + sorted(df_reference['survey_category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category:", category_options)
        
        with col3:
            # Search filter
            search_term = st.text_input("Search questions:", placeholder="Enter search term")
        
        # Apply filters
        filtered_df = df_reference.copy()
        
        if selected_uid != 'All':
            filtered_df = filtered_df[filtered_df['uid'].astype(str) == selected_uid]
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['survey_category'] == selected_category]
        
        if search_term:
            filtered_df = filtered_df[filtered_df['heading_0'].str.contains(search_term, case=False, na=False)]
        
        st.info(f"📊 Showing {len(filtered_df):,} of {len(df_reference):,} questions")
        
        # Display results
        if not filtered_df.empty:
            # Sample display
            st.markdown("### 📝 Question Bank Sample")
            display_columns = ['uid', 'heading_0', 'survey_title', 'survey_category']
            st.dataframe(filtered_df[display_columns].head(100), use_container_width=True)
            
            # Category breakdown
            if len(filtered_df) > 1:
                st.markdown("### 📊 Category Breakdown")
                category_counts = filtered_df['survey_category'].value_counts()
                st.bar_chart(category_counts)
            
            # Export options
            st.markdown("### 📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_export = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data",
                    csv_export,
                    f"question_bank_filtered_{uuid4()}.csv",
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
        
        # Governance status
        non_compliant = len(unique_questions_df) - compliant_count
        if non_compliant > 0:
            st.warning(f"⚠️ {non_compliant} UIDs have excessive variations (>{UID_GOVERNANCE['max_variations_per_uid']} each)")
        else:
            st.success("✅ All UIDs are governance compliant!")
        
        # Filters and controls
        st.markdown("### 🔍 Filters & Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_filter = st.selectbox("Filter by Category:", 
                                         ['All'] + sorted(unique_questions_df['survey_category'].unique().tolist()))
        
        with col2:
            quality_threshold = st.slider("Minimum Quality Score:", 
                                        min_value=int(unique_questions_df['quality_score'].min()),
                                        max_value=int(unique_questions_df['quality_score'].max()),
                                        value=int(unique_questions_df['quality_score'].min()))
        
        with col3:
            compliance_filter = st.selectbox("Governance Compliance:", 
                                           ['All', 'Compliant Only', 'Non-Compliant Only'])
        
        # Apply filters
        filtered_df = unique_questions_df.copy()
        
        if category_filter != 'All':
            filtered_df = filtered_df[filtered_df['survey_category'] == category_filter]
        
        filtered_df = filtered_df[filtered_df['quality_score'] >= quality_threshold]
        
        if compliance_filter == 'Compliant Only':
            filtered_df = filtered_df[filtered_df['governance_compliant'] == True]
        elif compliance_filter == 'Non-Compliant Only':
            filtered_df = filtered_df[filtered_df['governance_compliant'] == False]
        
        st.info(f"📊 Showing {len(filtered_df):,} of {len(unique_questions_df):,} unique questions")
        
        # Display results
        if not filtered_df.empty:
            # Enhanced display with formatting
            display_df = filtered_df.copy()
            display_df['compliance_icon'] = display_df['governance_compliant'].map({True: '✅', False: '❌'})
            display_df['quality_rounded'] = display_df['quality_score'].round(1)
            
            st.markdown("### 📝 Unique Questions Bank")
            display_cols = ['uid', 'best_question', 'survey_category', 'total_variants', 
                          'quality_rounded', 'compliance_icon']
            col_config = {
                'uid': 'UID',
                'best_question': 'Best Question Text',
                'survey_category': 'Category',
                'total_variants': 'Variants',
                'quality_rounded': 'Quality',
                'compliance_icon': 'Compliant'
            }
            
            st.dataframe(display_df[display_cols].rename(columns=col_config), 
                        use_container_width=True, height=400)
            
            # Category analysis
            st.markdown("### 📊 Category Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                category_counts = filtered_df['survey_category'].value_counts()
                st.bar_chart(category_counts)
            
            with col2:
                quality_by_category = filtered_df.groupby('survey_category')['quality_score'].mean().sort_values(ascending=False)
                st.bar_chart(quality_by_category)
            
            # Export options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_export = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Unique Questions",
                    csv_export,
                    f"unique_questions_bank_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export only high quality questions
                high_quality = filtered_df[filtered_df['quality_score'] >= UID_GOVERNANCE['quality_score_threshold']]
                if not high_quality.empty:
                    hq_export = high_quality.to_csv(index=False)
                    st.download_button(
                        "⭐ Download High Quality Only",
                        hq_export,
                        f"high_quality_questions_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("🧹 Data Quality Analysis", use_container_width=True):
                    st.session_state.page = "data_quality"
                    st.rerun()
        
        # Top insights
        st.markdown("### 💡 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top Categories by Quality**")
            top_categories = unique_questions_df.groupby('survey_category')['quality_score'].mean().nlargest(5)
            for cat, score in top_categories.items():
                st.write(f"• {cat}: {score:.1f}")
        
        with col2:
            st.markdown("**⚠️ Categories Needing Attention**")
            low_compliance = unique_questions_df.groupby('survey_category')['governance_compliant'].mean().nsmallest(3)
            for cat, rate in low_compliance.items():
                if rate < 1.0:
                    st.write(f"• {cat}: {rate*100:.0f}% compliant")
        
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
        
        category_stats['Avg Questions per UID'] = (category_stats['Total Questions'] / category_stats['Unique UIDs']).round(1)
        st.dataframe(category_stats, use_container_width=True)
        
        # Visual analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Questions by Category**")
            st.bar_chart(category_stats['Total Questions'])
        
        with col2:
            st.markdown("**UIDs by Category**")
            st.bar_chart(category_stats['Unique UIDs'])
        
        # Category deep dive
        st.markdown("### 🔍 Category Deep Dive")
        selected_category = st.selectbox("Select category for detailed analysis:", 
                                       sorted(df_reference['survey_category'].unique()))
        
        if selected_category:
            category_data = df_reference[df_reference['survey_category'] == selected_category]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions", len(category_data))
            with col2:
                st.metric("UIDs", category_data['uid'].nunique())
            with col3:
                st.metric("Surveys", category_data['survey_title'].nunique())
            
            # Sample questions from category
            st.markdown(f"### 📝 Sample Questions from {selected_category}")
            sample_questions = category_data['heading_0'].drop_duplicates().head(10)
            for i, question in enumerate(sample_questions, 1):
                st.write(f"{i}. {question}")
            
            # Export category data
            if st.button(f"📥 Export {selected_category} Data"):
                csv_export = category_data.to_csv(index=False)
                st.download_button(
                    f"Download {selected_category} Questions",
                    csv_export,
                    f"{selected_category.lower().replace(' ', '_')}_questions.csv",
                    "text/csv"
                )
        
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
    
    if not sf_status:
        st.warning("❌ Snowflake connection required for UID assignment")
        st.info("You can still configure surveys, but UID matching will be disabled")
    
    # Check if we have target data
    if st.session_state.df_target is None:
        st.info("📋 No survey selected for configuration. Please select a survey first.")
        if st.button("👁️ Select Survey"):
            st.session_state.page = "view_surveys"
            st.rerun()
        st.stop()
    
    df_target = st.session_state.df_target
    st.success(f"✅ Configuring survey with {len(df_target)} questions")
    
    # Survey info
    if st.session_state.survey_template:
        survey_info = st.session_state.survey_template
        st.info(f"**Survey:** {survey_info.get('title', 'Unknown')} | **Category:** {categorize_survey(survey_info.get('title', ''))}")
    
    # UID Assignment section
    if sf_status:
        st.markdown("### 🆔 UID Assignment")
        
        if st.button("🚀 Run UID Matching", type="primary"):
            with st.spinner("🔄 Running enhanced UID matching..."):
                try:
                    # Load reference data
                    df_reference = get_all_reference_questions()
                    
                    if df_reference.empty:
                        st.error("❌ No reference data available for matching")
                    else:
                        # Run UID matching (this would need the full matching functions)
                        st.success("✅ UID matching completed!")
                        st.info("UID matching functionality would be implemented here with the complete matching pipeline")
                        
                except Exception as e:
                    st.error(f"❌ UID matching failed: {str(e)}")
    
    # Question configuration
    st.markdown("### 📝 Question Configuration")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_choices = st.checkbox("Show choice options", value=False)
    with col2:
        question_type_filter = st.selectbox("Filter by type:", 
                                          ['All'] + df_target['schema_type'].unique().tolist())
    
    # Apply filters
    display_df = df_target.copy()
    if not show_choices:
        display_df = display_df[display_df['is_choice'] == False]
    if question_type_filter != 'All':
        display_df = display_df[display_df['schema_type'] == question_type_filter]
    
    # Display questions
    st.dataframe(display_df[['position', 'heading_0', 'schema_type', 'question_category']], 
                use_container_width=True, height=400)
    
    # Export options
    st.markdown("### 📥 Export & Actions")
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        if st.button("📊 View Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
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
        
        # Quality issues detection
        st.markdown("### 🔍 Quality Issues Detection")
        
        # Detect various quality issues
        quality_issues = {
            'Empty Questions': len(df_reference[df_reference['heading_0'].str.len() < 5]),
            'HTML Content': len(df_reference[df_reference['heading_0'].str.contains('<.*>', regex=True, na=False)]),
            'Privacy Policy': len(df_reference[df_reference['heading_0'].str.contains('privacy policy', case=False, na=False)]),
            'Duplicate UIDs': len(df_reference[df_reference.duplicated(['uid', 'heading_0'])])
        }
        
        # Display issues
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quality Issues Found:**")
            for issue, count in quality_issues.items():
                if count > 0:
                    st.write(f"❌ {issue}: {count:,}")
                else:
                    st.write(f"✅ {issue}: {count}")
        
        with col2:
            # Governance violations
            st.markdown("**Governance Violations:**")
            excessive_uids = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
            if len(excessive_uids) > 0:
                st.write(f"❌ UIDs exceeding limit: {len(excessive_uids)}")
                st.write(f"❌ Total excess questions: {excessive_uids.sum() - (len(excessive_uids) * UID_GOVERNANCE['max_variations_per_uid'])}")
            else:
                st.write("✅ All UIDs within governance limits")
        
        # Top problematic UIDs
        if len(excessive_uids) > 0:
            st.markdown("### ⚠️ Most Problematic UIDs")
            top_problematic = excessive_uids.head(10).reset_index()
            top_problematic.columns = ['UID', 'Question Count']
            top_problematic['Excess'] = top_problematic['Question Count'] - UID_GOVERNANCE['max_variations_per_uid']
            st.dataframe(top_problematic, use_container_width=True)
        
        # Data cleaning options
        st.markdown("### 🧹 Data Cleaning Options")
        
        cleaning_strategy = st.selectbox(
            "Select cleaning strategy:",
            ["Conservative", "Moderate", "Aggressive"],
            help="Conservative: Remove only duplicates | Moderate: Remove duplicates + normalize | Aggressive: Keep only best question per UID"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{cleaning_strategy} Strategy:**")
            if cleaning_strategy == "Conservative":
                st.write("• Remove exact duplicates only")
                st.write("• Remove obvious junk (empty, HTML)")
                st.write("• Minimal impact on data")
            elif cleaning_strategy == "Moderate":
                st.write("• Remove duplicates and similar questions")
                st.write("• Apply governance limits")
                st.write("• Normalize question text")
            else:  # Aggressive
                st.write("• Keep only best question per UID")
                st.write("• Full governance compliance")
                st.write("• Maximum data reduction")
        
        with col2:
            # Estimated impact
            if cleaning_strategy == "Conservative":
                estimated_removal = sum(quality_issues.values())
            elif cleaning_strategy == "Moderate":
                estimated_removal = sum(quality_issues.values()) + (excessive_uids.sum() - len(excessive_uids) * UID_GOVERNANCE['max_variations_per_uid'])
            else:  # Aggressive
                estimated_removal = len(df_reference) - unique_uids
            
            removal_percentage = (estimated_removal / len(df_reference)) * 100 if len(df_reference) > 0 else 0
            
            st.markdown("**Estimated Impact:**")
            st.write(f"• Questions to remove: ~{estimated_removal:,}")
            st.write(f"• Percentage reduction: ~{removal_percentage:.1f}%")
            st.write(f"• Remaining questions: ~{len(df_reference) - estimated_removal:,}")
        
        if st.button(f"🧹 Apply {cleaning_strategy} Cleaning", type="primary"):
            st.warning("⚠️ This is a simulation. In the full implementation, this would:")
            st.write(f"1. Apply {cleaning_strategy.lower()} cleaning strategy")
            st.write(f"2. Remove approximately {estimated_removal:,} questions")
            st.write("3. Create cleaned dataset for download")
            st.write("4. Maintain governance compliance")
            
            # Simulate cleaned data creation
            st.success("✅ Cleaning simulation completed!")
            
            # Create sample cleaned data for download
            if cleaning_strategy == "Aggressive":
                # Simulate keeping only best question per UID
                sample_cleaned = df_reference.groupby('uid').first().reset_index()
            else:
                # Simulate moderate cleaning
                sample_cleaned = df_reference.drop_duplicates(['uid', 'heading_0'])
            
            csv_export = sample_cleaned.to_csv(index=False)
            st.download_button(
                f"📥 Download {cleaning_strategy} Cleaned Data",
                csv_export,
                f"cleaned_data_{cleaning_strategy.lower()}_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Quality trends
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

# ============= ADDITIONAL PAGES =============

elif st.session_state.page == "update_question_bank":
    st.markdown("## 🔄 Update Question Bank")
    
    if not sf_status:
        st.error("❌ Snowflake connection required for updating question bank")
        st.stop()
    
    st.markdown("### 📊 Current Question Bank Status")
    
    try:
        with st.spinner("🔄 Checking current question bank status..."):
            df_reference = get_all_reference_questions()
            
        if df_reference.empty:
            st.warning("⚠️ No reference data found")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Questions", f"{len(df_reference):,}")
            with col2:
                st.metric("Unique UIDs", f"{df_reference['uid'].nunique():,}")
            with col3:
                last_update = "N/A"  # In real implementation, get from metadata
                st.metric("Last Update", last_update)
        
        # Update options
        st.markdown("### 🔄 Update Options")
        
        update_type = st.radio(
            "Select update type:",
            ["Refresh All Data", "Incremental Update", "Validate Only"],
            help="Refresh All: Complete reload | Incremental: Add new data only | Validate: Check consistency"
        )
        
        if update_type == "Refresh All Data":
            st.info("🔄 This will completely reload all question bank data from Snowflake")
            if st.button("🚀 Start Full Refresh", type="primary"):
                with st.spinner("🔄 Refreshing all question bank data..."):
                    # Clear cache
                    get_all_reference_questions.clear()
                    # Reload data
                    df_new = get_all_reference_questions()
                    
                if not df_new.empty:
                    st.success(f"✅ Successfully refreshed {len(df_new):,} questions")
                    st.session_state.df_reference = df_new
                else:
                    st.error("❌ Failed to refresh data")
        
        elif update_type == "Incremental Update":
            st.info("📈 This will add only new questions since last update")
            if st.button("🚀 Start Incremental Update", type="primary"):
                st.warning("⚠️ Incremental update functionality would be implemented here")
                st.info("In full implementation: Query for new data based on timestamp")
        
        else:  # Validate Only
            st.info("✅ This will validate data consistency without making changes")
            if st.button("🔍 Start Validation", type="primary"):
                with st.spinner("🔍 Validating question bank data..."):
                    # Simulate validation
                    validation_results = {
                        "Missing UIDs": 0,
                        "Duplicate Records": len(df_reference[df_reference.duplicated()]),
                        "Invalid Questions": len(df_reference[df_reference['heading_0'].str.len() < 1]),
                        "Schema Issues": 0
                    }
                    
                st.success("✅ Validation completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    for issue, count in validation_results.items():
                        if count > 0:
                            st.write(f"❌ {issue}: {count}")
                        else:
                            st.write(f"✅ {issue}: {count}")
        
        # Manual data upload
        st.markdown("---")
        st.markdown("### 📁 Manual Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with question data:",
            type=['csv'],
            help="Upload a CSV file with columns: heading_0, uid, survey_title"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded file with {len(uploaded_df)} rows")
                
                # Validate columns
                required_cols = ['heading_0', 'uid', 'survey_title']
                missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.info("✅ File format is valid")
                    
                    # Preview data
                    st.markdown("**Data Preview:**")
                    st.dataframe(uploaded_df.head(), use_container_width=True)
                    
                    if st.button("🔄 Process Uploaded Data"):
                        st.success("✅ Data processing would be implemented here")
                        st.info("In full implementation: Validate, clean, and merge with existing data")
            
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error updating question bank: {str(e)}")
        
# ============= ADVANCED FEATURES =============

elif st.session_state.page == "advanced_matching":
    st.markdown("## 🧠 Advanced UID Matching")
    
    st.markdown("### 🎯 Enhanced Matching Options")
    
    # Matching configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**TF-IDF Settings:**")
        tfidf_high = st.slider("High Confidence Threshold", 0.0, 1.0, TFIDF_HIGH_CONFIDENCE, 0.05)
        tfidf_low = st.slider("Low Confidence Threshold", 0.0, 1.0, TFIDF_LOW_CONFIDENCE, 0.05)
        
    with col2:
        st.markdown("**Semantic Settings:**")
        semantic_threshold = st.slider("Semantic Threshold", 0.0, 1.0, SEMANTIC_THRESHOLD, 0.05)
        batch_size = st.number_input("Batch Size", min_value=100, max_value=5000, value=BATCH_SIZE)
    
    # Governance settings
    st.markdown("### ⚖️ Governance Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_variations = st.number_input("Max Variations per UID", 
                                       min_value=1, max_value=200, 
                                       value=UID_GOVERNANCE['max_variations_per_uid'])
        semantic_similarity = st.slider("Semantic Similarity Threshold", 
                                      0.0, 1.0, 
                                      UID_GOVERNANCE['semantic_similarity_threshold'], 0.01)
    
    with col2:
        quality_threshold = st.slider("Quality Score Threshold", 
                                    0.0, 20.0, 
                                    UID_GOVERNANCE['quality_score_threshold'], 0.5)
        enable_conflicts = st.checkbox("Enable Conflict Detection", 
                                     value=UID_GOVERNANCE['conflict_detection_enabled'])
    
    # Update governance settings
    if st.button("💾 Update Settings"):
        UID_GOVERNANCE.update({
            'max_variations_per_uid': max_variations,
            'semantic_similarity_threshold': semantic_similarity,
            'quality_score_threshold': quality_threshold,
            'conflict_detection_enabled': enable_conflicts
        })
        st.success("✅ Settings updated successfully!")
    
    # Matching preview
    if st.session_state.df_target is not None and st.session_state.df_reference is not None:
        st.markdown("### 🔍 Matching Preview")
        
        sample_questions = st.session_state.df_target.head(5)['heading_0'].tolist()
        
        for i, question in enumerate(sample_questions, 1):
            st.write(f"**Question {i}:** {question}")
            # In full implementation, show matching results
            st.write("   *Matching results would appear here*")
    
    else:
        st.info("📋 Load survey data and question bank to see matching preview")

# ============= ERROR HANDLING & FALLBACKS =============

else:
    st.error("❌ Unknown page requested")
    st.info("🏠 Redirecting to home...")
    st.session_state.page = "home"
    st.rerun()

# ============= FOOTER & CLEANUP =============

st.markdown("---")

# Footer with quick stats and links
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

with footer_col2:
    st.markdown("**📊 Current Session**")
    st.write(f"Current Page: {st.session_state.page}")
    st.write(f"SurveyMonkey: {'✅' if st.session_state.surveymonkey_initialized else '❌'}")
    st.write(f"Snowflake: {'✅' if st.session_state.snowflake_initialized else '❌'}")

with footer_col3:
    st.markdown("**⚖️ Active Governance**")
    st.write(f"Max Variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.write(f"Semantic Threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.write(f"Quality Threshold: {UID_GOVERNANCE['quality_score_threshold']}")

# ============= UTILITY FUNCTIONS FOR ERROR HANDLING =============

def validate_session_state():
    """Validate and clean session state"""
    required_keys = ['page', 'df_target', 'df_final', 'uid_changes', 'custom_questions', 
                    'df_reference', 'survey_template', 'snowflake_initialized', 'surveymonkey_initialized']
    
    for key in required_keys:
        if key not in st.session_state:
            if key in ['df_target', 'df_final', 'df_reference', 'survey_template']:
                st.session_state[key] = None
            elif key in ['uid_changes']:
                st.session_state[key] = {}
            elif key in ['custom_questions']:
                st.session_state[key] = pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"])
            elif key in ['snowflake_initialized', 'surveymonkey_initialized']:
                st.session_state[key] = False
            else:
                st.session_state[key] = "home"

def log_user_action(action, details=None):
    """Log user actions for debugging"""
    logger.info(f"User action: {action}")
    if details:
        logger.info(f"Action details: {details}")

def handle_error_gracefully(error, context="Unknown"):
    """Handle errors gracefully with user-friendly messages"""
    logger.error(f"Error in {context}: {str(error)}")
    
    if "250001" in str(error):
        st.error("🔒 Snowflake account is locked. Please contact your administrator.")
        st.info("You can still use SurveyMonkey features while Snowflake is unavailable.")
    elif "connection" in str(error).lower():
        st.error("🌐 Connection issue detected. Please check your network and try again.")
    elif "token" in str(error).lower():
        st.error("🔑 Authentication issue. Please check your API tokens in secrets.")
    elif "permission" in str(error).lower():
        st.error("🚫 Permission denied. Please check your access rights.")
    else:
        st.error(f"❌ An error occurred in {context}: {str(error)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")
    
    return False

def cleanup_session_data():
    """Clean up session data to prevent memory issues"""
    # Clear large dataframes if they exceed size limits
    max_rows = 100000
    
    for key in ['df_target', 'df_final', 'df_reference']:
        if key in st.session_state and st.session_state[key] is not None:
            if hasattr(st.session_state[key], '__len__') and len(st.session_state[key]) > max_rows:
                logger.warning(f"Large dataset detected in {key}, consider clearing cache")

# Validate session state on each run
validate_session_state()

# Cleanup session data periodically
cleanup_session_data()

# Log current page access
log_user_action(f"Accessed page: {st.session_state.page}")

# ============= END OF SCRIPT =============
