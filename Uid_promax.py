import streamlit as st
import pandas as pd
import requests
import re
import logging
import json
import time
import os
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup
st.set_page_config(
    page_title="UID Matcher Optimized", 
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
    
    .data-source-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
MAX_SURVEYS_PER_BATCH = 10

# Synonym Mapping
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "df_target": None,
        "df_final": None,
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "question_bank": None,
        "survey_template": None,
        "preview_df": None,
        "all_questions": None,
        "dedup_questions": [],
        "dedup_choices": [],
        "pending_survey": None,
        "snowflake_initialized": False,
        "surveymonkey_initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# Cached Resources
@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_snowflake_engine():
    sf = st.secrets["snowflake"]
    engine = create_engine(
        f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
        f"?warehouse={sf.warehouse}&role={sf.role}"
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT CURRENT_VERSION()"))
    return engine

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# Cache Management
def load_cached_survey_data():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            cache_time = cache.get("timestamp", 0)
            if time.time() - cache_time < 24 * 3600:
                return (
                    pd.DataFrame(cache.get("all_questions", [])),
                    cache.get("dedup_questions", []),
                    cache.get("dedup_choices", [])
                )
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None, [], []

def save_cached_survey_data(all_questions, dedup_questions, dedup_choices):
    cache = {
        "timestamp": time.time(),
        "all_questions": all_questions.to_dict(orient="records"),
        "dedup_questions": dedup_questions,
        "dedup_choices": dedup_choices
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# Calculate Matched Questions Percentage
def calculate_matched_percentage(df_final):
    if df_final is None or df_final.empty:
        return 0.0
    df_main = df_final[df_final["is_choice"] == False].copy()
    privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
    eligible_questions = df_main[privacy_filter & html_filter]
    if eligible_questions.empty:
        return 0.0
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    return round((len(matched_questions) / len(eligible_questions)) * 100, 2)

# Snowflake Queries
def run_snowflake_reference_query(limit=10000, offset=0):
    query = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
        GROUP BY HEADING_0
        LIMIT :limit OFFSET :offset
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account is locked. UID matching is disabled.")
        raise

# SurveyMonkey API
@st.cache_data
def get_surveys_cached(token):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("data", [])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.HTTPError)
)
def get_survey_details_with_retry(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests")
    response.raise_for_status()
    return response.json()

def create_survey(token, survey_template):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={
        "title": survey_template["title"],
        "nickname": survey_template["nickname"],
        "language": survey_template.get("language", "en")
    })
    response.raise_for_status()
    return response.json().get("id")

def create_page(token, survey_id, page_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={
        "title": page_template.get("title", ""),
        "description": page_template.get("description", "")
    })
    response.raise_for_status()
    return response.json().get("id")

def create_question(token, survey_id, page_id, question_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages/{page_id}/questions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "family": question_template["family"],
        "subtype": question_template["subtype"],
        "headings": [{"heading": question_template["heading"]}],
        "position": question_template["position"],
        "required": question_template.get("is_required", False)
    }
    if "choices" in question_template:
        payload["answers"] = {"choices": question_template["choices"]}
    if question_template["family"] == "matrix":
        payload["answers"] = {
            "rows": question_template.get("rows", []),
            "choices": question_template.get("choices", [])
        }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json().get("id")

def extract_questions(survey_json):
    questions = []
    global_position = 0
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            subtype = question.get("subtype", None)
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
                    "survey_title": survey_json.get("title", "")
                })
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
                            "survey_title": survey_json.get("title", "")
                        })
    return questions

# UID Matching Functions (keeping all original logic)
def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
    df_reference["norm_text"] = df_reference["heading_0"].apply(enhanced_normalize)
    df_target["norm_text"] = df_target["heading_0"].apply(enhanced_normalize)

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs = [], [], [], []
    for sim_row in similarity_matrix:
        best_idx = sim_row.argmax()
        best_score = sim_row[best_idx]
        if best_score >= TFIDF_HIGH_CONFIDENCE:
            conf = "✅ High"
        elif best_score >= TFIDF_LOW_CONFIDENCE:
            conf = "⚠️ Low"
        else:
            conf = "❌ No match"
            best_idx = None
        matched_uids.append(df_reference.iloc[best_idx]["uid"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["heading_0"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    model = load_sentence_transformer()
    emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
    emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_target, emb_ref)

    sem_matches, sem_scores = [], []
    for i in range(len(df_target)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        sem_matches.append(df_reference.iloc[best_idx]["uid"] if score >= SEMANTIC_THRESHOLD else None)
        sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

    df_target["Semantic_UID"] = sem_matches
    df_target["Semantic_Similarity"] = sem_scores
    return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )
    
    df_target["Final_UID"] = df_target.apply(
        lambda row: df_target[df_target["heading_0"] == row["parent_question"]]["Final_UID"].iloc[0]
        if row["is_choice"] and pd.notnull(row["parent_question"]) else row["Final_UID"],
        axis=1
    )
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Change_UID"] = df_target["Final_UID"].apply(
        lambda x: f"{x} - {df_reference[df_reference['uid'] == x]['heading_0'].iloc[0]}" if pd.notnull(x) and x in df_reference["uid"].values else None
    )
    
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        st.error("Input data is empty.")
        return pd.DataFrame()

    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"Processing batch {start//batch_size + 1}..."):
            batch_target = compute_tfidf_matches(df_reference, batch_target, synonym_map)
            batch_target = compute_semantic_matches(df_reference, batch_target)
            batch_target = finalize_matches(batch_target, df_reference)
            batch_target = detect_uid_conflicts(batch_target)
        df_results.append(batch_target)
    
    return pd.concat(df_results, ignore_index=True) if df_results else pd.DataFrame()

# Connection checks
def check_surveymonkey_connection():
    """Check SurveyMonkey API connection"""
    try:
        token = st.secrets.get("surveymonkey", {}).get("token")
        if not token:
            return False, "No SurveyMonkey token found"
        
        surveys = get_surveys_cached(token)
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
    st.markdown("### 🧠 UID Matcher Optimized")
    st.markdown("Navigate through the application")
    
    # Connection status
    sm_status, sm_msg = check_surveymonkey_connection()
    sf_status, sf_msg = check_snowflake_connection()
    
    st.markdown("**🔗 Connection Status**")
    st.write(f"📊 SurveyMonkey: {'✅' if sm_status else '❌'}")
    st.write(f"❄️ Snowflake: {'✅' if sf_status else '❌'}")
    
    # Data source info
    with st.expander("📊 Data Sources"):
        st.markdown("**SurveyMonkey (Source):**")
        st.markdown("• Survey data and questions")
        st.markdown("**Snowflake (Reference):**")
        st.markdown("• HEADING_0 → reference questions")
        st.markdown("• UID → target assignment")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # Survey Management
    st.markdown("**📊 Survey Management**")
    if st.button("📋 Survey Selection", use_container_width=True):
        st.session_state.page = "survey_selection"
        st.rerun()
    if st.button("🔧 UID Matching", use_container_width=True):
        st.session_state.page = "uid_matching"
        st.rerun()
    if st.button("🏗️ Survey Creation", use_container_width=True):
        st.session_state.page = "survey_creation"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank
    st.markdown("**📚 Question Bank**")
    if st.button("📖 View Question Bank", use_container_width=True):
        st.session_state.page = "question_bank"
        st.rerun()

# ============= MAIN APP HEADER =============

st.markdown('<div class="main-header">🧠 UID Matcher: SurveyMonkey Optimization</div>', unsafe_allow_html=True)

# Data source clarification
st.markdown('<div class="data-source-info"><strong>📊 Data Flow:</strong> SurveyMonkey surveys → UID matching → Snowflake reference</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Load initial data
try:
    token = st.secrets["surveymonkey"]["token"]
    surveys = get_surveys_cached(token)
    if not surveys:
        st.error("No surveys found.")
        st.stop()

    # Load cached survey data
    if st.session_state.all_questions is None:
        cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
        if cached_questions is not None:
            st.session_state.all_questions = cached_questions
            st.session_state.dedup_questions = cached_dedup_questions
            st.session_state.dedup_choices = cached_dedup_choices
            st.session_state.fetched_survey_ids = cached_questions["survey_id"].unique().tolist()

    # Load question bank
    if st.session_state.question_bank is None:
        try:
            st.session_state.question_bank = run_snowflake_reference_query()
        except Exception:
            st.warning("Failed to load question bank. Standardization checks disabled.")
            st.session_state.question_bank = pd.DataFrame(columns=["heading_0", "uid"])

except Exception as e:
    st.error(f"SurveyMonkey initialization failed: {e}")
    st.stop()

# ============= PAGE ROUTING =============

if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to UID Matcher Optimized")
    
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
                    st.metric("❄️ Snowflake UIDs", f"{count:,}")
            except:
                st.metric("❄️ Snowflake UIDs", "Error")
        else:
            st.metric("❄️ Snowflake UIDs", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 SM Surveys", len(surveys))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        matched_percentage = calculate_matched_percentage(st.session_state.df_final) if st.session_state.df_final is not None else 0
        st.metric("🎯 Matched %", f"{matched_percentage}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow guide
    st.markdown("## 🚀 Recommended Workflow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Survey Selection")
        st.markdown("Select and analyze surveys:")
        st.markdown("• Browse available surveys")
        st.markdown("• Extract questions")
        st.markdown("• Review question bank")
        
        if st.button("📋 Start Survey Selection", use_container_width=True):
            st.session_state.page = "survey_selection"
            st.rerun()
    
    with col2:
        st.markdown("### 2️⃣ UID Matching")
        st.markdown("Match questions to UIDs:")
        st.markdown("• Automatic UID matching")
        st.markdown("• Configure assignments")
        st.markdown("• Validate results")
        
        if st.button("🔧 Start UID Matching", use_container_width=True):
            st.session_state.page = "uid_matching"
            st.rerun()
    
    with col3:
        st.markdown("### 3️⃣ Survey Creation")
        st.markdown("Create new surveys:")
        st.markdown("• Design survey structure")
        st.markdown("• Configure questions")
        st.markdown("• Deploy to SurveyMonkey")
        
        if st.button("🏗️ Start Survey Creation", use_container_width=True):
            st.session_state.page = "survey_creation"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if sm_status:
            st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            st.write(f"Status: {sm_msg}")
        else:
            st.markdown('<div class="warning-card">❌ SurveyMonkey: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Error: {sm_msg}")
    
    with status_col2:
        if sf_status:
            st.markdown('<div class="success-card">✅ Snowflake: Connected</div>', unsafe_allow_html=True)
            st.write(f"Status: {sf_msg}")
        else:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Error: {sf_msg}")

elif st.session_state.page == "survey_selection":
    st.markdown("## 📋 Survey Selection & Question Bank")
    st.markdown('<div class="data-source-info">📊 <strong>Data Source:</strong> SurveyMonkey API - Survey selection and question extraction</div>', unsafe_allow_html=True)
    
    # Survey Selection
    st.markdown("### 🔍 Select Surveys")
    survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
    selected_surveys = st.multiselect("Choose surveys to analyze:", survey_options)
    selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]
    
    # Refresh button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Survey Data"):
            st.session_state.all_questions = None
            st.session_state.dedup_questions = []
            st.session_state.dedup_choices = []
            st.session_state.fetched_survey_ids = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.rerun()
    
    # Process selected surveys
    if selected_survey_ids:
        combined_questions = []
        
        # Initialize session state for fetched survey IDs if not exists
        if not hasattr(st.session_state, 'fetched_survey_ids'):
            st.session_state.fetched_survey_ids = []
        
        # Check which surveys need to be fetched
        surveys_to_fetch = [sid for sid in selected_survey_ids 
                           if sid not in st.session_state.fetched_survey_ids]
        
        if surveys_to_fetch:
            progress_bar = st.progress(0)
            for i, survey_id in enumerate(surveys_to_fetch):
                with st.spinner(f"Fetching survey {survey_id}... ({i+1}/{len(surveys_to_fetch)})"):
                    try:
                        survey_json = get_survey_details_with_retry(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                        st.session_state.fetched_survey_ids.append(survey_id)
                        time.sleep(REQUEST_DELAY)
                        progress_bar.progress((i + 1) / len(surveys_to_fetch))
                    except Exception as e:
                        st.error(f"Failed to fetch survey {survey_id}: {e}")
                        continue
            progress_bar.empty()
        
        if combined_questions:
            new_questions = pd.DataFrame(combined_questions)
            if st.session_state.all_questions is None:
                st.session_state.all_questions = new_questions
            else:
                st.session_state.all_questions = pd.concat([st.session_state.all_questions, new_questions], ignore_index=True)
            
            st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == False
            ]["heading_0"].unique().tolist())
            st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == True
            ]["heading_0"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
            
            save_cached_survey_data(
                st.session_state.all_questions,
                st.session_state.dedup_questions,
                st.session_state.dedup_choices
            )

        # Filter data for selected surveys
        if st.session_state.all_questions is not None:
            st.session_state.df_target = st.session_state.all_questions[
                st.session_state.all_questions["survey_id"].isin(selected_survey_ids)
            ].copy()
            
            if st.session_state.df_target.empty:
                st.markdown('<div class="warning-card">⚠️ No questions found for selected surveys.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card">✅ Questions loaded successfully!</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Questions", len(st.session_state.df_target))
                with col2:
                    main_questions = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                    st.metric("❓ Main Questions", main_questions)
                with col3:
                    choices = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == True])
                    st.metric("🔘 Choice Options", choices)
                
                st.markdown("### 📋 Selected Questions Preview")
                show_main_only = st.checkbox("Show main questions only", value=True)
                display_df = st.session_state.df_target[st.session_state.df_target["is_choice"] == False] if show_main_only else st.session_state.df_target
                st.dataframe(display_df[["heading_0", "schema_type", "is_choice", "survey_title"]], height=400)
                
                # Next step
                if st.button("➡️ Proceed to UID Matching", type="primary", use_container_width=True):
                    st.session_state.page = "uid_matching"
                    st.rerun()

    # Question Bank Section
    st.markdown("---")
    st.markdown("### 📚 Standardized Question Bank")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👁️ View Question Bank", use_container_width=True):
            st.session_state.page = "question_bank"
            st.rerun()
    
    with col2:
        if st.button("➕ Add to Question Bank", use_container_width=True):
            st.markdown("**Submit new questions:**")
            st.markdown("[📝 Question Submission Form](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

elif st.session_state.page == "uid_matching":
    st.markdown("## 🔧 UID Matching & Configuration")
    st.markdown('<div class="data-source-info">🔄 <strong>Process:</strong> Match survey questions → Snowflake references → Assign UIDs</div>', unsafe_allow_html=True)
    
    if st.session_state.df_target is None or st.session_state.df_target.empty:
        st.markdown('<div class="warning-card">⚠️ No survey data selected. Please select surveys first.</div>', unsafe_allow_html=True)
        if st.button("📋 Go to Survey Selection"):
            st.session_state.page = "survey_selection"
            st.rerun()
        st.stop()
    
    # Run UID Matching
    if st.session_state.df_final is None or st.button("🚀 Run UID Matching", type="primary"):
        try:
            with st.spinner("🔄 Matching UIDs with Snowflake references..."):
                if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
                    st.session_state.df_final = run_uid_match(st.session_state.question_bank, st.session_state.df_target)
                else:
                    st.session_state.df_final = st.session_state.df_target.copy()
                    st.session_state.df_final["Final_UID"] = None
        except Exception as e:
            st.markdown('<div class="warning-card">⚠️ UID matching failed. Continuing without UIDs.</div>', unsafe_allow_html=True)
            st.session_state.df_final = st.session_state.df_target.copy()
            st.session_state.df_final["Final_UID"] = None

    if st.session_state.df_final is not None:
        # Matching Results
        matched_percentage = calculate_matched_percentage(st.session_state.df_final)
        
        # Results Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Match Rate", f"{matched_percentage}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            high_conf = len(st.session_state.df_final[st.session_state.df_final.get("Match_Confidence", "") == "✅ High"])
            st.metric("✅ High Confidence", high_conf)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            low_conf = len(st.session_state.df_final[st.session_state.df_final.get("Match_Confidence", "") == "⚠️ Low"])
            st.metric("⚠️ Low Confidence", low_conf)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            no_match = len(st.session_state.df_final[st.session_state.df_final.get("Final_UID", pd.Series()).isna()])
            st.metric("❌ No Match", no_match)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 UID Matching Results")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            show_main_only = st.checkbox("Show main questions only", value=True)
        with col2:
            match_filter = st.multiselect(
                "Filter by match status:",
                ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                default=["✅ High", "⚠️ Low", "🧠 Semantic"]
            )
        with col3:
            schema_filter = st.multiselect(
                "Filter by question type:",
                ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                default=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"]
            )
        
        # Search
        search_query = st.text_input("🔍 Search questions/choices:")
        
        # Apply filters
        result_df = st.session_state.df_final.copy()
        if search_query:
            result_df = result_df[result_df["heading_0"].str.contains(search_query, case=False, na=False)]
        if match_filter and "Final_Match_Type" in result_df.columns:
            result_df = result_df[result_df["Final_Match_Type"].isin(match_filter)]
        if show_main_only:
            result_df = result_df[result_df["is_choice"] == False]
        if schema_filter:
            result_df = result_df[result_df["schema_type"].isin(schema_filter)]
        
        # Configure UIDs
        if not result_df.empty:
            uid_options = [None]
            if st.session_state.question_bank is not None:
                uid_options.extend([f"{row['uid']} - {row['heading_0']}" for _, row in st.session_state.question_bank.iterrows()])
            
            # Create required column if it doesn't exist
            if "required" not in result_df.columns:
                result_df["required"] = False
            
            display_columns = ["heading_0", "schema_type", "is_choice"]
            if "Final_UID" in result_df.columns:
                display_columns.append("Final_UID")
            if "Change_UID" in result_df.columns:
                display_columns.append("Change_UID")
            display_columns.append("required")
            
            edited_df = st.data_editor(
                result_df[display_columns],
                column_config={
                    "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                    "schema_type": st.column_config.TextColumn("Type", width="medium"),
                    "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                    "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                    "Change_UID": st.column_config.SelectboxColumn(
                        "Change UID",
                        options=uid_options,
                        default=None,
                        width="large"
                    ),
                    "required": st.column_config.CheckboxColumn("Required", width="small")
                },
                disabled=["heading_0", "schema_type", "is_choice", "Final_UID"],
                hide_index=True,
                height=400
            )
            
            # Apply UID changes
            for idx, row in edited_df.iterrows():
                if pd.notnull(row.get("Change_UID")):
                    new_uid = row["Change_UID"].split(" - ")[0]
                    st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                    st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                    st.session_state.uid_changes[idx] = new_uid
        
        # Customize Questions Section
        st.markdown("---")
        st.markdown("### ✏️ Customize Questions")
        
        if not st.session_state.df_target.empty:
            customize_df = pd.DataFrame({
                "Pre-existing Question": [None],
                "Customized Question": [""]
            })
            question_options = [None] + st.session_state.df_target[st.session_state.df_target["is_choice"] == False]["heading_0"].tolist()
            
            customize_edited_df = st.data_editor(
                customize_df,
                column_config={
                    "Pre-existing Question": st.column_config.SelectboxColumn(
                        "Pre-existing Question",
                        options=question_options,
                        default=None,
                        width="large"
                    ),
                    "Customized Question": st.column_config.TextColumn(
                        "Customized Question",
                        default="",
                        width="large"
                    )
                },
                hide_index=True,
                num_rows="dynamic"
            )
            
            # Process customizations
            for _, row in customize_edited_df.iterrows():
                if row["Pre-existing Question"] and row["Customized Question"]:
                    original_question = row["Pre-existing Question"]
                    custom_question = row["Customized Question"]
                    uid_match = st.session_state.df_final[st.session_state.df_final["heading_0"] == original_question]
                    uid = uid_match["Final_UID"].iloc[0] if not uid_match.empty else None
                    
                    new_row = pd.DataFrame({
                        "Customized Question": [custom_question],
                        "Original Question": [original_question],
                        "Final_UID": [uid]
                    })
                    st.session_state.custom_questions = pd.concat([st.session_state.custom_questions, new_row], ignore_index=True)
            
            if not st.session_state.custom_questions.empty:
                st.markdown("#### 📝 Custom Questions")
                st.dataframe(st.session_state.custom_questions, use_container_width=True)
        
        # Export Section
        st.markdown("---")
        st.markdown("### 📥 Export & Upload")
        
        export_columns = ["survey_id", "survey_title", "heading_0", "schema_type", "is_choice"]
        if "configured_final_UID" in st.session_state.df_final.columns:
            export_columns.append("configured_final_UID")
        if "required" in st.session_state.df_final.columns:
            export_columns.append("required")
        
        export_df = st.session_state.df_final[export_columns].copy()
        if "configured_final_UID" in export_df.columns:
            export_df = export_df.rename(columns={"configured_final_UID": "uid"})
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv_data,
                f"survey_with_uids_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🚀 Upload to Snowflake", use_container_width=True):
                main_questions_without_uid = export_df[
                    (export_df["is_choice"] == False) & 
                    (export_df.get("uid", pd.Series()).isna())
                ]
                
                if not main_questions_without_uid.empty:
                    st.markdown('<div class="warning-card">⚠️ All main questions must have a UID before upload.</div>', unsafe_allow_html=True)
                else:
                    try:
                        with st.spinner("⬆️ Uploading to Snowflake..."):
                            with get_snowflake_engine().connect() as conn:
                                export_df.to_sql(
                                    'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                                    conn,
                                    schema='DBT_SURVEY_MONKEY',
                                    if_exists='append',
                                    index=False
                                )
                        st.markdown('<div class="success-card">✅ Successfully uploaded to Snowflake!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Snowflake upload failed: {str(e)}</div>', unsafe_allow_html=True)

elif st.session_state.page == "survey_creation":
    st.markdown("## 🏗️ Survey Creation")
    st.markdown('<div class="data-source-info">🏗️ <strong>Process:</strong> Design survey → Configure questions → Deploy to SurveyMonkey</div>', unsafe_allow_html=True)
    
    with st.form("survey_creation_form"):
        st.markdown("### 📝 Survey Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            survey_title = st.text_input("Survey Title*", value="New Survey")
            survey_nickname = st.text_input("Survey Nickname", value=survey_title)
        with col2:
            survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
        
        st.markdown("### 📋 Questions")
        
        # Initialize edited_df in session state if it doesn't exist
        if "edited_df" not in st.session_state:
            st.session_state.edited_df = pd.DataFrame(columns=["heading_0", "schema_type", "is_choice", "required"])

        def highlight_duplicates(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if not df.empty:
                main_questions = df[df["is_choice"] == False]["heading_0"]
                duplicates = main_questions[main_questions.duplicated(keep=False)]
                if not duplicates.empty:
                    mask = (df["is_choice"] == False) & (df["heading_0"].isin(duplicates))
                    styles.loc[mask, "heading_0"] = 'background-color: #ffcccc'
            return styles

        edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config={
                "heading_0": st.column_config.SelectboxColumn(
                    "Question/Choice",
                    options=[""] + st.session_state.dedup_questions + st.session_state.dedup_choices,
                    default="",
                    width="large"
                ),
                "schema_type": st.column_config.SelectboxColumn(
                    "Question Type",
                    options=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                    default="Open-Ended",
                    width="medium"
                ),
                "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                "required": st.column_config.CheckboxColumn("Required", width="small")
            },
            hide_index=True,
            num_rows="dynamic",
            height=300
        )
        st.session_state.edited_df = edited_df

        # Validation and actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            validate_btn = st.form_submit_button("✅ Validate Questions", use_container_width=True)
        with col2:
            preview_btn = st.form_submit_button("👁️ Preview Survey", use_container_width=True)
        with col3:
            create_btn = st.form_submit_button("🚀 Create Survey", type="primary", use_container_width=True)
        
        # Process form submissions
        if validate_btn and st.session_state.question_bank is not None:
            non_standard = edited_df[~edited_df["heading_0"].isin(st.session_state.question_bank["heading_0"])]
            if not non_standard.empty:
                st.markdown('<div class="warning-card">⚠️ Non-standard questions detected:</div>', unsafe_allow_html=True)
                st.dataframe(non_standard[["heading_0"]], use_container_width=True)
                st.markdown("[📝 Submit New Questions](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
            else:
                st.markdown('<div class="success-card">✅ All questions are validated!</div>', unsafe_allow_html=True)
        
        if preview_btn or create_btn:
            if not survey_title or edited_df.empty:
                st.markdown('<div class="warning-card">⚠️ Survey title and questions are required.</div>', unsafe_allow_html=True)
            else:
                # Create survey template
                questions = []
                preview_rows = []
                position = 1
                
                for idx, row in edited_df.iterrows():
                    if not row["heading_0"]:
                        continue
                    
                    question_template = {
                        "heading": row["heading_0"].split(" - ")[0] if row["is_choice"] else row["heading_0"],
                        "position": position,
                        "is_required": row["required"]
                    }
                    
                    # Set question type
                    if row["schema_type"] == "Single Choice":
                        question_template["family"] = "single_choice"
                        question_template["subtype"] = "vertical"
                    elif row["schema_type"] == "Multiple Choice":
                        question_template["family"] = "multiple_choice"
                        question_template["subtype"] = "vertical"
                    elif row["schema_type"] == "Open-Ended":
                        question_template["family"] = "open_ended"
                        question_template["subtype"] = "essay"
                    elif row["schema_type"] == "Matrix":
                        question_template["family"] = "matrix"
                        question_template["subtype"] = "rating"
                    
                    if row["is_choice"]:
                        parent_question = row["heading_0"].split(" - ")[0]
                        parent_rows = edited_df[edited_df["heading_0"] == parent_question]
                        if not parent_rows.empty:
                            parent_idx = parent_rows.index[0]
                            if parent_idx < len(questions):
                                if "choices" not in questions[parent_idx]:
                                    questions[parent_idx]["choices"] = []
                                questions[parent_idx]["choices"].append({
                                    "text": row["heading_0"].split(" - ")[1],
                                    "position": len(questions[parent_idx]["choices"]) + 1
                                })
                    else:
                        questions.append(question_template)
                        position += 1
                    
                    preview_rows.append({
                        "position": question_template["position"],
                        "title": survey_title,
                        "nickname": survey_nickname,
                        "heading_0": row["heading_0"],
                        "schema_type": row["schema_type"],
                        "is_choice": row["is_choice"],
                        "required": row["required"]
                    })

                survey_template = {
                    "title": survey_title,
                    "nickname": survey_nickname,
                    "language": survey_language,
                    "pages": [{
                        "title": "Page 1",
                        "description": "",
                        "questions": questions
                    }]
                }

                preview_df = pd.DataFrame(preview_rows)
                
                # Add UID matching for preview
                if st.session_state.question_bank is not None:
                    uid_target = preview_df[preview_df["is_choice"] == False][["heading_0"]].copy()
                    if not uid_target.empty:
                        try:
                            uid_matched = run_uid_match(st.session_state.question_bank, uid_target)
                            preview_df = preview_df.merge(
                                uid_matched[["heading_0", "Final_UID"]],
                                on="heading_0",
                                how="left"
                            )
                        except:
                            preview_df["Final_UID"] = None
                else:
                    preview_df["Final_UID"] = None

                st.session_state.preview_df = preview_df
                st.session_state.survey_template = survey_template

                if preview_btn:
                    st.markdown('<div class="success-card">✅ Preview generated successfully!</div>', unsafe_allow_html=True)
                    st.markdown("### 📋 Survey Preview")
                    
                    # Show styled preview
                    if not preview_df.empty:
                        styled_df = preview_df.style.apply(highlight_duplicates, axis=None)
                        st.dataframe(styled_df, use_container_width=True)
                    
                    with st.expander("📄 Survey Template JSON"):
                        st.json(survey_template)

                if create_btn:
                    try:
                        with st.spinner("🏗️ Creating survey in SurveyMonkey..."):
                            survey_id = create_survey(token, survey_template)
                            
                            # Create pages and questions
                            for page_template in survey_template["pages"]:
                                page_id = create_page(token, survey_id, page_template)
                                for question_template in page_template["questions"]:
                                    create_question(token, survey_id, page_id, question_template)
                        
                        st.markdown('<div class="success-card">✅ Survey created successfully!</div>', unsafe_allow_html=True)
                        st.info(f"**Survey ID:** {survey_id}")
                        
                        st.session_state.pending_survey = {
                            "survey_id": survey_id,
                            "survey_title": survey_title,
                            "df": preview_df
                        }
                        
                        # Next steps
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("⚙️ Configure Questions", use_container_width=True):
                                st.session_state.df_final = preview_df.copy()
                                st.session_state.df_final["survey_id"] = survey_id
                                st.session_state.df_final["survey_title"] = survey_title
                                st.session_state.page = "uid_matching"
                                st.rerun()
                        with col2:
                            if st.button("📋 View All Surveys", use_container_width=True):
                                st.session_state.page = "survey_selection"
                                st.rerun()
                        
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Failed to create survey: {str(e)}</div>', unsafe_allow_html=True)

elif st.session_state.page == "question_bank":
    st.markdown("## 📖 Question Bank Viewer")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Snowflake reference database</div>', unsafe_allow_html=True)
    
    if st.session_state.question_bank is None or st.session_state.question_bank.empty:
        st.markdown('<div class="warning-card">⚠️ Question bank not available. Check Snowflake connection.</div>', unsafe_allow_html=True)
        st.stop()
    
    # Question bank overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total Questions", f"{len(st.session_state.question_bank):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_uids = st.session_state.question_bank["uid"].nunique()
        st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_per_uid = len(st.session_state.question_bank) / unique_uids if unique_uids > 0 else 0
        st.metric("📈 Avg per UID", f"{avg_per_uid:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Search functionality
    st.markdown("### 🔍 Search Question Bank")
    search_query = st.text_input("Search questions or UIDs:")
    
    display_df = st.session_state.question_bank.copy()
    if search_query:
        mask = (display_df["heading_0"].str.contains(search_query, case=False, na=False) |
                display_df["uid"].str.contains(search_query, case=False, na=False))
        display_df = display_df[mask]
    
    # Display question bank
    st.dataframe(display_df[["uid", "heading_0"]], use_container_width=True, height=400)
    
    # Export option
    if st.button("📥 Download Question Bank", use_container_width=True):
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"question_bank_{uuid4()}.csv",
            "text/csv"
        )

else:
    st.error("❌ Unknown page requested")
    st.info("🏠 Redirecting to home...")
    st.session_state.page = "home"
    st.rerun()

# ============= FOOTER =============

st.markdown("---")

# Footer with quick links and status
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

with footer_col2:
    st.markdown("**📊 Data Sources**")
    st.write("📊 SurveyMonkey: Surveys & Questions")
    st.write("❄️ Snowflake: UIDs & References")

with footer_col3:
    st.markdown("**📊 Current Session**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Status: {'✅' if sm_status else '❌'}")
    st.write(f"SF Status: {'✅' if sf_status else '❌'}")

# ============= END OF SCRIPT =============