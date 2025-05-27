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
import pickle
import time
from datetime import datetime, timedelta

# Ensure st.set_page_config is the first Streamlit command
_PAGE_CONFIG_SET = False
if not _PAGE_CONFIG_SET:
    try:
        st.set_page_config(
            page_title="UID Matcher Pro",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🧠"
        )
        _PAGE_CONFIG_SET = True
    except Exception as e:
        logging.error(f"Failed to set page config: {e}")
        st.error(f"Page configuration error: {str(e)}")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DURATION = 3600  # 1 hour
EMBEDDING_CACHE_SIZE = 50000
BATCH_SIZE = 100
SEMANTIC_THRESHOLD = 0.75
UID_GOVERNANCE = {'conflict_resolution_threshold': 10}
ENHANCED_SYNONYM_MAP = {}  # Placeholder for synonym map

# Custom CSS for UI
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Performance Monitoring
def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper

# Sidebar navigation
st.sidebar.title("🧠 UID Matcher Pro")
page = st.sidebar.selectbox(
    "Navigate",
    ["Home Dashboard", "View Surveys", "Create Survey", "Configure Survey",
     "Build Question Bank", "Optimized 1:1 Question Bank", "Matching Dashboard", "Settings"]
)
st.session_state.page = page

# Initialize session state
if 'snowflake_engine' not in st.session_state:
    st.session_state.snowflake_engine = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'primary_matching_reference' not in st.session_state:
    st.session_state.primary_matching_reference = None
if 'last_optimization_time' not in st.session_state:
    st.session_state.last_optimization_time = None

# Utility Functions
def enhanced_normalize(text):
    if not isinstance(text, str):
        return ""
    try:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return ""

def score_question_quality(question):
    try:
        if not isinstance(question, str) or len(question.strip()) < 5:
            return 0
        length_score = min(len(question) / 100, 1.0)
        word_count = len(question.split())
        word_score = min(word_count / 20, 1.0)
        has_numbers = 1.0 if any(c.isdigit() for c in question) else 0.5
        return (length_score + word_score + has_numbers) / 3
    except Exception as e:
        logger.error(f"Error scoring question quality: {e}")
        return 0

def get_best_question_for_uid(variants):
    try:
        if not variants:
            return None
        scored_variants = [(v, score_question_quality(v)) for v in variants if isinstance(v, str)]
        if not scored_variants:
            return None
        return max(scored_variants, key=lambda x: x[1])[0]
    except Exception as e:
        logger.error(f"Error selecting best question: {e}")
        return None

# Snowflake Connection
@st.cache_resource
def get_snowflake_engine():
    try:
        sf = st.secrets["snowflake"]
        logger.info(f"Attempting Snowflake connection: {sf['user']}")
        for attempt in range(3):
            try:
                engine = create_engine(
                    f"snowflake://{sf['user']}:{sf['password']}@{sf['account']}/{sf['database']}/{sf['schema']}?"
                    f"warehouse={sf['warehouse']}&role={sf['role']}"
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT CURRENT_VERSION()"))
                return engine
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Snowflake connection attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Snowflake engine creation failed: {e}")
        if "250001" in str(e):
            st.warning(
                "🔒 Snowflake connection failed: User account is locked. "
                "UID matching is disabled, but you can use SurveyMonkey features. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
            )
        raise

# SurveyMonkey Functions
@st.cache_data(ttl=300)
def get_surveys(token):
    try:
        url = "https://api.surveymonkey.com/v3/surveys"
        headers = {"Authorization": f"Bearer {token}"}
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json().get("data", [])
                logger.info(f"SurveyMonkey surveys response: {json.dumps(data[:2], indent=2)}")
                for survey in data:
                    if 'question_count' not in survey or survey['question_count'] is None:
                        details = get_survey_details(survey['id'], token)
                        questions = extract_questions_from_surveymonkey(details)
                        survey['question_count'] = len(questions)
                return data
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                logger.warning(f"SurveyMonkey API attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to fetch surveys: {e}")
        st.error(f"❌ Failed to fetch surveys: {str(e)}")
        return []

def get_survey_details(survey_id, token):
    try:
        url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
        headers = {"Authorization": f"Bearer {token}"}
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                logger.info(f"SurveyMonkey details for ID {survey_id}: {json.dumps(data.get('pages', [])[:1], indent=2)}")
                if not data.get('pages'):
                    logger.warning(f"No pages found for survey ID {survey_id}")
                return data
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                logger.warning(f"Survey details attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
        st.error(f"❌ Failed to fetch survey details: {str(e)}")
        return {}

def extract_questions_from_surveymonkey(survey_data):
    try:
        questions = []
        if not survey_data or 'pages' not in survey_data:
            logger.warning("No pages found in survey data")
            return questions
        for page in survey_data.get('pages', []):
            if 'questions' not in page:
                continue
            for question in page.get('questions', []):
                if 'headings' in question and question['headings']:
                    question_text = question['headings'][0].get('heading', '')
                    if question_text.strip():
                        choices = []
                        if 'answers' in question and 'choices' in question['answers']:
                            choices = [choice.get('text', '') for choice in question['answers']['choices'] if choice.get('text')]
                        questions.append({
                            'question_id': question.get('id', ''),
                            'question_text': question_text,
                            'survey_title': survey_data.get('title', ''),
                            'choices': choices
                        })
        logger.info(f"Extracted {len(questions)} questions from survey")
        return questions
    except Exception as e:
        logger.error(f"Error extracting SurveyMonkey questions: {e}")
        st.error(f"❌ Failed to extract questions: {str(e)}")
        return []

# Matching Functions
def load_sentence_transformer():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def prepare_matching_data():
    try:
        ref_questions = get_cached_reference_questions()
        if ref_questions.empty:
            return [], None, {}
        model = load_sentence_transformer()
        ref_texts = ref_questions['heading_0'].tolist()
        ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)
        uid_lookup = {i: uid for i, uid in enumerate(ref_questions['uid'])}
        return ref_texts, ref_embeddings, uid_lookup
    except Exception as e:
        logger.error(f"Error preparing matching data: {e}")
        st.error(f"❌ Failed to prepare matching data: {str(e)}")
        return [], None, {}

@st.cache_data(ttl=CACHE_DURATION)
def get_cached_reference_questions():
    try:
        engine = get_snowflake_engine()
        table_name = st.secrets.get("snowflake", {}).get("table", "YOUR_SNOWFLAKE_TABLE")
        if table_name == "YOUR_SNOWFLAKE_TABLE":
            raise ValueError("Snowflake table name not configured. Set 'table' in st.secrets['snowflake'].")
        query = f"SELECT HEADING_0, UID, TITLE FROM {table_name} WHERE HEADING_0 IS NOT NULL"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch reference questions: {e}")
        error_msg = f"❌ Failed to fetch reference data: {str(e)}"
        if "Object" in str(e) and "does not exist or not authorized" in str(e):
            error_msg += "\nPlease ensure the Snowflake table exists and your user has SELECT permissions. Check 'table' in st.secrets['snowflake']."
        st.error(error_msg)
        return pd.DataFrame()

def perform_semantic_matching(surveymonkey_questions, df_reference):
    if not surveymonkey_questions or df_reference.empty:
        st.warning("⚠️ No questions or reference data provided for matching")
        return []
    try:
        model = load_sentence_transformer()
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        ref_texts = df_reference['heading_0'].tolist()
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)
        similarities = util.cos_sim(sm_embeddings, ref_embeddings)
        matched_results = []
        for i, sm_question in enumerate(surveymonkey_questions):
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            result = sm_question.copy()
            if best_score >= SEMANTIC_THRESHOLD:
                result['matched_uid'] = df_reference.iloc[best_match_idx]['uid']
                result['matched_heading_0'] = df_reference.iloc[best_match_idx]['heading_0']
                result['match_score'] = best_score
                result['match_confidence'] = "High" if best_score >= 0.8 else "Medium"
            else:
                result['matched_uid'] = None
                result['matched_heading_0'] = None
                result['match_score'] = best_score
                result['match_confidence'] = "Low"
            matched_results.append(result)
        return matched_results
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        st.error(f"❌ Semantic matching failed: {str(e)}")
        return []

def ultra_fast_semantic_matching(surveymonkey_questions, use_optimized_reference=True):
    if not surveymonkey_questions:
        st.warning("⚠️ No SurveyMonkey questions provided for matching")
        return []
    try:
        if use_optimized_reference:
            optimized_ref = get_optimized_matching_reference()
            if optimized_ref.empty:
                st.warning("⚠️ Optimized reference not built. Falling back to fast matching.")
                return fast_semantic_matching(surveymonkey_questions, use_cached_data=True)
            logger.info(f"Using optimized reference with {len(optimized_ref)} unique questions")
            ref_texts = optimized_ref['best_question'].tolist()
        else:
            return fast_semantic_matching(surveymonkey_questions, use_cached_data=True)
        model = load_sentence_transformer()
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        logger.info(f"Encoding {len(sm_texts)} SurveyMonkey questions against {len(ref_texts)} optimized references")
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)
        similarities = util.cos_sim(sm_embeddings, ref_embeddings)
        matched_results = []
        for i, sm_question in enumerate(surveymonkey_questions):
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            result = sm_question.copy()
            if best_score >= SEMANTIC_THRESHOLD:
                matched_row = optimized_ref.iloc[best_match_idx]
                result['matched_uid'] = matched_row['winner_uid']
                result['matched_heading_0'] = matched_row['best_question']
                result['match_score'] = best_score
                result['match_confidence'] = "High" if best_score >= 0.8 else "Medium"
                result['conflict_resolved'] = matched_row['has_conflicts']
                result['uid_authority'] = matched_row['winner_count']
                result['conflict_severity'] = matched_row.get('conflict_severity', 0)
            else:
                result['matched_uid'] = None
                result['matched_heading_0'] = None
                result['match_score'] = best_score
                result['match_confidence'] = "Low"
                result['conflict_resolved'] = False
                result['uid_authority'] = 0
                result['conflict_severity'] = 0
            matched_results.append(result)
        logger.info(f"Ultra-fast semantic matching completed for {len(matched_results)} questions")
        return matched_results
    except Exception as e:
        logger.error(f"Ultra-fast semantic matching failed: {e}")
        st.error(f"❌ Ultra-fast matching failed: {str(e)}. Falling back to fast matching.")
        return fast_semantic_matching(surveymonkey_questions, use_cached_data=True)

def fast_semantic_matching(surveymonkey_questions, use_cached_data=True):
    if not surveymonkey_questions:
        st.warning("⚠️ No SurveyMonkey questions provided for matching")
        return []
    try:
        if use_cached_data:
            ref_questions, ref_embeddings, uid_lookup = prepare_matching_data()
            if ref_embeddings is None:
                st.warning("⚠️ Cached data not available, falling back to slow matching")
                return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())
        else:
            return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())
        model = load_sentence_transformer()
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        logger.info(f"Encoding {len(sm_texts)} SurveyMonkey questions")
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        logger.info("Calculating similarities using pre-computed embeddings")
        similarities = util.cos_sim(sm_embeddings, ref_embeddings)
        matched_results = []
        for i, sm_question in enumerate(surveymonkey_questions):
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            result = sm_question.copy()
            if best_score >= SEMANTIC_THRESHOLD:
                matched_uid = uid_lookup.get(best_match_idx)
                matched_question = ref_questions[best_match_idx]
                result['matched_uid'] = matched_uid
                result['matched_heading_0'] = matched_question
                result['match_score'] = best_score
                result['match_confidence'] = "High" if best_score >= 0.8 else "Medium"
            else:
                result['matched_uid'] = None
                result['matched_heading_0'] = None
                result['match_score'] = best_score
                result['match_confidence'] = "Low"
            matched_results.append(result)
        logger.info(f"Fast semantic matching completed for {len(matched_results)} questions")
        return matched_results
    except Exception as e:
        logger.error(f"Fast semantic matching failed: {e}")
        st.error(f"❌ Fast matching failed: {str(e)}. Falling back to standard matching.")
        return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())

def batch_process_matching(surveymonkey_questions, batch_size=100):
    if not surveymonkey_questions:
        st.warning("⚠️ No SurveyMonkey questions provided for batch processing")
        return []
    try:
        total_questions = len(surveymonkey_questions)
        if total_questions <= batch_size:
            return fast_semantic_matching(surveymonkey_questions)
        logger.info(f"Processing {total_questions} questions in batches of {batch_size}")
        all_results = []
        for i in range(0, total_questions, batch_size):
            batch = surveymonkey_questions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_questions-1)//batch_size + 1}")
            with st.spinner(f"Processing batch {i//batch_size + 1}/{(total_questions-1)//batch_size + 1}..."):
                batch_results = fast_semantic_matching(batch)
                all_results.extend(batch_results)
        logger.info(f"Batch processing completed: {len(all_results)} total results")
        return all_results
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        st.error(f"❌ Batch processing failed: {str(e)}. Falling back to regular processing.")
        return fast_semantic_matching(surveymonkey_questions)

@st.cache_data(ttl=CACHE_DURATION)
@monitor_performance
def get_optimized_matching_reference():
    try:
        if 'primary_matching_reference' in st.session_state and st.session_state.primary_matching_reference is not None:
            return st.session_state.primary_matching_reference
        else:
            st.warning("⚠️ Optimized 1:1 question bank not built yet")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to get optimized matching reference: {e}")
        st.error(f"❌ Failed to access optimized reference: {str(e)}")
        return pd.DataFrame()

@monitor_performance
def build_optimized_1to1_question_bank(df_reference):
    if df_reference.empty:
        st.warning("⚠️ No Snowflake reference data provided for optimization")
        return pd.DataFrame()
    try:
        logger.info(f"Building optimized 1:1 question bank from {len(df_reference):,} Snowflake records")
        df_reference['normalized_question'] = df_reference['heading_0'].apply(enhanced_normalize)
        question_analysis = []
        grouped = df_reference.groupby('normalized_question')
        for norm_question, group in grouped:
            if not norm_question or len(norm_question.strip()) < 3:
                continue
            uid_counts = group['uid'].value_counts()
            all_variants = group['heading_0'].unique()
            best_question = get_best_question_for_uid(all_variants)
            if not best_question:
                continue
            uid_conflicts = [{'uid': uid, 'count': count, 'percentage': (count / len(group)) * 100} 
                             for uid, count in uid_counts.items()]
            uid_conflicts.sort(key=lambda x: x['count'], reverse=True)
            winner_uid = uid_conflicts[0]['uid']
            winner_count = uid_conflicts[0]['count']
            conflicts = [conflict for conflict in uid_conflicts[1:] 
                         if conflict['count'] >= UID_GOVERNANCE['conflict_resolution_threshold']]
            question_analysis.append({
                'normalized_question': norm_question,
                'best_question': best_question,
                'winner_uid': winner_uid,
                'winner_count': winner_count,
                'total_occurrences': len(group),
                'unique_uids_count': len(uid_counts),
                'has_conflicts': len(conflicts) > 0,
                'conflict_count': len(conflicts),
                'conflicts': conflicts,
                'all_uid_counts': dict(uid_counts),
                'variants_count': len(all_variants),
                'quality_score': score_question_quality(best_question),
                'conflict_severity': sum(c['count'] for c in conflicts) if conflicts else 0
            })
        optimized_df = pd.DataFrame(question_analysis)
        if not optimized_df.empty:
            st.session_state.primary_matching_reference = optimized_df
            st.session_state.last_optimization_time = datetime.now()
            logger.info(f"Built optimized 1:1 question bank: {len(optimized_df):,} unique questions")
            st.success(f"✅ Built optimized 1:1 question bank with {len(optimized_df):,} unique questions")
        else:
            logger.warning("No valid questions found for optimization")
            st.warning("⚠️ No valid questions found for optimization")
        return optimized_df
    except Exception as e:
        logger.error(f"Failed to build optimized question bank: {e}")
        st.error(f"❌ Failed to build optimized question bank: {str(e)}")
        return pd.DataFrame()

def run_uid_match(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP, batch_size=BATCH_SIZE):
    try:
        if df_target.empty:
            st.warning("⚠️ No target questions provided for matching")
            return df_target
        sm_questions = df_target.to_dict('records')
        opt_ref = get_optimized_matching_reference()
        if not opt_ref.empty:
            logger.info("Using ultra-fast matching with 1:1 optimization")
            matched_results = ultra_fast_semantic_matching(sm_questions, use_optimized_reference=True)
        else:
            logger.info("Using fast semantic matching")
            matched_results = fast_semantic_matching(sm_questions, use_cached_data=True)
        if matched_results:
            final_df = pd.DataFrame(matched_results)
            return final_df
        else:
            st.warning("⚠️ No matching results generated")
            return df_target
    except Exception as e:
        logger.error(f"UID matching failed: {e}")
        st.error(f"❌ UID matching failed: {str(e)}")
        return df_target

# Performance Stats
def get_performance_stats():
    try:
        unique_questions = len(st.session_state.get('primary_matching_reference', pd.DataFrame()))
        last_optimization = st.session_state.get('last_optimization_time', None)
        return {
            'unique_questions_loaded': unique_questions,
            'last_optimization_time': last_optimization
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {'unique_questions_loaded': 0, 'last_optimization_time': None}

# Survey Categorization
def categorize_survey_from_surveymonkey(survey_title):
    try:
        if not isinstance(survey_title, str):
            return 'General'
        title_lower = survey_title.lower()
        if 'customer' in title_lower:
            return 'Customer Satisfaction'
        elif 'employee' in title_lower:
            return 'Employee Engagement'
        elif 'product' in title_lower:
            return 'Product Feedback'
        else:
            return 'General'
    except Exception as e:
        logger.error(f"Error categorizing survey: {e}")
        return 'General'

# Page Definitions
def home_dashboard():
    st.title("🏠 UID Matcher Pro Dashboard")
    st.markdown("Welcome to UID Matcher Pro, the ultimate tool for matching SurveyMonkey questions to Snowflake UIDs!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Surveys Processed", "0")  # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Questions Matched", "0")  # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Conflicts Resolved", "0")  # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)

def view_surveys():
    st.title("📋 View SurveyMonkey Surveys")
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", "")
        if not token:
            token = st.text_input("Enter SurveyMonkey API Token", type="password")
        if token:
            surveys = get_surveys(token)
            if not surveys:
                st.warning("⚠️ No surveys found or API request failed")
                return
            
            # Create dropdown options: survey_id - survey_title
            survey_options = [
                f"{survey.get('id', 'N/A')} - {survey.get('title', 'Untitled Survey')}"
                for survey in surveys
            ]
            selected_survey = st.selectbox("Select Survey", survey_options)
            
            if selected_survey:
                # Extract survey_id from selection
                selected_id = selected_survey.split(" - ")[0]
                survey = next((s for s in surveys if s.get('id') == selected_id), None)
                if survey:
                    # Display survey metadata
                    st.write(f"**Survey ID:** {survey.get('id', 'N/A')}")
                    st.write(f"**Title:** {survey.get('title', 'Untitled Survey')}")
                    question_count = survey.get('question_count', 0)
                    st.write(f"**Questions:** {question_count if question_count > 0 else 'No questions found'}")
                    if question_count == 0:
                        st.warning("⚠️ This survey has no questions")
                    
                    # Load and display questions in a table
                    if st.button("View Survey Details"):
                        details = get_survey_details(survey['id'], token)
                        questions = extract_questions_from_surveymonkey(details)
                        if not questions:
                            st.warning("⚠️ No questions extracted from this survey")
                        else:
                            df_questions = pd.DataFrame(questions)
                            df_questions['choices'] = df_questions['choices'].apply(lambda x: ', '.join(x) if x else 'N/A')
                            st.markdown("### Survey Questions")
                            st.dataframe(df_questions[['question_id', 'question_text', 'choices', 'survey_title']])
                            st.session_state.questions = questions
                            st.session_state.page = "configure_survey"
                            st.rerun()
        else:
            st.warning("⚠️ No SurveyMonkey token provided")
    except Exception as e:
        logger.error(f"Failed to fetch surveys: {e}")
        st.error(f"❌ Failed to fetch surveys: {str(e)}")

def create_survey():
    st.title("✨ Create New Survey")
    st.info("Survey creation is not implemented in this version.")

def configure_survey():
    st.title("⚙️ Configure Survey")
    sf_status = True
    try:
        engine = get_snowflake_engine()
        st.session_state.snowflake_engine = engine
    except Exception:
        sf_status = False
        st.warning("⚠️ Snowflake connection not established")
    
    if 'questions' in st.session_state and st.session_state.questions:
        df_target = pd.DataFrame(st.session_state.questions)
        st.markdown("### 📋 Survey Questions")
        st.dataframe(df_target[['survey_title', 'question_text', 'choices']] if 'choices' in df_target.columns else df_target[['survey_title', 'question_text']])
        
        if sf_status:
            st.markdown("### 🔄 UID Assignment Process")
            perf_stats = get_performance_stats()
            opt_ref = get_optimized_matching_reference()
            if not opt_ref.empty:
                st.success(f"✅ Optimized 1:1 reference ready: {len(opt_ref):,} conflict-resolved questions")
            elif perf_stats['unique_questions_loaded'] > 0:
                st.success("✅ Standard optimization ready! Consider building 1:1 optimization for best performance.")
            else:
                st.warning("⚠️ Question Bank not optimized! Matching will be slower.")
                if st.button("🏗️ Build Question Bank for Better Performance"):
                    st.session_state.page = "build_question_bank"
                    st.rerun()
            
            st.markdown("### ⚡ Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**❌ Standard Matching**")
                st.markdown("• Loads 1M+ records")
                st.markdown("• 2-5 minutes per matching")
                st.markdown("• Memory intensive")
                st.markdown("• App crashes possible")
            with col2:
                st.markdown("**✅ Fast Matching**")
                st.markdown("• Uses cached embeddings")
                st.markdown("• 10-30 seconds per matching")
                st.markdown("• Memory efficient")
                st.markdown("• Stable performance")
            with col3:
                st.markdown("**🎯 Ultra-Fast Matching**")
                st.markdown("• Uses 1:1 optimized bank")
                st.markdown("• 2-5 seconds per matching")
                st.markdown("• Highly optimized")
                st.markdown("• Conflict-free results")
            
            matching_approach = st.radio(
                "Select matching approach:",
                [
                    "🎯 Ultra-Fast Matching (1:1 Optimized, Recommended)",
                    "✅ Fast Matching (Standard)",
                    "❌ Standard Matching (Slowest)"
                ],
                help="Ultra-fast uses conflict-resolved 1:1 mapping for maximum speed and accuracy"
            )
            
            use_batching = st.checkbox(
                "Use batch processing",
                value=len(df_target) > BATCH_SIZE,
                help="Recommended for large datasets (>1000 questions)"
            )
            
            if st.button("🚀 Run UID Matching", type="primary"):
                try:
                    df_reference = get_cached_reference_questions()
                    if matching_approach == "🎯 Ultra-Fast Matching (1:1 Optimized, Recommended)":
                        if opt_ref.empty:
                            st.error("❌ 1:1 optimization not built yet. Please build it first.")
                            if st.button("🎯 Build 1:1 Optimization Now"):
                                st.session_state.page = "optimized_question_bank"
                                st.rerun()
                        else:
                            with st.spinner("🎯 Running ULTRA-FAST semantic matching with 1:1 optimization..."):
                                matched_results = batch_process_matching(df_target.to_dict('records'), batch_size=BATCH_SIZE) if use_batching else ultra_fast_semantic_matching(df_target.to_dict('records'), use_optimized_reference=True)
                    
                    elif matching_approach == "✅ Fast Matching (Standard)":
                        with st.spinner("✅ Running FAST semantic matching with pre-computed embeddings..."):
                            matched_results = batch_process_matching(df_target.to_dict('records'), batch_size=BATCH_SIZE) if use_batching else fast_semantic_matching(df_target.to_dict('records'), use_cached_data=True)
                    
                    else:
                        with st.spinner("❌ Running standard semantic matching (slower)..."):
                            matched_results = perform_semantic_matching(df_target.to_dict('records'), df_reference)
                    
                    if matched_results:
                        matched_df = pd.DataFrame(matched_results)
                        st.session_state.df_final = matched_df
                        st.success(f"✅ UID matching completed!")
                        high_conf = len(matched_df[matched_df['match_confidence'] == 'High'])
                        medium_conf = len(matched_df[matched_df['match_confidence'] == 'Medium'])
                        low_conf = len(matched_df[matched_df['match_confidence'] == 'Low'])
                        conflicts_resolved = len(matched_df[matched_df.get('conflict_resolved', False) == True]) if 'conflict_resolved' in matched_df else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 High Confidence", high_conf)
                        with col2:
                            st.metric("⚠️ Medium Confidence", medium_conf)
                        with col3:
                            st.metric("❌ Low/No Match", low_conf)
                        with col4:
                            st.metric("🔥 Conflicts Resolved", conflicts_resolved)
                        
                        st.markdown("### 📋 Sample Matching Results")
                        sample_matched = matched_df[matched_df['matched_uid'].notna()].head(5)
                        for idx, row in sample_matched.iterrows():
                            conflict_badge = " 🔥 CONFLICT RESOLVED" if row.get('conflict_resolved', False) else ""
                            authority_info = f" (Authority: {row.get('uid_authority', 0)} records)" if row.get('uid_authority', 0) > 0 else ""
                            with st.expander(f"Match {idx+1}: UID {row['matched_uid']} (Confidence: {row['match_confidence']}){conflict_badge}"):
                                st.write(f"**SurveyMonkey Question:** {row['question_text']}")
                                st.write(f"**Matched Snowflake Question:** {row['matched_heading_0']}")
                                st.write(f"**Match Score:** {row['match_score']:.3f}")
                                if 'choices' in row and row['choices']:
                                    st.write(f"**Choices:** {', '.join(row['choices'])}")
                                if row.get('conflict_resolved', False):
                                    st.write(f"**UID Authority:** {row['uid_authority']} records{authority_info}")
                                    st.info("🔥 This question had multiple competing UIDs. Assigned to highest-count UID.")
                    else:
                        st.error("❌ No matching results generated")
                except Exception as e:
                    logger.error(f"UID matching failed: {e}")
                    st.error(f"❌ UID matching failed: {str(e)}")
        else:
            st.warning("❌ Snowflake connection required for UID assignment")
            st.info("Configure surveys is available, but UID matching requires Snowflake connection")
    else:
        st.warning("⚠️ No survey questions loaded. Please view and select a survey first.")

def build_question_bank():
    st.title("🏗️ Build Question Bank")
    try:
        df_reference = get_cached_reference_questions()
        if not df_reference.empty:
            st.success(f"✅ Loaded {len(df_reference):,} reference questions")
            if st.button("🔄 Refresh Question Bank"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("⚠️ No reference questions loaded. Check Snowflake table configuration.")
    except Exception as e:
        st.error(f"❌ Failed to build question bank: {str(e)}")

def optimized_question_bank():
    st.title("🎯 Build Optimized 1:1 Question Bank")
    try:
        df_reference = get_cached_reference_questions()
        if not df_reference.empty:
            st.success(f"✅ Loaded {len(df_reference):,} reference questions")
            if st.button("🚀 Build Optimized 1:1 Question Bank"):
                with st.spinner("Building optimized question bank..."):
                    optimized_df = build_optimized_1to1_question_bank(df_reference)
                    if not optimized_df.empty:
                        st.success(f"✅ Optimization complete!")
        else:
            st.warning("⚠️ No reference questions loaded. Check Snowflake table configuration.")
    except Exception as e:
        st.error(f"❌ Failed to build optimized question bank: {str(e)}")

def matching_dashboard():
    st.title("📊 Matching Dashboard")
    if 'df_final' in st.session_state and st.session_state.df_final is not None:
        df = st.session_state.df_final
        st.dataframe(df)
    else:
        st.warning("⚠️ No matching results available. Run UID matching first.")

def settings():
    st.title("⚙️ Settings")
    st.info("Settings page not implemented in this version.")

# Page Routing
if st.session_state.page == "Home Dashboard":
    home_dashboard()
elif st.session_state.page == "View Surveys":
    view_surveys()
elif st.session_state.page == "Create Survey":
    create_survey()
elif st.session_state.page == "Configure Survey":
    configure_survey()
elif st.session_state.page == "Build Question Bank":
    build_question_bank()
elif st.session_state.page == "Optimized 1:1 Question Bank":
    optimized_question_bank()
elif st.session_state.page == "Matching Dashboard":
    matching_dashboard()
elif st.session_state.page == "Settings":
    settings()
