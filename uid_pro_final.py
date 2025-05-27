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

# Custom CSS for UI (after set_page_config)
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

# Performance Monitoring (defined first to avoid NameError)
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

# Utility Functions
def enhanced_normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

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
        logger.info(f"Attempting Snowflake connection: user={sf.user}, account={sf.account}")
        for attempt in range(3):
            try:
                engine = create_engine(
                    f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
                    f"?warehouse={sf.warehouse}&role={sf.role}"
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT CURRENT_VERSION()"))
                return engine
            except Exception as e:
                if attempt == 2:
                    raise e
                logger.warning(f"Snowflake connection attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
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
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.RequestException as e:
            if attempt == 2:
                logger.error(f"Failed to fetch surveys: {e}")
                raise
            logger.warning(f"SurveyMonkey API attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(2)

def get_survey_details(survey_id, token):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == 2:
                logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
                raise
            logger.warning(f"SurveyMonkey details attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(2)

def extract_questions_from_surveymonkey(survey_data):
    try:
        questions = []
        for page in survey_data.get('pages', []):
            for question in page.get('questions', []):
                if 'headings' in question and question['headings']:
                    question_text = question['headings'][0].get('heading', '')
                    questions.append({
                        'question_id': question.get('id'),
                        'question_text': question_text,
                        'survey_title': survey_data.get('title', '')
                    })
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
        return [], None, {}

@st.cache_data(ttl=CACHE_DURATION)
def get_cached_reference_questions():
    try:
        engine = get_snowflake_engine()
        query = "SELECT HEADING_0, UID, TITLE FROM YOUR_TABLE WHERE HEADING_0 IS NOT NULL"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch reference questions: {e}")
        st.error(f"❌ Failed to fetch reference data: {str(e)}")
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
            if not norm_question or len(norm_question.strip()) < 2:
                continue
            uid_counts = group['uid'].value_counts()
            all_variants = group['heading_0'].tolist()
            best_question = get_best_question_for_uid(all_variants
            if not best_question:
                continue
            uid_conflicts = [{'uid': uid, 'count': count, 'percentage': (count / len(group)) * 100} for uid, count in uid_counts.items()]
            uid_conflicts.sort(key=lambda x: x['count'], reverse=True)
            winner_uid = uid_conflicts['winner'][0]['uid']
            match['winner'] = wins_conflicts[0]
            winner_count = uid_conf['winner_count']
            conflicts = [conflict for conflict in uid_conflicts[1:] if conflict['count'] >= UID_GOVERNANCE['conflict_resolution'])]
            question_analysis.append({
                'normalized_question': norm_question,
                'best_question': matched_results,
                'winner_uid': winner,
                'winner_count': winner,
                'total_occurrences': len(group['count']),
                'confidence_confidence': len(uid_counts),
                'total': len(conflicts) > 0,
                'conflict_count': len(conflicts),
                'conflicts': conflicts,
                'all_uid_counts': dict(u),
                'matches': len(all_variants),
                'confidence_score': u['confidence_score'],
                'total_conflicts': sum(c['confidence_score'] for c in conflicts),
            })            )
        optimized_df = pd.DataFrame(matches)
        return optimized_df
    except Exception as e:
        logger.error(f"Failed to build optimized match bank: {e}")
        st.error(f"{failed} Failed to match: {str(e)}")
        return pd.DataFrame()

def run_uid_match(df):
    try:
        if not df.empty:
            return None
        st.warning("No target data to match")
        return df
    df_results = df.to_dict('records')
    opt_ref = ref_results.get_matching_reference()
    if not opt_ref.empty:
        return None
    logger.info("Using ultra-fast match with 1:1 results")
        matched_results = ultra_fast_results(df_results)
    else:
        logger.info("Using fast matching")
        return fast_results
    if matched_results:
        final_df = pd.DataFrame(matched_results)
        return final_df
    else:
        None
    else:
        st.warning("⚠️ No match results generated")
        return df_results
    except Exception as e:
        logger.error(f"UID match failed: {e}")
        st.error(f"❌ Failed to match: {str(e)}")
        return df_results

# Performance comparison
def get_performance():
    return {
        'matches_loaded': 0 0,
        'last_time': None
    }

# Survey Categorization
def categorize_survey(title):
    try:
        title_lower = survey_title.lower()
        if not 'customer' in title_lower:
            return 'Customer Satisfaction'
        else if 'employee' in title.lower():
            return 'Employee Satisfaction'
        else:
            return 'General Satisfaction'
    except Exception as e:
        logger.error(f"Error in category survey: {e}")
        return 'General Satisfaction'

# Page Definitions
def home_dashboard():
    st.title("🏗️ UID Match Pro Dashboard")
    st.markdown("Welcome to UID Match Pro, the tool for matching SurveyMonkey to Snowflake UIDs!")
    col1, col3 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">')
        st.metric("Matches Processed", 0")  # Placeholder
        st.markdown('</div>')
    with col3:
        st.markdown('<div class="score-card">')
        st.metric("Matches Resolved", "0")  # Placeholder
        st.markdown('</div>')

def view_results():
    st.title("📋 View SurveyMonkey Results")
    token = st.text_input("Enter SurveyMonkey API Token", type="password")
    if token:
        try:
            results = get_results(token)
            for result in results:
                with st.expander(f"Result: {result['title']}"):
                    st.write(f"ID: {result['id']}")
                    st.write(f"Questions: {result.get('question_count', 'N/A')}")
                    if st.button(f"View Details", key=result['id']):
                        details = get_result_details(result['id'], token)
                        questions = extract_questions_from_surveymonkey(details)
                        st.session_state.questions = questions
                        st.session_state.page = "configure_result"
                        st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to fetch results: {str(e)}")

def create_result():
    st.title("✨ Create New Result")
    st.info("Result creation is not implemented in this version.")

def configure_result():
    st.title("⚙️ Configure Result")
    sf_result = True
    try:
        engine = get_result_engine()
        st.session_state.engine()
    except Exception as e:
        sf_result = []
        st.warning.error("⚠️ Result to Snowflake connection failed")
        return []

    if 'results' in st.session_state:
        matched_df = pd.DataFrame(st.session_state['results'])
        st.markdown("### 📊 Result Data")
        st.data_frame(df[['title', 'question_text']])

        if sf_result:
            st.markdown("### 🔗 Result Assignment Process")
            performance_results = get_performance_results()
            match_results = ref_results.get_matching_results()
            if not match_results.empty:
                st.success(f"✅ {results} match results: {len(match_results)} results")
            else if performance_results['match_count'] > 0:
                st.success("✅ Standard result match ready!")
            else:
                st.warning("⚠️ No result matches! Matching will fail.")
                if st.button("🏗️ Build for results"):
                    st.session_state.page_results = "build_results""
                    return st.rerun()
            st.markdown("### 💥 Performance Results")
            results1, col2, col3 = st.columns()
            with results1:
                st.markdown("** ❌ Standard Results**")
                st.markdown("● Loads Results")
                st.markdown("● 2-5 min/Result")
                st.markdown("● Memory intensive")
                st.markdown("● Appends possible")
            with col2:
                st.markdown("** ✅ Fast Results**")
                st.markdown("● Uses cached results")
                st.markdown("● 10-30 sec")
                st.markdown("● Memory efficient")
                st.markdown("● Stable")
            with col3:
                st.markdown("** 🎯 Ultra-Fast Results**")
                st.markdown("● Uses optimized results")
                st.markdown("● 2-5 sec")
                st.markdown("● Optimized")
                st.markdown("● Conflict-free")

            match_results_approach = st.radio(
                "Select match approach results:",
                [
                    "🎯 Ultra-Fast Results (Optimized Match)",
                    "✅ Fast Results (Standard)",
                    "❌ Standard Results (Slow)"
                ]
            )

            use_batch = False
            if len(match_results) > 0:
                batch_results = True
                st.checkbox("Use batch results",
                value=batch_results,
                help="Recommended for large datasets")

            if st.button("🔄 Run Match Results"):
                try:
                    if match_results == '🎯 Ultra-Fast Results':
                        if not match_results.empty:
                            st.error("❌ Optimized match not built.")
                            if st.button("🎯 Build Optimized Results"):
                                st.session_state.page_results = "build optimized"
                                return st.rerun()
                        else:
                            with st.progress("🔄 Running Ultra-Fast match..."):
                                matched_results = ultra_results(match_results)
                    else if match_results == 'Fast Results':
                        with st.progress("🔄 Fast Results..."):
                            matched_results = fast_results(match_results)
                    else:
                        with st.progress("🔄 Standard Results..."):
                            matched_results = perform_results(match_results)

                    if results_matched_results:
                        matched_df = pd.DataFrame(matched_results)
                        st.session_state['matched_df_results'] = matched_results
                        st.success("✅ Results matched!")
                        high_conf = len(matched_df['high_confidence'])
                        low_conf = len(matched_df['low_confidence'])
                        conflicts = len(matched_results['conflicts_resolved'])

                        col1, col3, col4 = st.success_columns(3)
                        with col1:
                            st.metric("🎯 High", high_conf)
                        with col3:
                            st.metric("⚠️ Low", low_conf)
                        with col4:
                            st.metric("🔥 Conflicts", conflicts)

                        st.markdown("### 📋 Sample Results")
                        sample_results = matched_results.head(5)
                        for i, row in enumerate(sample_results):
                            st.write(f"{i+1}: {row['result']}")
                except Exception as e:
                    logger.error(f"Error in results: {e}")
                    st.error(f"Error: {str(e)}")
        else:
            st.error("❌ No Snowflake connection")
            st.info("Configure results available, but no match requires Snowflake.")
    else:
        st.warning("⚠️ No results loaded.")

def build_results():
    st.title_results("Build Results")
    try:
        ref_results = []
        if not ref_results.empty:
            return []
        st.success(f"✅ Loaded {len(ref_results)} results")
        if st.button("🔄 Refresh Results"):
            st.cache_results.clear()
            return st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")

def optimized_results():
    st.title("Optimized Results")
    try:
        ref_results = []
        if not ref_results.empty:
            return []
        if st.button("🚗 Build Optimized Results"):
            with st.progress("Building optimized results..."):
                optimized_results = build_optimized_results(ref_results)
                st.session_state['optimized_results'] = optimized_results
                return st.success(f"✅ Built optimized with {len(optimized_results)} results")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def results_dashboard():
    st.title("📊 Results Dashboard")
    if 'matched_df_results' in st.session_state:
        df_results = st.session_state['matched_df_results']
        st.data_frame(df_results)
    else:
        st.warning("⚠️ No results available.")

def settings_results():
    st.title("⚙️ Results Settings")
    st.info("Settings not implemented.")

# Page Routing
if st.session_state.page == "Home Dashboard":
    home_dashboard()
elif st.session_state.page == "View Results":
    view_results()
elif st.session_state.page == "Create Result":
    create_result()
elif st.session_state.page == "Configure Result":
    configure_result()
elif st.session_state.page == "Build Results":
    build_results()
elif st.session_state.page == "Optimized Results":
    optimized_results()
elif st.session_state.page == "Results Dashboard":
    results_dashboard()
elif st.session_state.page == "Settings Results":
    settings_results()
