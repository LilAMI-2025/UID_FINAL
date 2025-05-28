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
            page_title="UID Matcher Pro Enhanced",
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
UID_GOVERNANCE = {
    'conflict_resolution_threshold': 10,
    'max_variations_per_uid': 50,
    'min_authority_difference': 0.2,
    'high_conflict_threshold': 100
}
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

# Custom CSS for UI
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
        margin-bottom: 10px;
    }
    
    .conflict-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
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

# Initialize session state
if 'snowflake_engine' not in st.session_state:
    st.session_state.snowflake_engine = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'primary_matching_reference' not in st.session_state:
    st.session_state.primary_matching_reference = None
if 'last_optimization_time' not in st.session_state:
    st.session_state.last_optimization_time = None
if 'optimization_attempted' not in st.session_state:
    st.session_state.optimization_attempted = False
if 'uid_conflicts_summary' not in st.session_state:
    st.session_state.uid_conflicts_summary = None
if 'optimized_question_bank' not in st.session_state:
    st.session_state.optimized_question_bank = None

# Utility Functions
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
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

def score_question_quality(question):
    """Enhanced question quality scoring"""
    try:
        if not isinstance(question, str) or len(question.strip()) < 5:
            return 0
        
        score = 0
        text = question.lower().strip()
        
        # Length scoring (optimal range 10-100 characters)
        length = len(question)
        if 10 <= length <= 100:
            score += 25
        elif 5 <= length <= 150:
            score += 15
        elif length < 5:
            score -= 25
        
        # Question format scoring
        if text.endswith('?'):
            score += 20
        
        # English question words at the beginning
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
        first_three_words = text.split()[:3]
        if any(word in first_three_words for word in question_words):
            score += 20
        
        # Proper capitalization
        if question and question[0].isupper():
            score += 10
        
        # Avoid common artifacts and low-quality indicators
        bad_patterns = [
            'click here', 'please select', '...', 'n/a', 'other', 
            'select one', 'choose all', 'privacy policy', 'thank you',
            'contact us', 'submit', 'continue', '<div', '<span', 'html'
        ]
        if any(pattern in text for pattern in bad_patterns):
            score -= 25
        
        # Avoid HTML content
        if '<' in question and '>' in question:
            score -= 30
        
        # Word count scoring
        word_count = len(question.split())
        if 3 <= word_count <= 15:
            score += 15
        elif word_count < 3:
            score -= 15
        
        return max(0, score)  # Ensure non-negative score
        
    except Exception as e:
        logger.error(f"Error scoring question quality: {e}")
        return 0

def get_best_question_for_uid(variants):
    """Select the best quality question from a list of variants"""
    try:
        if not variants:
            return None
        
        # Filter out invalid variants
        valid_variants = [v for v in variants if isinstance(v, str) and len(v.strip()) > 3]
        if not valid_variants:
            return None
        
        # Score all variants
        scored_variants = [(v, score_question_quality(v)) for v in valid_variants]
        
        # Sort by score (descending) and then by length (shorter is better for same score)
        scored_variants.sort(key=lambda x: (-x[1], len(x[0])))
        
        return scored_variants[0][0]
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
                logger.info("Snowflake connection successful")
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

# Enhanced Snowflake Query Functions
@st.cache_data(ttl=600)
def get_all_reference_questions_from_snowflake():
    """Fetch ALL reference questions from Snowflake with enhanced query."""
    all_data = []
    limit = 10000
    offset = 0
    
    # Enhanced query to get more metadata
    query = """
    SELECT 
        HEADING_0, 
        UID, 
        TITLE,
        COUNT(*) as OCCURRENCE_COUNT,
        MIN(CREATED_DATE) as FIRST_SEEN,
        MAX(CREATED_DATE) as LAST_SEEN
    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
    WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
    AND TRIM(HEADING_0) != ''
    GROUP BY HEADING_0, UID, TITLE
    ORDER BY UID, OCCURRENCE_COUNT DESC
    LIMIT :limit OFFSET :offset
    """
    
    while True:
        try:
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
            
            if result.empty:
                logger.info(f"No more data at offset {offset}")
                break
                
            all_data.append(result)
            offset += limit
            
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            st.error(f"❌ Failed to fetch reference data: {str(e)}")
            return pd.DataFrame()
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Ensure proper column naming
        expected_columns = ['HEADING_0', 'UID', 'TITLE', 'OCCURRENCE_COUNT', 'FIRST_SEEN', 'LAST_SEEN']
        if len(final_df.columns) >= 3:
            # Map columns to expected format
            final_df.columns = expected_columns[:len(final_df.columns)]
        
        logger.info(f"Total reference questions fetched from Snowflake: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched from Snowflake")
        st.warning("⚠️ No reference questions found in Snowflake")
        return pd.DataFrame()

# Enhanced Question Bank Builder with 1:1 Optimization
@monitor_performance
def build_optimized_1to1_question_bank(df_reference):
    """Build optimized 1:1 question bank with conflict resolution"""
    if df_reference.empty:
        logger.warning("No Snowflake reference data provided for optimization")
        st.warning("⚠️ No Snowflake reference data provided for optimization")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        logger.info(f"Building optimized 1:1 question bank from {len(df_reference):,} Snowflake records")
        
        # Normalize questions for grouping
        df_reference['normalized_question'] = df_reference['HEADING_0'].apply(
            lambda x: enhanced_normalize(x, ENHANCED_SYNONYM_MAP)
        )
        
        # Group by normalized question to find conflicts
        question_analysis = []
        conflict_summary = []
        
        grouped = df_reference.groupby('normalized_question')
        
        for norm_question, group in grouped:
            if not norm_question or len(norm_question.strip()) < 3:
                continue
            
            # Count UIDs for this normalized question
            uid_counts = group.groupby('UID')['OCCURRENCE_COUNT'].sum().sort_values(ascending=False)
            
            if len(uid_counts) == 0:
                continue
            
            # Get all unique variants of the question text
            all_variants = group['HEADING_0'].unique()
            best_question = get_best_question_for_uid(all_variants)
            
            if not best_question:
                continue
            
            # Analyze conflicts
            total_occurrences = uid_counts.sum()
            winner_uid = uid_counts.index[0]
            winner_count = uid_counts.iloc[0]
            winner_percentage = (winner_count / total_occurrences) * 100
            
            # Identify significant conflicts
            conflicts = []
            for i in range(1, len(uid_counts)):
                competitor_uid = uid_counts.index[i]
                competitor_count = uid_counts.iloc[i]
                competitor_percentage = (competitor_count / total_occurrences) * 100
                
                # Only consider as conflict if count is above threshold
                if competitor_count >= UID_GOVERNANCE['conflict_resolution_threshold']:
                    conflicts.append({
                        'uid': competitor_uid,
                        'count': competitor_count,
                        'percentage': competitor_percentage,
                        'authority_difference': winner_percentage - competitor_percentage
                    })
            
            # Determine conflict severity
            has_high_conflicts = any(c['count'] >= UID_GOVERNANCE['high_conflict_threshold'] for c in conflicts)
            conflict_severity = sum(c['count'] for c in conflicts)
            
            # Add to question analysis
            question_analysis.append({
                'normalized_question': norm_question,
                'best_question': best_question,
                'uid': winner_uid,
                'winner_count': winner_count,
                'winner_percentage': winner_percentage,
                'total_occurrences': total_occurrences,
                'unique_uids_count': len(uid_counts),
                'has_conflicts': len(conflicts) > 0,
                'conflict_count': len(conflicts),
                'conflicts': conflicts,
                'all_uid_counts': dict(uid_counts),
                'variants_count': len(all_variants),
                'quality_score': score_question_quality(best_question),
                'conflict_severity': conflict_severity,
                'has_high_conflicts': has_high_conflicts,
                'authority_margin': winner_percentage - (conflicts[0]['percentage'] if conflicts else 0)
            })
            
            # Add to conflict summary if there are conflicts
            if len(conflicts) > 0:
                conflict_summary.append({
                    'question': best_question[:100] + "..." if len(best_question) > 100 else best_question,
                    'winner_uid': winner_uid,
                    'winner_count': winner_count,
                    'winner_percentage': winner_percentage,
                    'competing_uids': len(conflicts),
                    'top_competitor_uid': conflicts[0]['uid'] if conflicts else None,
                    'top_competitor_count': conflicts[0]['count'] if conflicts else 0,
                    'top_competitor_percentage': conflicts[0]['percentage'] if conflicts else 0,
                    'authority_difference': conflicts[0]['authority_difference'] if conflicts else 100,
                    'conflict_severity': conflict_severity,
                    'is_high_conflict': has_high_conflicts,
                    'total_variants': len(all_variants)
                })
        
        # Create DataFrames
        optimized_df = pd.DataFrame(question_analysis)
        conflicts_df = pd.DataFrame(conflict_summary)
        
        if not optimized_df.empty:
            # Sort by quality score and winner authority
            optimized_df = optimized_df.sort_values(['quality_score', 'winner_count'], ascending=[False, False])
            
            # Store in session state
            st.session_state.primary_matching_reference = optimized_df
            st.session_state.uid_conflicts_summary = conflicts_df
            st.session_state.optimized_question_bank = optimized_df[['uid', 'best_question', 'winner_count', 'quality_score']].rename(columns={
                'uid': 'UID',
                'best_question': 'Question',
                'winner_count': 'Authority_Count',
                'quality_score': 'Quality_Score'
            })
            st.session_state.last_optimization_time = datetime.now()
            
            logger.info(f"Built optimized 1:1 question bank: {len(optimized_df):,} unique questions, {len(conflicts_df):,} conflicts resolved")
            
        return optimized_df, conflicts_df
        
    except Exception as e:
        logger.error(f"Failed to build optimized question bank: {e}")
        st.error(f"❌ Failed to build optimized question bank: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# SurveyMonkey Functions with question_id
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
                logger.info(f"SurveyMonkey surveys fetched: {len(data)} surveys")
                
                # Enhance survey data with question counts
                for survey in data:
                    if 'question_count' not in survey or survey['question_count'] is None:
                        try:
                            details = get_survey_details(survey['id'], token)
                            questions = extract_questions_from_surveymonkey(details)
                            survey['question_count'] = len(questions)
                        except:
                            survey['question_count'] = 0
                
                return data
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                logger.warning(f"SurveyMonkey API attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to fetch surveys: {e}")
        if "401" in str(e):
            st.error("❌ Invalid SurveyMonkey token. Please check your API token.")
        else:
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
                pages = data.get('pages', [])
                logger.info(f"SurveyMonkey details for ID {survey_id}: {len(pages)} pages")
                return data
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                logger.warning(f"Survey details attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
        st.error(f"❌ Failed to fetch survey details for ID {survey_id}: {str(e)}")
        return {}

def extract_questions_from_surveymonkey(survey_data):
    """Enhanced question extraction with question_id and choice details"""
    try:
        questions = []
        if not survey_data or 'pages' not in survey_data:
            logger.warning("No pages found in survey data")
            return questions
        
        global_position = 0
        survey_title = survey_data.get('title', 'Unknown Survey')
        survey_id = survey_data.get('id', '')
        
        for page_idx, page in enumerate(survey_data.get('pages', [])):
            if 'questions' not in page:
                continue
                
            for question in page.get('questions', []):
                if 'headings' not in question or not question['headings']:
                    continue
                
                question_text = question['headings'][0].get('heading', '').strip()
                if not question_text:
                    continue
                
                global_position += 1
                question_id = question.get('id', f'q_{global_position}')
                family = question.get('family', 'unknown')
                subtype = question.get('subtype', 'unknown')
                
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
                    schema_type = "Other"
                
                # Extract choices
                choices = []
                if 'answers' in question and 'choices' in question['answers']:
                    for choice in question['answers']['choices']:
                        choice_text = choice.get('text', '').strip()
                        if choice_text:
                            choices.append({
                                'choice_id': choice.get('id', ''),
                                'choice_text': choice_text,
                                'position': choice.get('position', 0)
                            })
                
                # Add main question
                questions.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'survey_id': survey_id,
                    'survey_title': survey_title,
                    'position': global_position,
                    'page_number': page_idx + 1,
                    'schema_type': schema_type,
                    'family': family,
                    'subtype': subtype,
                    'is_choice': False,
                    'parent_question': None,
                    'choices': choices,
                    'choice_count': len(choices),
                    'is_required': question.get('required', {}).get('value', False)
                })
                
                # Add individual choices as separate records
                for choice in choices:
                    questions.append({
                        'question_id': question_id,
                        'question_text': f"{question_text} - {choice['choice_text']}",
                        'survey_id': survey_id,
                        'survey_title': survey_title,
                        'position': global_position,
                        'page_number': page_idx + 1,
                        'schema_type': schema_type,
                        'family': family,
                        'subtype': subtype,
                        'is_choice': True,
                        'parent_question': question_text,
                        'choice_id': choice['choice_id'],
                        'choice_text': choice['choice_text'],
                        'choice_position': choice['position'],
                        'choices': [],
                        'choice_count': 0,
                        'is_required': False
                    })
        
        logger.info(f"Extracted {len(questions)} questions/choices from survey")
        return questions
        
    except Exception as e:
        logger.error(f"Error extracting SurveyMonkey questions: {e}")
        st.error(f"❌ Failed to extract questions: {str(e)}")
        return []

# Enhanced Matching Functions
def load_sentence_transformer():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def ultra_fast_semantic_matching(surveymonkey_questions, use_optimized_reference=True):
    """Ultra-fast matching using optimized 1:1 question bank"""
    if not surveymonkey_questions:
        st.warning("⚠️ No SurveyMonkey questions provided for matching")
        return []
    
    try:
        # Use session state threshold if available, otherwise use default
        threshold = st.session_state.get('semantic_threshold', SEMANTIC_THRESHOLD)
        
        if use_optimized_reference:
            optimized_ref = st.session_state.get('primary_matching_reference')
            if optimized_ref is None or optimized_ref.empty:
                logger.warning("Optimized reference not available, building now...")
                df_reference = get_all_reference_questions_from_snowflake()
                if not df_reference.empty:
                    optimized_ref, _ = build_optimized_1to1_question_bank(df_reference)
                else:
                    st.error("❌ No reference data available for optimization")
                    return []
            
            ref_texts = optimized_ref['best_question'].tolist()
            logger.info(f"Using optimized reference with {len(optimized_ref)} unique questions")
        else:
            df_reference = get_all_reference_questions_from_snowflake()
            ref_texts = df_reference['HEADING_0'].tolist()
            optimized_ref = df_reference
        
        # Perform semantic matching
        model = load_sentence_transformer()
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        
        logger.info(f"Encoding {len(sm_texts)} SurveyMonkey questions against {len(ref_texts)} reference")
        
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)
        similarities = util.cos_sim(sm_embeddings, ref_embeddings)
        
        matched_results = []
        
        for i, sm_question in enumerate(surveymonkey_questions):
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            
            result = sm_question.copy()
            
            if best_score >= threshold:  # Use session state threshold
                matched_row = optimized_ref.iloc[best_match_idx]
                
                if use_optimized_reference:
                    result['matched_uid'] = matched_row['uid']
                    result['matched_heading_0'] = matched_row['best_question']
                    result['conflict_resolved'] = matched_row.get('has_conflicts', False)
                    result['uid_authority'] = matched_row.get('winner_count', 0)
                    result['conflict_severity'] = matched_row.get('conflict_severity', 0)
                    result['quality_score'] = matched_row.get('quality_score', 0)
                else:
                    result['matched_uid'] = matched_row['UID']
                    result['matched_heading_0'] = matched_row['HEADING_0']
                    result['conflict_resolved'] = False
                    result['uid_authority'] = matched_row.get('OCCURRENCE_COUNT', 0)
                    result['conflict_severity'] = 0
                    result['quality_score'] = score_question_quality(matched_row['HEADING_0'])
                
                result['match_score'] = best_score
                result['match_confidence'] = "High" if best_score >= 0.8 else "Medium"
            else:
                result['matched_uid'] = None
                result['matched_heading_0'] = None
                result['match_score'] = best_score
                result['match_confidence'] = "Low"
                result['conflict_resolved'] = False
                result['uid_authority'] = 0
                result['conflict_severity'] = 0
                result['quality_score'] = 0
            
            matched_results.append(result)
        
        logger.info(f"Ultra-fast semantic matching completed: {len(matched_results)} results")
        return matched_results
        
    except Exception as e:
        logger.error(f"Ultra-fast semantic matching failed: {e}")
        st.error(f"❌ Ultra-fast matching failed: {str(e)}")
        return []

# Export Functions (from streamlit_uid_combined_with_choices_updated.py)
def prepare_export_data(matched_results):
    """Prepare data for Snowflake export with enhanced metadata"""
    if not matched_results:
        return pd.DataFrame()
    
    try:
        export_data = []
        
        for result in matched_results:
            # Base export record
            export_record = {
                'survey_id': result.get('survey_id', ''),
                'survey_title': result.get('survey_title', ''),
                'question_id': result.get('question_id', ''),  # Added question_id
                'heading_0': result.get('question_text', ''),
                'uid': result.get('matched_uid', ''),
                'position': result.get('position', 0),
                'is_choice': result.get('is_choice', False),
                'parent_question': result.get('parent_question', ''),
                'schema_type': result.get('schema_type', ''),
                'match_confidence': result.get('match_confidence', ''),
                'match_score': result.get('match_score', 0.0),
                'conflict_resolved': result.get('conflict_resolved', False),
                'uid_authority': result.get('uid_authority', 0),
                'quality_score': result.get('quality_score', 0),
                'is_required': result.get('is_required', False),
                'page_number': result.get('page_number', 1),
                'choice_id': result.get('choice_id', ''),
                'choice_text': result.get('choice_text', ''),
                'choice_position': result.get('choice_position', 0)
            }
            export_data.append(export_record)
        
        export_df = pd.DataFrame(export_data)
        
        # Add main question UID and position for choices
        if not export_df.empty:
            main_questions_df = export_df[export_df['is_choice'] == False].copy()
            
            export_df['Main_Question_UID'] = export_df.apply(
                lambda row: main_questions_df[main_questions_df['heading_0'] == row['parent_question']]['uid'].iloc[0]
                if row['is_choice'] and pd.notnull(row['parent_question']) and 
                   not main_questions_df[main_questions_df['heading_0'] == row['parent_question']].empty
                else row['uid'],
                axis=1
            )
            
            export_df['Main_Question_Position'] = export_df.apply(
                lambda row: main_questions_df[main_questions_df['heading_0'] == row['parent_question']]['position'].iloc[0]
                if row['is_choice'] and pd.notnull(row['parent_question']) and 
                   not main_questions_df[main_questions_df['heading_0'] == row['parent_question']].empty
                else row['position'],
                axis=1
            )
        
        return export_df
        
    except Exception as e:
        logger.error(f"Failed to prepare export data: {e}")
        st.error(f"❌ Failed to prepare export data: {str(e)}")
        return pd.DataFrame()

def create_export_preview(export_df):
    """Create preview for Snowflake upload with question_id"""
    if export_df.empty:
        return pd.DataFrame()
    
    try:
        # Create preview with essential columns including question_id
        preview_columns = [
            'survey_id', 'survey_title', 'question_id', 'heading_0', 
            'Main_Question_Position', 'Main_Question_UID', 'match_confidence',
            'conflict_resolved', 'uid_authority'
        ]
        
        preview_df = export_df[preview_columns].copy()
        preview_df = preview_df.rename(columns={
            'survey_id': 'SurveyID',
            'survey_title': 'SurveyName',
            'question_id': 'QuestionID',  # Added QuestionID to preview
            'heading_0': 'Question Info',
            'Main_Question_Position': 'QuestionPosition',
            'Main_Question_UID': 'UID',
            'match_confidence': 'Confidence',
            'conflict_resolved': 'ConflictResolved',
            'uid_authority': 'UIDAuthority'
        })
        
        return preview_df
        
    except Exception as e:
        logger.error(f"Failed to create export preview: {e}")
        return pd.DataFrame()

def upload_to_snowflake(export_df):
    """Upload data to Snowflake"""
    try:
        if export_df.empty:
            st.error("❌ No data to upload")
            return False
        
        # Check for required UIDs
        main_questions_without_uid = export_df[
            (export_df['is_choice'] == False) & 
            (export_df['uid'].isna() | (export_df['uid'] == ''))
        ]
        
        if not main_questions_without_uid.empty:
            st.error(f"❌ {len(main_questions_without_uid)} main questions missing UIDs. All main questions must have UIDs before upload.")
            return False
        
        with st.spinner("⬆️ Uploading to Snowflake..."):
            with get_snowflake_engine().connect() as conn:
                export_df.to_sql(
                    'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                    conn,
                    schema='DBT_SURVEY_MONKEY',
                    if_exists='append',
                    index=False
                )
        
        st.success(f"✅ Successfully uploaded {len(export_df)} records to Snowflake!")
        return True
        
    except Exception as e:
        logger.error(f"Snowflake upload failed: {e}")
        if "250001" in str(e):
            st.error("❌ Snowflake upload failed: User account is locked. Contact your Snowflake admin.")
        else:
            st.error(f"❌ Snowflake upload failed: {str(e)}")
        return False

# Auto-optimization
def ensure_optimized_reference():
    """Ensure optimized reference is available"""
    if st.session_state.get('optimization_attempted', False):
        return
    
    try:
        opt_ref = st.session_state.get('primary_matching_reference')
        if opt_ref is None or opt_ref.empty:
            with st.spinner("🔧 Building optimized 1:1 question bank..."):
                df_reference = get_all_reference_questions_from_snowflake()
                if not df_reference.empty:
                    optimized_df, conflicts_df = build_optimized_1to1_question_bank(df_reference)
                    if not optimized_df.empty:
                        logger.info("✅ Auto-built optimized 1:1 question bank")
                    else:
                        logger.warning("❌ Failed to build optimized question bank")
                else:
                    logger.warning("❌ No Snowflake reference data for optimization")
        
        st.session_state.optimization_attempted = True
        
    except Exception as e:
        logger.error(f"Auto-build optimization failed: {e}")
        st.session_state.optimization_attempted = True

# Page Definitions
def home_dashboard():
    st.markdown('<div class="main-header">🏠 UID Matcher Pro Enhanced Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("Welcome to UID Matcher Pro Enhanced with advanced 1:1 question bank optimization!")
    
    # Performance stats
    perf_stats = get_performance_stats()
    opt_ref = st.session_state.get('primary_matching_reference')
    conflicts_summary = st.session_state.get('uid_conflicts_summary')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_questions = len(opt_ref) if opt_ref is not None and not opt_ref.empty else 0
        st.metric("🎯 Unique Questions", f"{unique_questions:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        conflicts_resolved = len(conflicts_summary) if conflicts_summary is not None and not conflicts_summary.empty else 0
        st.metric("🔥 Conflicts Resolved", f"{conflicts_resolved:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        optimization_status = "✅ Ready" if opt_ref is not None and not opt_ref.empty else "❌ Not Built"
        st.metric("⚡ Optimization Status", optimization_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        last_opt = perf_stats.get('last_optimization_time')
        last_opt_str = last_opt.strftime("%H:%M") if last_opt else "Never"
        st.metric("🕒 Last Optimization", last_opt_str)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("---")
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 View Surveys", use_container_width=True):
            st.session_state.page = "View Surveys"
            st.rerun()
    
    with col2:
        if st.button("🎯 Build Question Bank", use_container_width=True):
            st.session_state.page = "Optimized 1:1 Question Bank"
            st.rerun()
    
    with col3:
        if st.button("📊 View Conflicts", use_container_width=True):
            st.session_state.page = "Conflict Dashboard"
            st.rerun()

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
            
            # Enhanced survey display
            survey_df = pd.DataFrame(surveys)
            if 'question_count' in survey_df.columns:
                st.dataframe(survey_df[['id', 'title', 'question_count']], use_container_width=True)
            else:
                st.dataframe(survey_df[['id', 'title']], use_container_width=True)
            
            # Survey selection
            survey_options = [
                f"{survey.get('id', 'N/A')} - {survey.get('title', 'Untitled Survey')}"
                for survey in surveys
            ]
            selected_survey = st.selectbox("Select Survey", survey_options)
            
            if selected_survey:
                selected_id = selected_survey.split(" - ")[0]
                survey = next((s for s in surveys if s.get('id') == selected_id), None)
                
                if survey:
                    st.write(f"**Survey ID:** {survey.get('id', 'N/A')}")
                    st.write(f"**Title:** {survey.get('title', 'Untitled Survey')}")
                    question_count = survey.get('question_count', 0)
                    st.write(f"**Questions:** {question_count if question_count > 0 else 'No questions found'}")
                    
                    if st.button("📋 Extract Questions & Configure"):
                        details = get_survey_details(survey['id'], token)
                        questions = extract_questions_from_surveymonkey(details)
                        
                        if not questions:
                            st.warning("⚠️ No questions extracted from this survey")
                        else:
                            st.session_state.questions = questions
                            st.session_state.page = "Configure Survey"
                            st.success(f"✅ Extracted {len(questions)} questions/choices")
                            st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to view surveys: {e}")
        st.error(f"❌ Failed to fetch surveys: {str(e)}")

def configure_survey():
    st.title("⚙️ Configure Survey with Enhanced Matching")
    
    # Check Snowflake connection
    sf_status = True
    try:
        engine = get_snowflake_engine()
        st.session_state.snowflake_engine = engine
    except Exception as e:
        sf_status = False
        logger.error(f"Snowflake connection failed: {e}")
        st.warning("⚠️ Snowflake connection not established")

    # Ensure optimization is built
    ensure_optimized_reference()

    if 'questions' in st.session_state and st.session_state.questions:
        questions = st.session_state.questions
        df_target = pd.DataFrame(questions)
        
        st.markdown("### 📋 Survey Questions with IDs")
        
        # Display enhanced question information
        display_columns = ['question_id', 'question_text', 'schema_type', 'is_choice', 'choice_count']
        if not df_target.empty:
            st.dataframe(df_target[display_columns], use_container_width=True, height=400)
        
        if sf_status:
            st.markdown("### 🔄 Enhanced UID Assignment Process")
            
            # Show optimization status
            opt_ref = st.session_state.get('primary_matching_reference')
            conflicts_summary = st.session_state.get('uid_conflicts_summary')
            
            if opt_ref is not None and not opt_ref.empty:
                st.markdown('<div class="success-card">✅ Optimized 1:1 question bank ready</div>', unsafe_allow_html=True)
                st.write(f"• {len(opt_ref):,} unique questions optimized")
                if conflicts_summary is not None and not conflicts_summary.empty:
                    st.write(f"• {len(conflicts_summary):,} UID conflicts resolved")
                    high_conflicts = len(conflicts_summary[conflicts_summary['is_high_conflict'] == True])
                    if high_conflicts > 0:
                        st.write(f"• {high_conflicts} high-severity conflicts resolved")
            else:
                st.markdown('<div class="warning-card">⚠️ Question bank optimization not built</div>', unsafe_allow_html=True)
                if st.button("🎯 Build Optimization Now"):
                    st.session_state.page = "Optimized 1:1 Question Bank"
                    st.rerun()
                    return
            
            # Matching options
            st.markdown("### ⚡ Matching Configuration")
            
            matching_approach = st.radio(
                "Select matching approach:",
                [
                    "🎯 Ultra-Fast Matching (1:1 Optimized, Recommended)",
                    "✅ Standard Matching"
                ],
                help="Ultra-fast uses conflict-resolved 1:1 mapping for maximum speed and accuracy"
            )
            
            # Run matching
            if st.button("🚀 Run Enhanced UID Matching", type="primary"):
                try:
                    if matching_approach == "🎯 Ultra-Fast Matching (1:1 Optimized, Recommended)":
                        if opt_ref is None or opt_ref.empty:
                            st.error("❌ 1:1 optimization not available. Please build it first.")
                            return
                        
                        with st.spinner("🎯 Running ULTRA-FAST matching with 1:1 optimization..."):
                            matched_results = ultra_fast_semantic_matching(questions, use_optimized_reference=True)
                    else:
                        with st.spinner("✅ Running standard semantic matching..."):
                            df_reference = get_all_reference_questions_from_snowflake()
                            matched_results = ultra_fast_semantic_matching(questions, use_optimized_reference=False)
                    
                    if matched_results:
                        matched_df = pd.DataFrame(matched_results)
                        st.session_state.df_final = matched_df
                        
                        # Show results
                        st.markdown('<div class="success-card">✅ UID matching completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Enhanced metrics
                        high_conf = len(matched_df[matched_df['match_confidence'] == 'High'])
                        medium_conf = len(matched_df[matched_df['match_confidence'] == 'Medium'])
                        low_conf = len(matched_df[matched_df['match_confidence'] == 'Low'])
                        conflicts_resolved = len(matched_df[matched_df.get('conflict_resolved', False) == True])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 High Confidence", high_conf)
                        with col2:
                            st.metric("⚠️ Medium Confidence", medium_conf)
                        with col3:
                            st.metric("❌ Low/No Match", low_conf)
                        with col4:
                            st.metric("🔥 Conflicts Resolved", conflicts_resolved)
                        
                        # Export section
                        st.markdown("### 📤 Export to Snowflake")
                        
                        # Prepare export data
                        export_df = prepare_export_data(matched_results)
                        preview_df = create_export_preview(export_df)
                        
                        if not preview_df.empty:
                            st.markdown("#### 👁️ Preview Data for Snowflake Upload")
                            st.dataframe(preview_df, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download CSV
                                csv_data = export_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Enhanced CSV",
                                    csv_data,
                                    f"enhanced_survey_export_{uuid4()}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Upload to Snowflake
                                if st.button("🚀 Upload to Snowflake", use_container_width=True):
                                    upload_to_snowflake(export_df)
                        
                    else:
                        st.error("❌ No matching results generated")
                        
                except Exception as e:
                    logger.error(f"UID matching failed: {e}")
                    st.error(f"❌ UID matching failed: {str(e)}")
        
        else:
            st.warning("❌ Snowflake connection required for UID assignment")
    
    else:
        st.warning("⚠️ No survey questions loaded. Please view and select a survey first.")

def optimized_question_bank():
    st.title("🎯 Optimized 1:1 Question Bank Builder")
    
    try:
        # Load reference data
        df_reference = get_all_reference_questions_from_snowflake()
        
        if df_reference.empty:
            st.warning("⚠️ No reference questions loaded from Snowflake")
            return
        
        st.markdown('<div class="success-card">✅ Loaded reference data</div>', unsafe_allow_html=True)
        st.write(f"• **Total records:** {len(df_reference):,}")
        st.write(f"• **Unique UIDs:** {df_reference['UID'].nunique():,}")
        st.write(f"• **Unique questions:** {df_reference['HEADING_0'].nunique():,}")
        
        # Build optimization
        if st.button("🚀 Build Optimized 1:1 Question Bank", type="primary"):
            with st.spinner("🔧 Building optimized question bank with conflict resolution..."):
                optimized_df, conflicts_df = build_optimized_1to1_question_bank(df_reference)
                
                if not optimized_df.empty:
                    st.markdown('<div class="success-card">✅ Optimization completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Unique Questions", len(optimized_df))
                    with col2:
                        st.metric("🔥 Conflicts Resolved", len(conflicts_df))
                    with col3:
                        high_conflicts = len(conflicts_df[conflicts_df['is_high_conflict'] == True]) if not conflicts_df.empty else 0
                        st.metric("⚠️ High Conflicts", high_conflicts)
                    
                    # Show optimized question bank
                    st.markdown("### 📊 Optimized 1:1 Question Bank")
                    if st.session_state.optimized_question_bank is not None:
                        display_df = st.session_state.optimized_question_bank.head(100)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download option
                        csv_data = st.session_state.optimized_question_bank.to_csv(index=False)
                        st.download_button(
                            "📥 Download Optimized Question Bank",
                            csv_data,
                            f"optimized_question_bank_{uuid4()}.csv",
                            "text/csv"
                        )
                
                else:
                    st.error("❌ Failed to build optimization")
        
        # Show current optimization status
        opt_ref = st.session_state.get('primary_matching_reference')
        if opt_ref is not None and not opt_ref.empty:
            st.markdown("---")
            st.markdown("### ℹ️ Current Optimization Status")
            st.markdown('<div class="success-card">✅ Optimization already built and ready</div>', unsafe_allow_html=True)
            st.write(f"• **Questions optimized:** {len(opt_ref):,}")
            
            conflicts_summary = st.session_state.get('uid_conflicts_summary')
            if conflicts_summary is not None and not conflicts_summary.empty:
                st.write(f"• **Conflicts resolved:** {len(conflicts_summary):,}")
                
                if st.button("📊 View Conflict Details"):
                    st.session_state.page = "Conflict Dashboard"
                    st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to build optimized question bank: {e}")
        st.error(f"❌ Failed to build optimized question bank: {str(e)}")

def conflict_dashboard():
    st.title("📊 UID Conflict Resolution Dashboard")
    
    conflicts_summary = st.session_state.get('uid_conflicts_summary')
    
    if conflicts_summary is None or conflicts_summary.empty:
        st.warning("⚠️ No conflict data available. Please build the optimized question bank first.")
        if st.button("🎯 Build Question Bank"):
            st.session_state.page = "Optimized 1:1 Question Bank"
            st.rerun()
        return
    
    st.markdown('<div class="success-card">✅ Conflict analysis complete</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔥 Total Conflicts", len(conflicts_summary))
    
    with col2:
        high_conflicts = len(conflicts_summary[conflicts_summary['is_high_conflict'] == True])
        st.metric("⚠️ High Severity", high_conflicts)
    
    with col3:
        avg_authority_diff = conflicts_summary['authority_difference'].mean()
        st.metric("📊 Avg Authority Gap", f"{avg_authority_diff:.1f}%")
    
    with col4:
        total_competing_uids = conflicts_summary['competing_uids'].sum()
        st.metric("🆔 Competing UIDs", total_competing_uids)
    
    # Detailed conflict analysis
    st.markdown("### 🔍 Detailed Conflict Analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.selectbox(
            "Filter by severity:",
            ["All", "High Conflicts Only", "Medium Conflicts Only"]
        )
    
    with col2:
        min_authority_diff = st.slider(
            "Minimum authority difference (%):",
            0.0, 100.0, 0.0, 5.0
        )
    
    # Apply filters
    filtered_conflicts = conflicts_summary.copy()
    
    if severity_filter == "High Conflicts Only":
        filtered_conflicts = filtered_conflicts[filtered_conflicts['is_high_conflict'] == True]
    elif severity_filter == "Medium Conflicts Only":
        filtered_conflicts = filtered_conflicts[filtered_conflicts['is_high_conflict'] == False]
    
    if min_authority_diff > 0:
        filtered_conflicts = filtered_conflicts[filtered_conflicts['authority_difference'] >= min_authority_diff]
    
    # Display conflicts
    if not filtered_conflicts.empty:
        st.dataframe(filtered_conflicts, use_container_width=True)
        
        # Download option
        csv_data = filtered_conflicts.to_csv(index=False)
        st.download_button(
            "📥 Download Conflict Analysis",
            csv_data,
            f"uid_conflicts_analysis_{uuid4()}.csv",
            "text/csv"
        )
        
        # Most problematic questions
        st.markdown("### ⚠️ Most Problematic Questions")
        top_conflicts = filtered_conflicts.nlargest(10, 'conflict_severity')
        
        for idx, row in top_conflicts.iterrows():
            with st.expander(f"Conflict: {row['question'][:80]}..."):
                st.write(f"**Winner UID:** {row['winner_uid']} ({row['winner_count']} occurrences, {row['winner_percentage']:.1f}%)")
                st.write(f"**Top Competitor:** UID {row['top_competitor_uid']} ({row['top_competitor_count']} occurrences, {row['top_competitor_percentage']:.1f}%)")
                st.write(f"**Authority Difference:** {row['authority_difference']:.1f}%")
                st.write(f"**Conflict Severity:** {row['conflict_severity']} competing records")
                if row['is_high_conflict']:
                    st.write("🚨 **HIGH SEVERITY CONFLICT**")
    
    else:
        st.info("ℹ️ No conflicts match the selected filters")

def get_performance_stats():
    """Get performance statistics"""
    try:
        opt_ref = st.session_state.get('primary_matching_reference')
        unique_questions = len(opt_ref) if opt_ref is not None and not opt_ref.empty else 0
        last_optimization = st.session_state.get('last_optimization_time', None)
        
        return {
            'unique_questions_loaded': unique_questions,
            'last_optimization_time': last_optimization
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {'unique_questions_loaded': 0, 'last_optimization_time': None}

def matching_dashboard():
    st.title("📊 Matching Results Dashboard")
    
    if 'df_final' in st.session_state and st.session_state.df_final is not None:
        df = st.session_state.df_final
        
        # Enhanced display with question_id
        display_columns = ['question_id', 'question_text', 'matched_uid', 'match_confidence', 
                         'match_score', 'conflict_resolved', 'uid_authority']
        
        available_columns = [col for col in display_columns if col in df.columns]
        st.dataframe(df[available_columns], use_container_width=True)
        
        # Export current results
        if st.button("📥 Download Current Results"):
            export_df = prepare_export_data(df.to_dict('records'))
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"matching_results_{uuid4()}.csv",
                "text/csv"
            )
    else:
        st.warning("⚠️ No matching results available. Run UID matching first.")

def settings():
    st.title("⚙️ Settings")
    
    st.markdown("### 🔧 Configuration Settings")
    
    # UID Governance settings
    st.markdown("#### UID Governance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conflict_threshold = st.number_input(
            "Conflict Resolution Threshold",
            min_value=1,
            max_value=100,
            value=UID_GOVERNANCE['conflict_resolution_threshold'],
            help="Minimum count for a UID to be considered in conflicts"
        )
    
    with col2:
        high_conflict_threshold = st.number_input(
            "High Conflict Threshold",
            min_value=50,
            max_value=1000,
            value=UID_GOVERNANCE['high_conflict_threshold'],
            help="Threshold for high-severity conflicts"
        )
    
    # Semantic matching settings
    st.markdown("#### Semantic Matching")
    
    semantic_threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.5,
        max_value=1.0,
        value=SEMANTIC_THRESHOLD,
        step=0.05,
        help="Minimum similarity score for matching"
    )
    
    if st.button("💾 Save Settings"):
        # Save settings to session state instead of global variables
        st.session_state.semantic_threshold = semantic_threshold
        st.session_state.conflict_resolution_threshold = conflict_threshold
        st.session_state.high_conflict_threshold = high_conflict_threshold
        
        st.success("✅ Settings saved successfully!")
    
    # System information
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    
    perf_stats = get_performance_stats()
    
    st.write(f"**Optimized Questions Loaded:** {perf_stats['unique_questions_loaded']:,}")
    
    last_opt = perf_stats.get('last_optimization_time')
    if last_opt:
        st.write(f"**Last Optimization:** {last_opt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("**Last Optimization:** Never")
    
    # Cache management
    st.markdown("#### 🗂️ Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared")
    
    with col2:
        if st.button("🔄 Reset Optimization"):
            st.session_state.primary_matching_reference = None
            st.session_state.uid_conflicts_summary = None
            st.session_state.optimized_question_bank = None
            st.session_state.optimization_attempted = False
            st.success("✅ Optimization reset")

# Enhanced Sidebar Navigation
def create_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 UID Matcher Pro Enhanced")
        st.markdown("Advanced question bank optimization with conflict resolution")
        
        # Connection status
        try:
            get_snowflake_engine()
            sf_status = "✅ Connected"
        except:
            sf_status = "❌ Failed"
        
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", "")
            sm_status = "✅ Available" if token else "❌ Missing"
        except:
            sm_status = "❌ Failed"
        
        st.markdown("**🔗 Connection Status**")
        st.markdown(f"❄️ Snowflake: {sf_status}")
        st.markdown(f"📊 SurveyMonkey: {sm_status}")
        
        # Optimization status
        opt_ref = st.session_state.get('primary_matching_reference')
        opt_status = "✅ Ready" if opt_ref is not None and not opt_ref.empty else "❌ Not Built"
        st.markdown(f"🎯 Optimization: {opt_status}")
        
        st.markdown("---")
        
        # Navigation
        pages = [
            "Home Dashboard",
            "View Surveys", 
            "Configure Survey",
            "Optimized 1:1 Question Bank",
            "Conflict Dashboard",
            "Matching Dashboard",
            "Settings"
        ]
        
        selected_page = st.selectbox("📍 Navigate to:", pages)
        return selected_page

# Main App Execution
def main():
    try:
        # Create sidebar and get selected page
        selected_page = create_sidebar()
        st.session_state.page = selected_page
        
        # Route to appropriate page
        if st.session_state.page == "Home Dashboard":
            home_dashboard()
        elif st.session_state.page == "View Surveys":
            view_surveys()
        elif st.session_state.page == "Configure Survey":
            configure_survey()
        elif st.session_state.page == "Optimized 1:1 Question Bank":
            optimized_question_bank()
        elif st.session_state.page == "Conflict Dashboard":
            conflict_dashboard()
        elif st.session_state.page == "Matching Dashboard":
            matching_dashboard()
        elif st.session_state.page == "Settings":
            settings()
        
    except Exception as e:
        logger.error(f"Main app execution failed: {e}")
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please check the logs and try refreshing the page.")

# Footer
def create_footer():
    st.markdown("---")
    st.markdown("### 🔗 Quick Links & Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📝 Submit New Questions**")
        st.markdown("[Google Form](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    
    with col2:
        st.markdown("**🆔 Submit New UIDs**")
        st.markdown("[Google Form](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")
    
    with col3:
        st.markdown("**📊 Current Status**")
        perf_stats = get_performance_stats()
        st.write(f"Questions: {perf_stats['unique_questions_loaded']:,}")
        
        conflicts_summary = st.session_state.get('uid_conflicts_summary')
        conflicts_count = len(conflicts_summary) if conflicts_summary is not None and not conflicts_summary.empty else 0
        st.write(f"Conflicts: {conflicts_count:,}")

# Execute main application
if __name__ == "__main__":
    main()
    create_footer()
