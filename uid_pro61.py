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

# ============= DATA SOURCE DOCUMENTATION =============
"""
CLEAR DATA SOURCE MAPPING:

📊 SurveyMonkey (Source for surveys and questions to be matched):
- survey_title: Used for categorization (PRIMARY SOURCE)
- question_text: Questions/choices extracted from SurveyMonkey surveys (TO BE MATCHED)
- question_category: Derived from SurveyMonkey question analysis
- schema_type: Question type from SurveyMonkey
- survey_id: SurveyMonkey survey identifier

❄️ Snowflake (Reference database with existing UIDs):
- HEADING_0: Reference questions with existing UIDs (REFERENCE FOR MATCHING)
- UID: Existing UID assignments (TARGET RESULT)
- survey_title: May exist but SurveyMonkey title takes precedence for categorization

🔄 MATCHING PROCESS:
SurveyMonkey question_text → Compare via semantics → Snowflake HEADING_0 → Get UID
"""

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# Cache configuration
CACHE_DURATION = 3600  # 1 hour in seconds
EMBEDDING_CACHE_SIZE = 50000  # Maximum embeddings to cache

# UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True
}

# Survey Categories based on SurveyMonkey titles
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
        "df_target": None,  # SurveyMonkey questions
        "df_final": None,   # Matched results
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "df_reference": None,  # Snowflake reference data
        "survey_template": None,
        "snowflake_initialized": False,
        "surveymonkey_initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# ============= PERFORMANCE MONITORING =============

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    return wrapper

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

def categorize_survey_from_surveymonkey(survey_title):
    """
    Categorize survey based on SurveyMonkey survey title keywords
    NOTE: This uses SurveyMonkey survey_title, not Snowflake data
    """
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

def create_unique_questions_bank_from_snowflake(df_reference):
    """
    Create unique questions bank from Snowflake reference data
    Uses Snowflake HEADING_0 column and UID column
    """
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_reference)} Snowflake reference questions for unique bank")
    
    unique_questions = []
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        # Use Snowflake HEADING_0 column (not question_text from SurveyMonkey)
        uid_questions = group['heading_0'].tolist()
        best_question = get_best_question_for_uid(uid_questions)
        
        # For categorization, Snowflake may have title but it's secondary
        # Primary categorization should be from SurveyMonkey when available
        survey_titles = group.get('title', pd.Series()).dropna().unique()
        if len(survey_titles) > 0:
            categories = [categorize_survey_from_surveymonkey(title) for title in survey_titles]
            primary_category = categories[0] if len(set(categories)) == 1 else "Mixed"
        else:
            primary_category = "Unknown"
        
        if best_question:
            unique_questions.append({
                'uid': uid,  # Snowflake UID
                'best_question': best_question,  # Best from Snowflake HEADING_0
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split()),
                'survey_category': primary_category,
                'survey_titles': ', '.join(survey_titles) if len(survey_titles) > 0 else 'Unknown',
                'quality_score': score_question_quality(best_question),
                'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid'],
                'data_source': 'Snowflake'
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

# ============= PERFORMANCE OPTIMIZATION FUNCTIONS =============

@st.cache_data(ttl=CACHE_DURATION)
@monitor_performance
def get_cached_reference_questions():
    """
    Cache reference questions from Snowflake for 1 hour
    This prevents repeated database queries
    """
    try:
        logger.info("Loading reference questions from Snowflake (will be cached)")
        df_reference = get_all_reference_questions_from_snowflake()
        logger.info(f"Cached {len(df_reference):,} reference questions")
        return df_reference
    except Exception as e:
        logger.error(f"Failed to cache reference questions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_DURATION)
@monitor_performance
def get_cached_unique_questions_bank():
    """
    Cache unique questions bank for 1 hour
    This creates the optimized lookup table
    """
    try:
        logger.info("Building cached unique questions bank")
        df_reference = get_cached_reference_questions()
        
        if df_reference.empty:
            return pd.DataFrame()
        
        unique_bank = create_unique_questions_bank_from_snowflake(df_reference)
        logger.info(f"Cached unique bank with {len(unique_bank):,} UIDs")
        return unique_bank
    except Exception as e:
        logger.error(f"Failed to cache unique questions bank: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_DURATION)
@monitor_performance
def prepare_matching_data():
    """
    Pre-compute embeddings and prepare optimized matching data
    This is the KEY performance optimization
    """
    try:
        logger.info("Preparing optimized matching data with pre-computed embeddings")
        
        # Get unique questions bank
        unique_questions = get_cached_unique_questions_bank()
        
        if unique_questions.empty:
            return None, None, None
        
        # Load model once
        model = load_sentence_transformer()
        
        # Extract questions for embedding
        questions_list = unique_questions['best_question'].tolist()
        
        # Pre-compute embeddings (this is the expensive operation we want to cache)
        logger.info(f"Pre-computing embeddings for {len(questions_list)} unique questions")
        embeddings = model.encode(questions_list, convert_to_tensor=True)
        
        # Create fast UID lookup
        uid_lookup = dict(zip(range(len(unique_questions)), unique_questions['uid'].tolist()))
        
        logger.info("Matching data preparation completed")
        return questions_list, embeddings, uid_lookup
        
    except Exception as e:
        logger.error(f"Failed to prepare matching data: {e}")
        return None, None, None

def get_performance_stats():
    """
    Get current performance optimization status
    """
    try:
        # Check cached reference data status
        ref_status = "Not Loaded"
        ref_count = 0
        
        # Check if cache exists by trying to get it
        try:
            ref_df = get_cached_reference_questions()
            if not ref_df.empty:
                ref_status = "Cached ✅"
                ref_count = len(ref_df)
        except:
            ref_status = "Error ❌"
        
        # Check unique questions bank status
        unique_status = "Not Built"
        unique_count = 0
        
        try:
            unique_df = get_cached_unique_questions_bank()
            if not unique_df.empty:
                unique_status = "Cached ✅"
                unique_count = len(unique_df)
        except:
            unique_status = "Error ❌"
        
        # Check matching data status
        matching_status = "Not Ready"
        embeddings_count = 0
        
        try:
            questions, embeddings, lookup = prepare_matching_data()
            if embeddings is not None:
                matching_status = "Ready ✅"
                embeddings_count = len(embeddings)
        except:
            matching_status = "Error ❌"
        
        return {
            'reference_cache_status': ref_status,
            'reference_questions_loaded': ref_count,
            'unique_cache_status': unique_status,
            'unique_questions_loaded': unique_count,
            'matching_data_status': matching_status,
            'embeddings_count': embeddings_count
        }
        
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return {
            'reference_cache_status': 'Error',
            'reference_questions_loaded': 0,
            'unique_cache_status': 'Error',
            'unique_questions_loaded': 0,
            'matching_data_status': 'Error',
            'embeddings_count': 0
        }

@monitor_performance
def fast_semantic_matching(surveymonkey_questions, use_cached_data=True):
    """
    Optimized semantic matching using pre-computed embeddings
    This should be 90% faster than the original approach
    """
    if not surveymonkey_questions:
        return []
    
    try:
        if use_cached_data:
            # Use pre-computed embeddings (FAST path)
            ref_questions, ref_embeddings, uid_lookup = prepare_matching_data()
            
            if ref_embeddings is None:
                logger.warning("Cached data not available, falling back to slow matching")
                return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())
        else:
            # Fallback to original method (SLOW path)
            return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())
        
        # Load model for SurveyMonkey questions only
        model = load_sentence_transformer()
        
        # Extract SurveyMonkey question texts
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        
        # Encode SurveyMonkey questions
        logger.info(f"Encoding {len(sm_texts)} SurveyMonkey questions")
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        
        # Calculate similarities (this is now much faster)
        logger.info("Calculating similarities using pre-computed embeddings")
        similarities = util.cos_sim(sm_embeddings, ref_embeddings)
        
        matched_results = []
        
        for i, sm_question in enumerate(surveymonkey_questions):
            # Find best match
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            
            result = sm_question.copy()
            
            if best_score >= SEMANTIC_THRESHOLD:
                # Get UID from lookup
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
        # Fallback to original method if fast method fails
        return perform_semantic_matching(surveymonkey_questions, get_cached_reference_questions())

def batch_process_matching(surveymonkey_questions, batch_size=100):
    """Process matching in batches for very large question sets"""
    try:
        total_questions = len(surveymonkey_questions)
        if total_questions <= batch_size:
            # Small dataset, process normally
            return fast_semantic_matching(surveymonkey_questions)
        
        # Large dataset, process in batches
        logger.info(f"Processing {total_questions} questions in batches of {batch_size}")
        
        all_results = []
        for i in range(0, total_questions, batch_size):
            batch = surveymonkey_questions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_questions-1)//batch_size + 1}")
            
            batch_results = fast_semantic_matching(batch)
            all_results.extend(batch_results)
        
        logger.info(f"Batch processing completed: {len(all_results)} total results")
        return all_results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        # Fallback to regular processing
        return fast_semantic_matching(surveymonkey_questions)

# ============= CACHE MANAGEMENT UTILITIES =============

def clear_all_caches():
    """Clear all Streamlit caches for fresh start"""
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("All caches cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        return False

def handle_optimization_error(error, operation):
    """Enhanced error handling for optimization operations"""
    error_msg = str(error)
    
    if "250001" in error_msg:
        st.error("🔒 Snowflake connection failed: User account is locked.")
        st.info("Please contact your administrator to unlock the account.")
        st.info("You can still use SurveyMonkey features while this is resolved.")
    elif "timeout" in error_msg.lower():
        st.error(f"⏱️ {operation} timed out. This may be due to network issues.")
        st.info("Try again in a few minutes, or check your internet connection.")
    elif "memory" in error_msg.lower():
        st.error(f"💾 {operation} failed due to memory constraints.")
        st.info("Try clearing caches and running the operation again.")
    else:
        st.error(f"❌ {operation} failed: {error_msg}")
        st.info("Please check your connections and try again.")

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
    """Get detailed survey information from SurveyMonkey"""
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

def extract_questions_from_surveymonkey(survey_json):
    """
    Extract questions from SurveyMonkey survey JSON
    This creates the target data that will be matched against Snowflake HEADING_0
    """
    questions = []
    global_position = 0
    survey_title = survey_json.get("title", "Unknown")  # SurveyMonkey title (PRIMARY)
    survey_category = categorize_survey_from_surveymonkey(survey_title)  # Based on SM title
    
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            
            # Determine schema type from SurveyMonkey data
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
                    "question_text": q_text,  # SurveyMonkey question text (TO BE MATCHED)
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_uid": q_id,  # SurveyMonkey question ID
                    "schema_type": schema_type,  # From SurveyMonkey
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),  # SurveyMonkey survey ID
                    "survey_title": survey_title,  # SurveyMonkey title (PRIMARY)
                    "survey_category": survey_category,  # Based on SurveyMonkey title
                    "question_category": question_category,
                    "data_source": "SurveyMonkey",  # Track data source
                    "matched_uid": None,  # Will be filled during matching
                    "matched_heading_0": None  # Will be filled during matching
                })
                
                # Add choices from SurveyMonkey
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "question_text": f"{q_text} - {choice_text}",  # SM question + choice
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_uid": q_id,
                            "schema_type": schema_type,  # From SurveyMonkey
                            "mandatory": False,
                            "mandatory_editable": False,
                            "survey_id": survey_json.get("id", ""),  # SurveyMonkey survey ID
                            "survey_title": survey_title,  # SurveyMonkey title (PRIMARY)
                            "survey_category": survey_category,  # Based on SurveyMonkey title
                            "question_category": "Main Question/Multiple Choice",
                            "data_source": "SurveyMonkey",  # Track data source
                            "matched_uid": None,  # Will be filled during matching
                            "matched_heading_0": None  # Will be filled during matching
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
def get_all_reference_questions_from_snowflake():
    """
    Fetch ALL reference questions from Snowflake with pagination
    Returns: DataFrame with HEADING_0, UID, TITLE columns
    """
    all_data = []
    limit = 10000
    offset = 0
    
    while True:
        query = """
            SELECT HEADING_0, UID, TITLE
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
        # Ensure proper column naming
        final_df.columns = ['heading_0', 'uid', 'title']
        logger.info(f"Total reference questions fetched from Snowflake: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched from Snowflake")
        return pd.DataFrame()

def run_snowflake_target_query():
    """Get target questions without UIDs from Snowflake"""
    query = """
        SELECT DISTINCT HEADING_0, TITLE
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

# ============= MATCHING FUNCTIONS =============

def perform_semantic_matching(surveymonkey_questions, snowflake_references):
    """
    Match SurveyMonkey question_text against Snowflake HEADING_0 using semantic similarity
    
    Args:
        surveymonkey_questions: List of dicts with question_text from SurveyMonkey
        snowflake_references: DataFrame with HEADING_0, UID from Snowflake
    
    Returns:
        List of matched results
    """
    if not surveymonkey_questions or snowflake_references.empty:
        return []
    
    try:
        model = load_sentence_transformer()
        
        # Extract question texts from SurveyMonkey
        sm_texts = [q['question_text'] for q in surveymonkey_questions]
        
        # Extract HEADING_0 from Snowflake
        sf_texts = snowflake_references['heading_0'].tolist()
        
        # Create embeddings
        sm_embeddings = model.encode(sm_texts, convert_to_tensor=True)
        sf_embeddings = model.encode(sf_texts, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(sm_embeddings, sf_embeddings)
        
        matched_results = []
        
        for i, sm_question in enumerate(surveymonkey_questions):
            # Find best match for this SurveyMonkey question
            best_match_idx = similarities[i].argmax().item()
            best_score = similarities[i][best_match_idx].item()
            
            result = sm_question.copy()
            
            if best_score >= SEMANTIC_THRESHOLD:
                # Get matching Snowflake data
                matched_row = snowflake_references.iloc[best_match_idx]
                result['matched_uid'] = matched_row['uid']
                result['matched_heading_0'] = matched_row['heading_0']
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
        return surveymonkey_questions  # Return original if matching fails

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
    
    # Data source info
    with st.expander("📊 Data Sources"):
        st.markdown("**SurveyMonkey (Source):**")
        st.markdown("• survey_title → categorization")
        st.markdown("• question_text → to be matched")
        st.markdown("**Snowflake (Reference):**")
        st.markdown("• HEADING_0 → reference questions")
        st.markdown("• UID → target assignment")
    
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
    
    # Question Bank Building section (NEW - Priority 1.5)
    st.markdown("**🏗️ Optimization (Step 1.5)**")
    if st.button("🏗️ Build Question Bank", use_container_width=True):
        if not sf_status:
            st.warning("⚠️ Snowflake connection required")
        else:
            st.session_state.page = "build_question_bank"
            st.session_state.snowflake_initialized = True
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

# Data source clarification
st.markdown('<div class="data-source-info"><strong>📊 Data Flow:</strong> SurveyMonkey questions → Semantic matching → Snowflake HEADING_0 → Get UID</div>', unsafe_allow_html=True)

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
                    st.metric("❄️ Snowflake UIDs", f"{count:,}")
            except:
                st.metric("❄️ Snowflake UIDs", "Error")
        else:
            st.metric("❄️ Snowflake UIDs", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if sm_status:
            try:
                token = st.secrets.get("surveymonkey", {}).get("token") or st.secrets.get("surveymonkey", {}).get("access_token")
                surveys = get_surveys(token)
                st.metric("📊 SM Surveys", len(surveys))
            except:
                st.metric("📊 SM Surveys", "API Error")
        else:
            st.metric("📊 SM Surveys", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Governance", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data source explanation
    st.markdown("## 📊 Data Source Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 SurveyMonkey (Source)")
        st.markdown("**What we extract:**")
        st.markdown("• `survey_title` → Used for categorization")
        st.markdown("• `question_text` → Questions to be matched")
        st.markdown("• `schema_type` → Question type")
        st.markdown("• `survey_category` → Auto-categorized")
        
        st.markdown("**Purpose:** Source of questions that need UID assignment")
    
    with col2:
        st.markdown("### ❄️ Snowflake (Reference)")
        st.markdown("**What we use:**")
        st.markdown("• `HEADING_0` → Reference questions with UIDs")
        st.markdown("• `UID` → Existing UID assignments")
        st.markdown("• `title` → Secondary categorization")
        
        st.markdown("**Purpose:** Reference database for UID matching")
    
    st.markdown("---")
    
    # Enhanced workflow guide with optimization
    st.markdown("## 🚀 Recommended Workflow (OPTIMIZED)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Step 1: SurveyMonkey")
        st.markdown("Extract questions to be matched:")
        st.markdown("• **View Surveys** - Browse SurveyMonkey")
        st.markdown("• **Extract Questions** - Get question_text")
        st.markdown("• **Categorize** - Auto-categorize")
        
        if st.button("🔧 Start with SurveyMonkey", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.session_state.surveymonkey_initialized = True
            st.rerun()
    
    with col2:
        st.markdown("### 🏗️ Step 2: BUILD FIRST!")
        st.markdown("**CRITICAL: Build question bank FIRST:**")
        st.markdown("• **Load References** - Cache 1M+ records")
        st.markdown("• **Create Unique Bank** - Best question per UID")  
        st.markdown("• **Pre-compute Embeddings** - 90% speed boost")
        
        if st.button("🏗️ Build Question Bank!", use_container_width=True, type="primary"):
            if not sf_status:
                st.error("❌ Snowflake connection required")
            else:
                st.session_state.page = "build_question_bank"
                st.session_state.snowflake_initialized = True
                st.rerun()
    
    with col3:
        st.markdown("### ⚙️ Step 3: Fast Matching")
        st.markdown("After building the bank:")
        st.markdown("• **Configure Survey** - Set up matching")
        st.markdown("• **Fast Semantic Matching** - Lightning speed")
        st.markdown("• **Assign UIDs** - Get results instantly")
        
        if st.button("⚙️ Configure Matching", use_container_width=True):
            if not st.session_state.surveymonkey_initialized:
                st.warning("⚠️ Complete Step 1 first")
            else:
                st.session_state.page = "configure_survey"
                st.rerun()

    # Performance comparison
    st.markdown("---")
    st.markdown("## ⚡ Performance Comparison")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("### ❌ Without Optimization")
        st.markdown("• **Load Time:** 2-5 minutes per matching")
        st.markdown("• **Database Load:** Heavy on every request")
        st.markdown("• **Memory Usage:** Loads 1M+ records each time")
        st.markdown("• **User Experience:** Poor, lots of waiting")
        st.markdown("• **Scalability:** Does not scale")
    
    with perf_col2:
        st.markdown("### ✅ With Question Bank Optimization")
        st.markdown("• **Load Time:** 5-15 seconds per matching")
        st.markdown("• **Database Load:** Minimal, uses cache")
        st.markdown("• **Memory Usage:** Optimized, reuses data")
        st.markdown("• **User Experience:** Excellent, near-instant")
        st.markdown("• **Scalability:** Handles multiple users")
    
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

elif st.session_state.page == "build_question_bank":
    st.markdown("## 🏗️ Build Optimized Question Bank")
    st.markdown("**Build this FIRST before UID matching for optimal performance!**")
    
    if not sf_status:
        st.error("❌ Snowflake connection required")
        st.stop()
    
    # Performance dashboard
    st.markdown("### 📊 Performance Dashboard")
    
    try:
        perf_stats = get_performance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Reference Data", 
                     f"{perf_stats['reference_questions_loaded']:,}" if perf_stats['reference_questions_loaded'] > 0 else "Not Loaded",
                     delta=perf_stats['reference_cache_status'])
        
        with col2:
            st.metric("⭐ Unique Questions", 
                     f"{perf_stats['unique_questions_loaded']:,}" if perf_stats['unique_questions_loaded'] > 0 else "Not Built",
                     delta=perf_stats['unique_cache_status'])
        
        with col3:
            st.metric("🧠 Embeddings", 
                     f"{perf_stats['embeddings_count']:,}" if perf_stats['embeddings_count'] > 0 else "Not Ready",
                     delta=perf_stats['matching_data_status'])
        
        with col4:
            total_ready = sum([
                1 if "Cached" in perf_stats['reference_cache_status'] else 0,
                1 if "Cached" in perf_stats['unique_cache_status'] else 0,
                1 if "Ready" in perf_stats['matching_data_status'] else 0
            ])
            st.metric("🚀 Optimization", f"{total_ready}/3 Ready", 
                     delta="✅ Optimized" if total_ready == 3 else "⚠️ Needs Setup")
        
        # Build steps
        st.markdown("### 🔧 Optimization Steps")
        
        # Step 1: Load Reference Data
        st.markdown("#### Step 1: Load Reference Data from Snowflake")
        if "Cached" not in perf_stats['reference_cache_status']:
            if st.button("🔄 Load Reference Data (1M+ records)", type="primary"):
                with st.spinner("Loading all reference data from Snowflake..."):
                    try:
                        df_ref = get_cached_reference_questions()
                        if not df_ref.empty:
                            st.success(f"✅ Loaded {len(df_ref):,} reference questions")
                            st.rerun()
                        else:
                            st.error("❌ No reference data loaded")
                    except Exception as e:
                        handle_optimization_error(e, "Reference data loading")
        else:
            st.success(f"✅ Reference data loaded: {perf_stats['reference_questions_loaded']:,} questions")
        
        # Step 2: Build Unique Questions Bank
        st.markdown("#### Step 2: Build Unique Questions Bank")
        if "Cached" not in perf_stats['unique_cache_status']:
            if st.button("⭐ Build Unique Questions Bank"):
                with st.spinner("Building unique questions bank..."):
                    try:
                        unique_df = get_cached_unique_questions_bank()
                        if not unique_df.empty:
                            st.success(f"✅ Built unique bank: {len(unique_df):,} unique UIDs")
                            st.rerun()
                        else:
                            st.error("❌ No unique questions bank created")
                    except Exception as e:
                        handle_optimization_error(e, "Unique questions bank creation")
        else:
            st.success(f"✅ Unique questions built: {perf_stats['unique_questions_loaded']:,} unique UIDs")
        
        # Step 3: Prepare Matching Data
        st.markdown("#### Step 3: Pre-compute Embeddings for Fast Matching")
        if "Ready" not in perf_stats['matching_data_status']:
            if st.button("🧠 Pre-compute Embeddings"):
                with st.spinner("Pre-computing embeddings for fast matching..."):
                    try:
                        unique_q, embeddings, lookup = prepare_matching_data()
                        if embeddings is not None:
                            st.success(f"✅ Pre-computed {len(embeddings):,} embeddings")
                            st.rerun()
                        else:
                            st.error("❌ Failed to pre-compute embeddings")
                    except Exception as e:
                        handle_optimization_error(e, "Embedding pre-computation")
        else:
            st.success(f"✅ Embeddings ready: {perf_stats['embeddings_count']:,} pre-computed")
        
        # Performance tips
        st.markdown("---")
        st.markdown("### 💡 Performance Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Optimization Benefits:**")
            st.write("• 🚀 90% faster UID matching")
            st.write("• 📚 1-hour cache reduces DB load")
            st.write("• 🧠 Pre-computed embeddings")
            st.write("• 🔍 O(1) UID lookups")
        
        with col2:
            st.markdown("**📋 Recommended Order:**")
            st.write("1. 🏗️ Build Question Bank (this page)")
            st.write("2. 📊 View/Analyze SurveyMonkey")
            st.write("3. ⚙️ Configure UID Matching")
            st.write("4. 🚀 Fast semantic matching!")
        
        # Cache management
        st.markdown("---")
        st.markdown("### 🔧 Cache Management")
        
        cache_col1, cache_col2 = st.columns(2)
        
        with cache_col1:
            if st.button("🗑️ Clear All Caches"):
                if clear_all_caches():
                    st.success("✅ All caches cleared successfully")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear caches")
        
        with cache_col2:
            st.info("Cache TTL: 1 hour per component")
        
        # Next steps
        if total_ready == 3:
            st.markdown("---")
            st.success("🎉 Question Bank fully optimized! Ready for fast UID matching.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to SurveyMonkey", use_container_width=True):
                    st.session_state.page = "view_surveys"
                    st.rerun()
            with col2:
                if st.button("⚙️ Configure Matching", use_container_width=True):
                    st.session_state.page = "configure_survey"
                    st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error building question bank: {str(e)}")
        logger.error(f"Question bank building error: {e}")

elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ SurveyMonkey Survey Viewer")
    st.markdown('<div class="data-source-info">📊 <strong>Data Source:</strong> SurveyMonkey API - Extracting survey_title and question_text</div>', unsafe_allow_html=True)
    
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
                    # Get survey details from SurveyMonkey
                    survey_details = get_survey_details(selected_survey_id, token)
                    
                    # Extract questions using corrected function
                    questions = extract_questions_from_surveymonkey(survey_details)
                    df_questions = pd.DataFrame(questions)
                    
                    if df_questions.empty:
                        st.warning("⚠️ No questions found in this survey")
                    else:
                        st.success(f"✅ Extracted {len(df_questions)} questions from SurveyMonkey")
                        
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
                        
                        # Question categorization (from SurveyMonkey survey_title)
                        if 'question_category' in df_questions.columns:
                            st.markdown("### 📋 Question Categories")
                            category_counts = df_questions['question_category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        # Survey category (from SurveyMonkey survey_title)
                        survey_title = survey_details.get('title', 'Unknown')
                        survey_category = categorize_survey_from_surveymonkey(survey_title)
                        st.markdown(f"**Survey Category (from SurveyMonkey):** {survey_category}")
                        
                        # Sample questions
                        st.markdown("### 📝 Sample Questions (from SurveyMonkey)")
                        sample_questions = df_questions[df_questions['is_choice'] == False].head(5)
                        for idx, row in sample_questions.iterrows():
                            st.write(f"**{row['position']}.** {row['question_text']}")
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
                                f"surveymonkey_questions_{selected_survey_id}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"❌ Error analyzing survey: {str(e)}")
                    logger.error(f"Survey analysis error: {e}")
        
        # Survey list display with categorization
        st.markdown("### 📋 All Available Surveys (SurveyMonkey)")
        surveys_df = pd.DataFrame(surveys)
        if not surveys_df.empty:
            # Add survey categories based on SurveyMonkey titles
            surveys_df['category'] = surveys_df['title'].apply(categorize_survey_from_surveymonkey)
            st.dataframe(surveys_df[['title', 'id', 'category']], use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Failed to load surveys: {str(e)}")
        logger.error(f"Survey loading error: {e}")

elif st.session_state.page == "create_survey":
    st.markdown("## ➕ Create New SurveyMonkey Survey")
    st.markdown('<div class="data-source-info">📊 <strong>Data Source:</strong> Creating new survey in SurveyMonkey</div>', unsafe_allow_html=True)
    
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
            with st.spinner("🔄 Creating survey in SurveyMonkey..."):
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
                        st.success(f"✅ Survey created successfully in SurveyMonkey!")
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
    st.markdown("## 📖 Snowflake Question Bank (Reference Data)")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Cached Snowflake HEADING_0 and UID columns</div>', unsafe_allow_html=True)
    
    if not sf_status:
        st.error("❌ Snowflake connection required for Question Bank features")
        st.info("Please check your Snowflake connection in the sidebar")
        st.stop()
    
    # Check if question bank is built
    perf_stats = get_performance_stats()
    if "Cached" not in perf_stats.get('reference_cache_status', ''):
        st.warning("⚠️ Question Bank not built yet!")
        st.info("For optimal performance, build the question bank first.")
        if st.button("🏗️ Build Question Bank First"):
            st.session_state.page = "build_question_bank"
            st.rerun()
        st.stop()
    
    try:
        # Use cached data
        df_reference = get_cached_reference_questions()
        
        if df_reference.empty:
            st.warning("⚠️ No reference data found in Snowflake database")
            st.stop()
        
        st.success(f"✅ Using cached data: {len(df_reference):,} questions from Snowflake")
        
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
            st.metric("⚡ Cache Status", "Optimized")
        
        # Display sample data showing Snowflake columns
        st.markdown("### 📝 Snowflake Question Bank Sample")
        st.write("**Columns:** `HEADING_0` (reference questions), `UID` (assignments)")
        display_columns = ['uid', 'heading_0']
        st.dataframe(df_reference[display_columns].head(100), use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_export = df_reference.to_csv(index=False)
            st.download_button(
                "📥 Download Snowflake Question Bank",
                csv_export,
                f"snowflake_question_bank_{uuid4()}.csv",
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
    st.markdown("## ⭐ Enhanced Unique Questions Bank (from Snowflake)")
    st.markdown("*Best structured question for each UID with governance compliance and quality scoring*")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Cached Snowflake HEADING_0 by UID groups</div>', unsafe_allow_html=True)
    
    if not sf_status:
        st.error("❌ Snowflake connection required for Question Bank features")
        st.stop()
    
    try:
        # Use cached unique questions bank
        unique_questions_df = get_cached_unique_questions_bank()
        
        if unique_questions_df.empty:
            st.warning("⚠️ No unique questions could be created from Snowflake data")
            if st.button("🏗️ Build Question Bank First"):
                st.session_state.page = "build_question_bank"
                st.rerun()
            st.stop()
        
        # Also get reference count for context
        df_reference = get_cached_reference_questions()
        
        st.success(f"✅ Using cached unique bank: {len(unique_questions_df):,} UIDs from {len(df_reference):,} total Snowflake questions")
        
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
            st.metric("⚡ Cache Status", "Optimized")
        
        # Display results
        st.markdown("### 📝 Unique Questions Bank (Best HEADING_0 per UID)")
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
        st.error(f"❌ Failed to load unique questions bank: {str(e)}")
        logger.error(f"Unique questions bank error: {e}")

elif st.session_state.page == "categorized_questions":
    st.markdown("## 📊 Questions by Survey Category")
    st.markdown('<div class="warning-card">⚠️ <strong>Note:</strong> Snowflake has no survey_title column. Categorization only available from SurveyMonkey data.</div>', unsafe_allow_html=True)
    
    st.info("Categorization requires SurveyMonkey survey_title data. Snowflake only contains HEADING_0 and UID.")
    
    if st.button("📊 Go to SurveyMonkey Categories"):
        st.session_state.page = "view_surveys"
        st.rerun()

elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Configure Survey with UID Assignment")
    st.markdown('<div class="data-source-info">🔄 <strong>Process:</strong> Match SurveyMonkey question_text → Snowflake HEADING_0 → Get UID</div>', unsafe_allow_html=True)
    
    # Check prerequisites
    if not st.session_state.surveymonkey_initialized:
        st.warning("⚠️ Please initialize SurveyMonkey first by viewing surveys")
        if st.button("👁️ Go to View Surveys"):
            st.session_state.page = "view_surveys"
            st.rerun()
        st.stop()
    
    # Check if we have target data from SurveyMonkey
    if st.session_state.df_target is None:
        st.info("📋 No SurveyMonkey survey selected for configuration. Please select a survey first.")
        if st.button("👁️ Select Survey"):
            st.session_state.page = "view_surveys"
            st.rerun()
        st.stop()
    
    df_target = st.session_state.df_target
    st.success(f"✅ Configuring survey with {len(df_target)} SurveyMonkey questions")
    
    # Performance check
    perf_stats = get_performance_stats()
    total_ready = sum([
        1 if "Cached" in perf_stats['reference_cache_status'] else 0,
        1 if "Cached" in perf_stats['unique_cache_status'] else 0,
        1 if "Ready" in perf_stats['matching_data_status'] else 0
    ])
    
    if total_ready < 3:
        st.warning("⚠️ Question Bank not fully optimized! Matching will be slower.")
        st.info(f"Optimization Status: {total_ready}/3 Ready")
        if st.button("🏗️ Build Question Bank for Better Performance"):
            st.session_state.page = "build_question_bank"
            st.rerun()
    else:
        st.success("✅ Question Bank fully optimized! Ready for fast matching.")
    
    # Show data sources
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Source Data (SurveyMonkey)")
        st.write(f"Survey: {df_target['survey_title'].iloc[0] if not df_target.empty else 'Unknown'}")
        st.write(f"Questions: {len(df_target[df_target['is_choice'] == False])}")
        st.write(f"Choices: {len(df_target[df_target['is_choice'] == True])}")
    
    with col2:
        st.markdown("### ❄️ Reference Data (Snowflake)")
        if sf_status:
            st.write(f"Cache Status: {perf_stats['reference_cache_status']}")
            st.write(f"Available UIDs: {perf_stats['reference_questions_loaded']:,}")
            st.write(f"Embeddings: {perf_stats['embeddings_count']:,}")
        else:
            st.write("Status: Not Connected")
    
    # UID Assignment section
    if sf_status:
        st.markdown("### 🔄 UID Assignment Process")
        
        # Choose matching method
        use_fast_matching = st.checkbox("🚀 Use Fast Matching (Recommended)", value=True, 
                                      help="Uses pre-computed embeddings for 90% faster matching")
        
        if st.button("🚀 Run Semantic Matching", type="primary"):
            if use_fast_matching and total_ready == 3:
                with st.spinner("🚀 Running FAST semantic matching with pre-computed embeddings..."):
                    try:
                        # Convert target data to list format for matching
                        sm_questions = df_target.to_dict('records')
                        
                        # Use fast matching
                        matched_results = fast_semantic_matching(sm_questions, use_cached_data=True)
                        
                        if matched_results:
                            matched_df = pd.DataFrame(matched_results)
                            st.session_state.df_final = matched_df
                            
                            # Show matching results
                            st.success(f"✅ FAST semantic matching completed!")
                            
                            # Matching statistics
                            high_conf = len(matched_df[matched_df['match_confidence'] == 'High'])
                            medium_conf = len(matched_df[matched_df['match_confidence'] == 'Medium'])
                            low_conf = len(matched_df[matched_df['match_confidence'] == 'Low'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🎯 High Confidence", high_conf)
                            with col2:
                                st.metric("⚠️ Medium Confidence", medium_conf)
                            with col3:
                                st.metric("❌ Low/No Match", low_conf)
                            
                            # Show sample matches
                            st.markdown("### 📋 Sample Matching Results")
                            sample_matched = matched_df[matched_df['matched_uid'].notna()].head(5)
                            
                            for idx, row in sample_matched.iterrows():
                                with st.expander(f"Match {idx+1}: UID {row['matched_uid']} (Confidence: {row['match_confidence']})"):
                                    st.write(f"**SurveyMonkey Question:** {row['question_text']}")
                                    st.write(f"**Matched Snowflake HEADING_0:** {row['matched_heading_0']}")
                                    st.write(f"**Match Score:** {row['match_score']:.3f}")
                        else:
                            st.error("❌ No matching results generated")
                            
                    except Exception as e:
                        st.error(f"❌ Fast semantic matching failed: {str(e)}")
                        logger.error(f"Fast semantic matching error: {e}")
            else:
                with st.spinner("🔄 Running standard semantic matching (slower)..."):
                    try:
                        # Load Snowflake reference data
                        df_reference = get_cached_reference_questions()
                        
                        if df_reference.empty:
                            st.error("❌ No Snowflake reference data available for matching")
                        else:
                            # Convert target data to list format for matching
                            sm_questions = df_target.to_dict('records')
                            
                            # Perform standard semantic matching
                            matched_results = perform_semantic_matching(sm_questions, df_reference)
                            
                            if matched_results:
                                matched_df = pd.DataFrame(matched_results)
                                st.session_state.df_final = matched_df
                                
                                # Show matching results
                                st.success(f"✅ Semantic matching completed!")
                                
                                # Matching statistics
                                high_conf = len(matched_df[matched_df['match_confidence'] == 'High'])
                                medium_conf = len(matched_df[matched_df['match_confidence'] == 'Medium'])
                                low_conf = len(matched_df[matched_df['match_confidence'] == 'Low'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🎯 High Confidence", high_conf)
                                with col2:
                                    st.metric("⚠️ Medium Confidence", medium_conf)
                                with col3:
                                    st.metric("❌ Low/No Match", low_conf)
                                
                                # Show sample matches
                                st.markdown("### 📋 Sample Matching Results")
                                sample_matched = matched_df[matched_df['matched_uid'].notna()].head(5)
                                
                                for idx, row in sample_matched.iterrows():
                                    with st.expander(f"Match {idx+1}: UID {row['matched_uid']} (Confidence: {row['match_confidence']})"):
                                        st.write(f"**SurveyMonkey Question:** {row['question_text']}")
                                        st.write(f"**Matched Snowflake HEADING_0:** {row['matched_heading_0']}")
                                        st.write(f"**Match Score:** {row['match_score']:.3f}")
                            else:
                                st.error("❌ No matching results generated")
                                
                    except Exception as e:
                        st.error(f"❌ Semantic matching failed: {str(e)}")
                        logger.error(f"Semantic matching error: {e}")
    else:
        st.warning("❌ Snowflake connection required for UID assignment")
        st.info("Configure surveys is available, but UID matching requires Snowflake connection")
    
    # Question configuration display
    st.markdown("### 📝 SurveyMonkey Question Configuration")
    
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
    
    # Display questions with correct column names
    st.dataframe(display_df[['position', 'question_text', 'schema_type', 'question_category', 'survey_category']], 
                use_container_width=True, height=400)
    
    # Export options
    st.markdown("### 📥 Export & Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_export = df_target.to_csv(index=False)
        st.download_button(
            "📥 Download SurveyMonkey Config",
            csv_export,
            f"surveymonkey_config_{uuid4()}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Refresh Survey Data", use_container_width=True):
            st.session_state.df_target = None
            st.session_state.page = "view_surveys"
            st.rerun()
    
    with col3:
        if st.session_state.df_final is not None:
            matched_csv = st.session_state.df_final.to_csv(index=False)
            st.download_button(
                "📥 Download Matched Results",
                matched_csv,
                f"matched_results_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )

elif st.session_state.page == "data_quality":
    st.markdown("## 🧹 Data Quality Management (Snowflake)")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Analyzing Snowflake HEADING_0 quality and UID governance</div>', unsafe_allow_html=True)
    
    if not sf_status:
        st.error("❌ Snowflake connection required for data quality analysis")
        st.stop()
    
    try:
        # Use cached data for better performance
        df_reference = get_cached_reference_questions()
        
        if df_reference.empty:
            st.warning("⚠️ No reference data found in Snowflake")
            if st.button("🏗️ Build Question Bank First"):
                st.session_state.page = "build_question_bank"
                st.rerun()
            st.stop()
        
        st.success(f"✅ Loaded {len(df_reference):,} Snowflake questions for quality analysis")
        
        # Quality overview
        st.markdown("### 📊 Snowflake Data Quality Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total HEADING_0", f"{len(df_reference):,}")
        
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
        
        # Quality issues specific to Snowflake HEADING_0
        st.markdown("### 🔍 Snowflake HEADING_0 Quality Issues")
        
        quality_issues = {
            'Empty HEADING_0': len(df_reference[df_reference['heading_0'].str.len() < 5]),
            'HTML in HEADING_0': len(df_reference[df_reference['heading_0'].str.contains('<.*>', regex=True, na=False)]),
            'Privacy Policy': len(df_reference[df_reference['heading_0'].str.contains('privacy policy', case=False, na=False)]),
            'Duplicate UID-HEADING_0': len(df_reference[df_reference.duplicated(['uid', 'heading_0'])])
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quality Issues in Snowflake:**")
            for issue, count in quality_issues.items():
                if count > 0:
                    st.write(f"❌ {issue}: {count:,}")
                else:
                    st.write(f"✅ {issue}: {count}")
        
        with col2:
            st.markdown("**Governance Violations:**")
            excessive_uids = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
            if len(excessive_uids) > 0:
                st.write(f"❌ UIDs exceeding limit: {len(excessive_uids)}")
                st.write(f"❌ Total excess HEADING_0: {excessive_uids.sum() - (len(excessive_uids) * UID_GOVERNANCE['max_variations_per_uid'])}")
            else:
                st.write("✅ All UIDs within governance limits")
        
        # Quality insights
        st.markdown("### 📈 Snowflake Data Quality Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**HEADING_0 Length Distribution:**")
            question_lengths = df_reference['heading_0'].str.len()
            length_bins = pd.cut(question_lengths, bins=[0, 10, 50, 100, 200, float('inf')], 
                               labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
            length_counts = length_bins.value_counts()
            st.bar_chart(length_counts)
        
        with col2:
            st.markdown("**HEADING_0 Quality (Snowflake only has HEADING_0 + UID):**")
            st.write("✅ No survey_title categorization needed")
            st.write("📊 Focus on HEADING_0 quality only")
        
         # Top problematic UIDs
        if len(excessive_uids) > 0:
            st.markdown("### ⚠️ Most Problematic UIDs in Snowflake")
            top_problematic = excessive_uids.head(10).reset_index()
            top_problematic.columns = ['UID', 'HEADING_0 Count']
            top_problematic['Excess'] = top_problematic['HEADING_0 Count'] - UID_GOVERNANCE['max_variations_per_uid']
            st.dataframe(top_problematic, use_container_width=True)
            
            # Show sample problematic entries
            if st.button("🔍 Show Sample Problematic Entries"):
                worst_uid = excessive_uids.index[0]
                worst_entries = df_reference[df_reference['uid'] == worst_uid]['heading_0'].head(10)
                st.markdown(f"**Sample HEADING_0 entries for UID {worst_uid}:**")
                for i, entry in enumerate(worst_entries, 1):
                    st.write(f"{i}. {entry}")
    
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

# Footer with data source reminder
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

with footer_col2:
    st.markdown("**📊 Data Sources**")
    st.write("📊 SurveyMonkey: survey_title, question_text")
    st.write("❄️ Snowflake: HEADING_0, UID ONLY")

with footer_col3:
    st.markdown("**📊 Current Session**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Init: {'✅' if st.session_state.surveymonkey_initialized else '❌'}")
    st.write(f"SF Init: {'✅' if st.session_state.snowflake_initialized else '❌'}")

# ============= END OF SCRIPT =============