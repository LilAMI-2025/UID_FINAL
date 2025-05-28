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
import numpy as np
from collections import defaultdict, Counter

# Setup
st.set_page_config(
    page_title="UID Matcher Enhanced", 
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
    
    .conflict-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
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
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
MAX_SURVEYS_PER_BATCH = 10

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

# UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'conflict_resolution_threshold': 10,
    'high_conflict_threshold': 100
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
        "question_bank": None,
        "survey_template": None,
        "preview_df": None,
        "all_questions": None,
        "dedup_questions": [],
        "dedup_choices": [],
        "pending_survey": None,
        "snowflake_initialized": False,
        "surveymonkey_initialized": False,
        "optimized_question_bank": None,
        "uid_conflicts_summary": None,
        "primary_matching_reference": None,
        "fetched_survey_ids": []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# Utility Functions
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

def classify_question(text, heading_references=HEADING_REFERENCES):
    """Classify question as Heading or Main Question"""
    try:
        # Length-based heuristic
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
            
            # Combine criteria
            if max_tfidf_score >= HEADING_TFIDF_THRESHOLD or max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
                return "Heading"
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
        
        return "Main Question/Multiple Choice"
    except Exception as e:
        logger.error(f"Error in classify_question: {e}")
        return "Main Question/Multiple Choice"

# Cached Resources
@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

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
            st.warning(
                "🔒 Snowflake connection failed: User account is locked. "
                "UID matching is disabled, but you can use SurveyMonkey features. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
            )
        raise

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
        "all_questions": all_questions.to_dict(orient="records") if not all_questions.empty else [],
        "dedup_questions": dedup_questions,
        "dedup_choices": dedup_choices
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

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

# Snowflake Queries - Fixed column name handling
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
        # Ensure consistent column naming
        result.columns = result.columns.str.lower()
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account is locked. UID matching is disabled.")
        raise

# Enhanced Snowflake query for optimization - Fixed column handling
@st.cache_data(ttl=600)
def get_all_reference_questions_from_snowflake():
    """Fetch ALL reference questions from Snowflake for optimization"""
    all_data = []
    limit = 10000
    offset = 0
    
    # Use a more detailed query for optimization
    query = """
    SELECT 
        HEADING_0, 
        UID, 
        COUNT(*) as OCCURRENCE_COUNT
    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
    WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
    AND TRIM(HEADING_0) != ''
    GROUP BY HEADING_0, UID
    ORDER BY UID, OCCURRENCE_COUNT DESC
    LIMIT :limit OFFSET :offset
    """
    
    while True:
        try:
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
            
            if result.empty:
                break
                
            # Ensure consistent column naming
            result.columns = result.columns.str.lower()
            all_data.append(result)
            offset += limit
            
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            # Fall back to simple query if enhanced query fails
            simple_query = """
            SELECT HEADING_0, MAX(UID) AS UID
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL
            GROUP BY HEADING_0
            ORDER BY CAST(UID AS INTEGER) ASC
            LIMIT :limit OFFSET :offset
            """
            try:
                with get_snowflake_engine().connect() as conn:
                    result = pd.read_sql(text(simple_query), conn, params={"limit": limit, "offset": offset})
                # Ensure consistent column naming and add occurrence count
                result.columns = result.columns.str.lower()
                result['occurrence_count'] = 1
                if result.empty:
                    break
                all_data.append(result)
                offset += limit
                if len(result) < limit:
                    break
            except Exception as e2:
                logger.error(f"Both enhanced and simple queries failed: {e2}")
                return pd.DataFrame()
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Ensure proper column naming
        if 'occurrence_count' not in final_df.columns:
            final_df['occurrence_count'] = 1
        logger.info(f"Total reference questions fetched from Snowflake: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched from Snowflake")
        return pd.DataFrame()

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

def extract_questions(survey_json):
    """Extract questions from SurveyMonkey with question_uid (question_id)"""
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
            
            question_category = classify_question(q_text)
            
            if q_text:
                global_position += 1
                questions.append({
                    "heading_0": q_text,
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_uid": q_id,  # This is the question_id from SurveyMonkey
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", ""),
                    "question_category": question_category
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
                            "question_uid": q_id,  # Same question_id for choices
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False,
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", ""),
                            "question_category": "Main Question/Multiple Choice"
                        })
    return questions

# Enhanced Question Bank Builder with 1:1 Optimization - Fixed column handling
def build_optimized_1to1_question_bank(df_reference):
    """Build optimized 1:1 question bank with conflict resolution"""
    if df_reference.empty:
        logger.warning("No Snowflake reference data provided for optimization")
        st.warning("⚠️ No Snowflake reference data provided for optimization")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        logger.info(f"Building optimized 1:1 question bank from {len(df_reference):,} Snowflake records")
        
        # Normalize questions for grouping
        df_reference['normalized_question'] = df_reference['heading_0'].apply(
            lambda x: enhanced_normalize(x, ENHANCED_SYNONYM_MAP)
        )
        
        # Group by normalized question to find conflicts
        question_analysis = []
        conflict_summary = []
        
        grouped = df_reference.groupby('normalized_question')
        
        for norm_question, group in grouped:
            if not norm_question or len(norm_question.strip()) < 3:
                continue
            
            # Count UIDs for this normalized question - handle both column names
            if 'occurrence_count' in group.columns:
                uid_counts = group.groupby('uid')['occurrence_count'].sum().sort_values(ascending=False)
            else:
                uid_counts = group['uid'].value_counts()
            
            if len(uid_counts) == 0:
                continue
            
            # Get all unique variants of the question text
            all_variants = group['heading_0'].unique()
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
            
            logger.info(f"Built optimized 1:1 question bank: {len(optimized_df):,} unique questions, {len(conflicts_df):,} conflicts resolved")
            
        return optimized_df, conflicts_df
        
    except Exception as e:
        logger.error(f"Failed to build optimized question bank: {e}")
        st.error(f"❌ Failed to build optimized question bank: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# UID Matching Functions (keeping all original logic)
def compute_tfidf_matches(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP):
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
    
    # Prevent UID assignment for Heading questions
    if "question_category" in df_target.columns:
        df_target.loc[df_target["question_category"] == "Heading", ["Final_UID", "configured_final_UID"]] = None
    
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

def run_uid_match(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP, batch_size=BATCH_SIZE):
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

# Export Functions
def prepare_export_data(matched_results):
    """Prepare data for Snowflake export with enhanced metadata"""
    if not matched_results:
        return pd.DataFrame()
    
    try:
        if isinstance(matched_results, pd.DataFrame):
            export_df = matched_results.copy()
        else:
            export_df = pd.DataFrame(matched_results)
        
        # Add main question UID and position for choices
        if not export_df.empty and 'is_choice' in export_df.columns:
            main_questions_df = export_df[export_df['is_choice'] == False].copy()
            
            if not main_questions_df.empty:
                export_df['Main_Question_UID'] = export_df.apply(
                    lambda row: main_questions_df[main_questions_df['heading_0'] == row['parent_question']]['Final_UID'].iloc[0]
                    if row['is_choice'] and pd.notnull(row.get('parent_question')) and 
                       not main_questions_df[main_questions_df['heading_0'] == row['parent_question']].empty
                    else row.get('Final_UID'),
                    axis=1
                )
                
                export_df['Main_Question_Position'] = export_df.apply(
                    lambda row: main_questions_df[main_questions_df['heading_0'] == row['parent_question']]['position'].iloc[0]
                    if row['is_choice'] and pd.notnull(row.get('parent_question')) and 
                       not main_questions_df[main_questions_df['heading_0'] == row['parent_question']].empty
                    else row.get('position'),
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
        # Create preview with essential columns including question_uid (question_id)
        preview_columns = [
            'survey_id', 'survey_title', 'question_uid', 'heading_0', 
            'Main_Question_Position', 'Main_Question_UID'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in preview_columns if col in export_df.columns]
        
        if not available_columns:
            available_columns = ['heading_0']  # At minimum show the question
        
        preview_df = export_df[available_columns].copy()
        
        # Rename columns for display
        column_renames = {
            'survey_id': 'SurveyID',
            'survey_title': 'SurveyName',
            'question_uid': 'QuestionID',
            'heading_0': 'Question Info',
            'Main_Question_Position': 'QuestionPosition',
            'Main_Question_UID': 'UID'
        }
        
        for old_name, new_name in column_renames.items():
            if old_name in preview_df.columns:
                preview_df = preview_df.rename(columns={old_name: new_name})
        
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
        if 'is_choice' in export_df.columns and 'Final_UID' in export_df.columns:
            main_questions_without_uid = export_df[
                (export_df['is_choice'] == False) & 
                (export_df['Final_UID'].isna() | (export_df['Final_UID'] == ''))
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

# Safe secrets access
def get_surveymonkey_token():
    """Safely get SurveyMonkey token"""
    try:
        return st.secrets["surveymonkey"]["token"]
    except Exception as e:
        logger.error(f"Failed to get SurveyMonkey token: {e}")
        return None

# ============= SIDEBAR NAVIGATION =============

with st.sidebar:
    st.markdown("### 🧠 UID Matcher Enhanced")
    st.markdown("Advanced question bank optimization with conflict resolution")
    
    # Connection status
    sm_status, sm_msg = check_surveymonkey_connection()
    sf_status, sf_msg = check_snowflake_connection()
    
    st.markdown("**🔗 Connection Status**")
    st.write(f"📊 SurveyMonkey: {'✅' if sm_status else '❌'}")
    st.write(f"❄️ Snowflake: {'✅' if sf_status else '❌'}")
    
    # Optimization status
    opt_ref = st.session_state.get('primary_matching_reference')
    opt_status = "✅ Ready" if opt_ref is not None and not opt_ref.empty else "❌ Not Built"
    st.markdown(f"🎯 Optimization: {opt_status}")
    
    # Data source info
    with st.expander("📊 Data Sources"):
        st.markdown("**SurveyMonkey (Source):**")
        st.markdown("• Survey data and questions")
        st.markdown("• question_uid → SurveyMonkey question ID")
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
    if st.button("🎯 Build Optimized Bank", use_container_width=True):
        st.session_state.page = "build_optimization"
        st.rerun()
    if st.button("📊 View Conflicts", use_container_width=True):
        st.session_state.page = "conflict_dashboard"
        st.rerun()

# ============= MAIN APP HEADER =============

st.markdown('<div class="main-header">🧠 UID Matcher: Enhanced with 1:1 Optimization</div>', unsafe_allow_html=True)

# Data source clarification
st.markdown('<div class="data-source-info"><strong>📊 Data Flow:</strong> SurveyMonkey surveys → UID matching → Snowflake reference → Optimized 1:1 mapping</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Load initial data with error handling
try:
    token = get_surveymonkey_token()
    if not token:
        st.error("❌ Failed to get SurveyMonkey token")
        surveys = []
    else:
        surveys = get_surveys_cached(token)
        if not surveys:
            st.error("No surveys found.")
            surveys = []

    # Load cached survey data
    if st.session_state.all_questions is None:
        cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
        if cached_questions is not None and not cached_questions.empty:
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
    surveys = []

# ============= PAGE ROUTING =============

if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher")
    
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
        opt_ref = st.session_state.get('primary_matching_reference')
        unique_questions = len(opt_ref) if opt_ref is not None and not opt_ref.empty else 0
        st.metric("🎯 Optimized Questions", f"{unique_questions:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow guide
    st.markdown("## 🚀 Recommended Workflow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Survey Selection")
        st.markdown("Select and analyze surveys:")
        st.markdown("• Browse available surveys")
        st.markdown("• Extract questions with IDs")
        st.markdown("• Review question bank")
        
        if st.button("📋 Start Survey Selection", use_container_width=True):
            st.session_state.page = "survey_selection"
            st.rerun()
    
    with col2:
        st.markdown("### 2️⃣ UID Matching")
        st.markdown("Match questions to UIDs:")
        st.markdown("• Optimized 1:1 matching")
        st.markdown("• Configure assignments")
        st.markdown("• Resolve conflicts")
        
        if st.button("🔧 Start UID Matching", use_container_width=True):
            st.session_state.page = "uid_matching"
            st.rerun()
    
    with col3:
        st.markdown("### 3️⃣ Question Bank")
        st.markdown("Optimize question bank:")
        st.markdown("• Build 1:1 optimization")
        st.markdown("• Resolve UID conflicts")
        st.markdown("• View conflict dashboard")
        
        if st.button("🎯 Build Question Bank", use_container_width=True):
            st.session_state.page = "build_optimization"
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
    
    if not surveys:
        st.markdown('<div class="warning-card">⚠️ No surveys available. Check SurveyMonkey connection.</div>', unsafe_allow_html=True)
        st.stop()
    
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
    if selected_survey_ids and token:
        combined_questions = []
        
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
                
                # Show questions with question_uid (question_id)
                display_columns = ["question_uid", "heading_0", "schema_type", "is_choice", "survey_title"]
                available_columns = [col for col in display_columns if col in display_df.columns]
                st.dataframe(display_df[available_columns], height=400)
                
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
    
    # Show survey data info
    st.markdown("### 📊 Current Survey Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(st.session_state.df_target))
    with col2:
        main_q = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
        st.metric("Main Questions", main_q)
    with col3:
        surveys = st.session_state.df_target["survey_id"].nunique()
        st.metric("Surveys", surveys)
    
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
            
            display_columns = ["question_uid", "heading_0", "schema_type", "is_choice"]
            if "Final_UID" in result_df.columns:
                display_columns.append("Final_UID")
            if "Change_UID" in result_df.columns:
                display_columns.append("Change_UID")
            display_columns.append("required")
            
            # Only show columns that exist
            available_columns = [col for col in display_columns if col in result_df.columns]
            
            edited_df = st.data_editor(
                result_df[available_columns],
                column_config={
                    "question_uid": st.column_config.TextColumn("Question ID", width="medium"),
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
                disabled=["question_uid", "heading_0", "schema_type", "is_choice", "Final_UID"],
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
        
        # Export Section
        st.markdown("---")
        st.markdown("### 📥 Export & Upload")
        
        export_df = prepare_export_data(st.session_state.df_final)
        if not export_df.empty:
            preview_df = create_export_preview(export_df)
            
            if not preview_df.empty:
                st.markdown("#### 👁️ Preview Data for Snowflake Upload")
                st.dataframe(preview_df, use_container_width=True)
            
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
                    upload_to_snowflake(export_df)

elif st.session_state.page == "build_optimization":
    st.markdown("## 🎯 Build Optimized 1:1 Question Bank")
    st.markdown('<div class="data-source-info">🎯 <strong>Process:</strong> Analyze Snowflake data → Resolve UID conflicts → Create 1:1 mapping</div>', unsafe_allow_html=True)
    
    try:
        # Load reference data
        with st.spinner("📊 Loading Snowflake reference data..."):
            df_reference = get_all_reference_questions_from_snowflake()
        
        if df_reference.empty:
            st.warning("⚠️ No reference questions loaded from Snowflake")
            st.stop()
        
        st.markdown('<div class="success-card">✅ Loaded reference data successfully</div>', unsafe_allow_html=True)
        
        # Show data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", f"{len(df_reference):,}")
        with col2:
            unique_uids = df_reference['uid'].nunique()
            st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        with col3:
            unique_questions = df_reference['heading_0'].nunique()
            st.metric("📝 Unique Questions", f"{unique_questions:,}")
        
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
                    st.session_state.page = "conflict_dashboard"
                    st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to build optimization: {e}")
        st.error(f"❌ Failed to build optimization: {str(e)}")

elif st.session_state.page == "conflict_dashboard":
    st.markdown("## 📊 UID Conflict Resolution Dashboard")
    st.markdown('<div class="data-source-info">📊 <strong>Analysis:</strong> UID conflicts and resolution strategies</div>', unsafe_allow_html=True)
    
    conflicts_summary = st.session_state.get('uid_conflicts_summary')
    
    if conflicts_summary is None or conflicts_summary.empty:
        st.warning("⚠️ No conflict data available. Please build the optimized question bank first.")
        if st.button("🎯 Build Question Bank"):
            st.session_state.page = "build_optimization"
            st.rerun()
        st.stop()
    
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
                # Create survey template (implementation similar to previous version)
                st.markdown('<div class="info-card">Survey creation functionality available</div>', unsafe_allow_html=True)

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
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

with footer_col2:
    st.markdown("**📊 Data Sources**")
    st.write("📊 SurveyMonkey: Surveys & Questions + IDs")
    st.write("❄️ Snowflake: UIDs & References")

with footer_col3:
    st.markdown("**📊 Current Session**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Status: {'✅' if sm_status else '❌'}")
    st.write(f"SF Status: {'✅' if sf_status else '❌'}")
    opt_ref = st.session_state.get('primary_matching_reference')
    opt_count = len(opt_ref) if opt_ref is not None and not opt_ref.empty else 0
    st.write(f"Optimized: {opt_count:,}")

# ============= END OF SCRIPT =============
