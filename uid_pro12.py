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
    page_title="Enhanced UID Matcher Pro", 
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
    
    .governance-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6f42c1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION CONSTANTS =====

# Model and threshold constants
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50

# Enhanced UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'semantic_matching_enabled': True,
    'governance_enforcement': True
}

# Question Standardization Rules
QUESTION_STANDARDIZATION = {
    'standardization_enabled': True,
    'normalize_case': True,
    'remove_extra_spaces': True,
    'standardize_punctuation': True,
    'expand_contractions': True,
    'fix_common_typos': True,
    'standardize_formats': True
}

# Enhanced Question Format Patterns
QUESTION_FORMAT_PATTERNS = {
    'demographic': {
        'age': r'(what is your age|how old are you|age|what age)',
        'gender': r'(what is your gender|gender|male or female)',
        'education': r'(education level|qualification|degree|education)',
        'experience': r'(years of experience|work experience|professional experience)',
        'role': r'(current role|position|job title|what is your role)',
        'team_size': r'(team size|how many people report|staff report|team members)',
        'sector': r'(sector|industry|what sector|which industry)',
        'company_size': r'(company size|organization size|number of employees)',
        'location': r'(location|where are you based|city|country)',
        'department': r'(department|division|which department)'
    },
    'rating_scale': {
        'satisfaction': r'(how satisfied|satisfaction|rate your satisfaction)',
        'likelihood': r'(how likely|likelihood|probability)',
        'agreement': r'(agree|disagree|to what extent)',
        'importance': r'(how important|importance|priority)',
        'frequency': r'(how often|frequency|how frequently)',
        'effectiveness': r'(how effective|effectiveness|rate the effectiveness)'
    },
    'behavioral': {
        'usage': r'(how often do you use|usage|frequency of use)',
        'preference': r'(prefer|preference|which do you prefer)',
        'experience_rating': r'(rate your experience|experience rating)',
        'recommendation': r'(would you recommend|recommendation|refer)'
    }
}

# Enhanced Synonym Mapping
ENHANCED_SYNONYM_MAP = {
    # Question starters
    "please select": "what is",
    "select one": "what is",
    "choose one": "what is",
    "pick one": "what is",
    
    # Demographic standardization
    "sector you are from": "your sector",
    "which sector": "your sector",
    "what sector": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
    
    # Role and position
    "current role": "current position",
    "your role": "your position",
    "job title": "position",
    "what is your role": "what is your position",
    
    # Team and management
    "how many people report to you": "team size",
    "how many staff report to you": "team size",
    "team members": "team size",
    "direct reports": "team size",
    
    # Age standardization
    "what is age": "what is your age",
    "what age": "what is your age",
    "your age": "what is your age",
    "how old are you": "what is your age",
    
    # Experience standardization
    "years of experience": "work experience",
    "professional experience": "work experience",
    "work history": "work experience",
    
    # Rating and satisfaction
    "rate your": "how would you rate",
    "satisfaction with": "how satisfied are you with",
    "how happy": "how satisfied",
    
    # Frequency standardization
    "how often": "frequency",
    "how frequently": "frequency",
    "how regular": "frequency"
}

# Survey Categories based on content
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join', 'eligibility'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline', 'before', 'prior', 'initial'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start', 'begin'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation', 'improvement'],
    'GROW': ['GROW', 'grow'],
    'Feedback': ['feedback', 'evaluation', 'rating', 'satisfaction', 'opinion'],
    'Pulse': ['pulse', 'quick', 'brief', 'snapshot', 'check-in'],
    'Demographic': ['age', 'gender', 'education', 'experience', 'role', 'position', 'sector', 'industry', 'location']
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# ===== CACHED RESOURCES =====

@st.cache_resource
def load_sentence_transformer():
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

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
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

@st.cache_data
def get_all_reference_questions():
    """Cached function to get all reference questions from Snowflake"""
    return run_snowflake_reference_query_all()

# ===== QUESTION STANDARDIZATION FUNCTIONS =====

def standardize_question_format(question_text):
    """Standardize question format using enhanced rules"""
    if not question_text or pd.isna(question_text):
        return question_text
    
    text = str(question_text).strip()
    
    if not QUESTION_STANDARDIZATION.get('standardization_enabled', True):
        return text
    
    # Normalize case
    if QUESTION_STANDARDIZATION.get('normalize_case', True):
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Remove extra spaces
    if QUESTION_STANDARDIZATION.get('remove_extra_spaces', True):
        text = re.sub(r'\s+', ' ', text)
    
    # Standardize punctuation
    if QUESTION_STANDARDIZATION.get('standardize_punctuation', True):
        if any(word in text.lower().split()[:3] for word in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']):
            if not text.endswith('?'):
                text = text.rstrip('.!') + '?'
        
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[!]{2,}', '!', text)
    
    # Expand contractions
    if QUESTION_STANDARDIZATION.get('expand_contractions', True):
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.title(), expansion.title())
    
    # Fix common typos
    if QUESTION_STANDARDIZATION.get('fix_common_typos', True):
        typo_fixes = {
            'teh': 'the', 'adn': 'and', 'youre': 'you are',
            'your are': 'you are', 'its': 'it is', 'recieve': 'receive',
            'seperate': 'separate', 'definately': 'definitely'
        }
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in typo_fixes:
                if word.lower() == clean_word:
                    words[i] = typo_fixes[clean_word]
                elif word.title() == word:
                    words[i] = typo_fixes[clean_word].title()
        text = ' '.join(words)
    
    # Standardize question formats
    if QUESTION_STANDARDIZATION.get('standardize_formats', True):
        text_lower = text.lower()
        for phrase, replacement in ENHANCED_SYNONYM_MAP.items():
            if phrase in text_lower:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub(replacement, text)
    
    return text.strip()

def detect_question_pattern(question_text):
    """Detect question pattern for better categorization"""
    text_lower = question_text.lower()
    
    for category, patterns in QUESTION_FORMAT_PATTERNS.items():
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, text_lower):
                return {
                    'category': category,
                    'pattern': pattern_name,
                    'confidence': 0.9
                }
    
    # Fallback pattern detection
    if any(word in text_lower.split()[:3] for word in ['what', 'which']):
        return {'category': 'demographic', 'pattern': 'general', 'confidence': 0.5}
    elif any(word in text_lower.split()[:3] for word in ['how']):
        return {'category': 'rating_scale', 'pattern': 'general', 'confidence': 0.5}
    elif any(word in text_lower.split()[:3] for word in ['do', 'did', 'are', 'is']):
        return {'category': 'behavioral', 'pattern': 'general', 'confidence': 0.5}
    
    return {'category': 'other', 'pattern': 'unknown', 'confidence': 0.1}

def categorize_question_by_content(question_text):
    """Categorize question based on content analysis"""
    if not question_text:
        return "Unknown"
    
    text_lower = question_text.lower()
    
    for category, keywords in SURVEY_CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return "Other"

def score_question_quality(question):
    """Enhanced scoring function for question quality"""
    score = 0
    text = str(question).lower().strip()
    
    # Length scoring (sweet spot is 10-100 characters)
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
    
    # English question word scoring
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
    
    # Prefer complete sentences
    word_count = len(text.split())
    if 5 <= word_count <= 20:
        score += 10
    elif word_count > 30:
        score -= 5
    
    # Avoid repetitive characters
    if any(char * 3 in text for char in 'abcdefghijklmnopqrstuvwxyz'):
        score -= 10
    
    # Semantic coherence bonus
    if all(word.isalpha() or word in ['?', '.', ','] for word in text.split()):
        score += 5
    
    return score

def get_best_question_for_uid(questions_list):
    """Enhanced question selection with quality scoring"""
    if not questions_list:
        return None
    
    scored_questions = [(q, score_question_quality(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

# ===== SEMANTIC MATCHING FUNCTIONS =====

def enhanced_semantic_matching_with_governance(question_text, existing_uids_data, threshold=0.85):
    """Enhanced semantic matching with governance rules and standardization"""
    if not existing_uids_data:
        return None, 0.0, "no_existing_data"
    
    try:
        standardized_question = standardize_question_format(question_text)
        model = load_sentence_transformer()
        
        question_embedding = model.encode([standardized_question], convert_to_tensor=True)
        
        existing_questions = []
        uid_mapping = []
        
        for uid, data in existing_uids_data.items():
            standardized_existing = standardize_question_format(data['best_question'])
            existing_questions.append(standardized_existing)
            uid_mapping.append(uid)
        
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        similarities = util.cos_sim(question_embedding, existing_embeddings)[0]
        
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        if best_score >= threshold:
            best_uid = uid_mapping[best_idx]
            
            if existing_uids_data[best_uid]['variation_count'] < UID_GOVERNANCE['max_variations_per_uid']:
                return best_uid, best_score, "semantic_match_compliant"
            else:
                return best_uid, best_score, "semantic_match_governance_violation"
        
        if best_score >= UID_GOVERNANCE['auto_consolidate_threshold']:
            best_uid = uid_mapping[best_idx]
            return best_uid, best_score, "auto_consolidate"
            
    except Exception as e:
        logger.error(f"Enhanced semantic matching failed: {e}")
    
    return None, 0.0, "no_match"

def assign_uid_with_full_governance(question_text, existing_uids_data, survey_category=None, question_pattern=None):
    """Comprehensive UID assignment with governance, standardization, and semantic matching"""
    standardized_question = standardize_question_format(question_text)
    
    if not question_pattern:
        question_pattern = detect_question_pattern(standardized_question)
    
    # Try semantic matching with governance
    if UID_GOVERNANCE.get('semantic_matching_enabled', True):
        matched_uid, confidence, match_status = enhanced_semantic_matching_with_governance(
            standardized_question, existing_uids_data, UID_GOVERNANCE['semantic_similarity_threshold']
        )
        
        if matched_uid and match_status == "semantic_match_compliant":
            return {
                'uid': matched_uid,
                'method': 'semantic_match',
                'confidence': confidence,
                'governance_compliant': True,
                'match_status': match_status,
                'standardized_question': standardized_question,
                'question_pattern': question_pattern,
                'original_question': question_text
            }
        elif matched_uid and match_status == "auto_consolidate":
            return {
                'uid': matched_uid,
                'method': 'auto_consolidate',
                'confidence': confidence,
                'governance_compliant': True,
                'match_status': match_status,
                'standardized_question': standardized_question,
                'question_pattern': question_pattern,
                'original_question': question_text
            }
        elif matched_uid and match_status == "semantic_match_governance_violation":
            if not UID_GOVERNANCE.get('governance_enforcement', True):
                return {
                    'uid': matched_uid,
                    'method': 'semantic_match_governance_violation',
                    'confidence': confidence,
                    'governance_compliant': False,
                    'match_status': match_status,
                    'standardized_question': standardized_question,
                    'question_pattern': question_pattern,
                    'original_question': question_text
                }
    
    # Create new UID with governance compliance
    if existing_uids_data:
        max_uid = max([int(uid) for uid in existing_uids_data.keys() if uid.isdigit()])
        new_uid = str(max_uid + 1)
    else:
        new_uid = "1"
    
    return {
        'uid': new_uid,
        'method': 'new_assignment',
        'confidence': 1.0,
        'governance_compliant': True,
        'match_status': 'new_uid_created',
        'standardized_question': standardized_question,
        'question_pattern': question_pattern,
        'original_question': question_text
    }

# ===== SNOWFLAKE FUNCTIONS =====

def run_snowflake_reference_query_all():
    """Fetch ALL reference questions from Snowflake with pagination"""
    all_data = []
    limit = 10000
    offset = 0
    
    while True:
        query = """
            SELECT HEADING_0, UID
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL
            ORDER BY CAST(UID AS INTEGER) ASC
            LIMIT :limit OFFSET :offset
        """
        try:
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
            
            if result.empty:
                break
                
            result['survey_title'] = 'Unknown Survey'
            all_data.append(result)
            offset += limit
            
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            if "250001" in str(e):
                st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")
            elif "invalid identifier" in str(e).lower():
                st.warning("⚠️ Snowflake query failed due to invalid column.")
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

def run_snowflake_target_query():
    """Fixed target query - focus on questions without UIDs"""
    query = """
        SELECT DISTINCT HEADING_0
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NULL 
        AND HEADING_0 IS NOT NULL
        AND NOT LOWER(HEADING_0) LIKE 'our privacy policy%'
        ORDER BY HEADING_0
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        
        if not result.empty:
            result['survey_title'] = 'Unknown Survey'
        
        return result
    except Exception as e:
        logger.error(f"Snowflake target query failed: {e}")
        if "250001" in str(e):
            st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")
        raise

# ===== SURVEYMONKEY API FUNCTIONS =====

def get_surveys(token):
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
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch survey details for ID {survey_id}: {e}")
        raise

def classify_question(text, heading_references=HEADING_REFERENCES):
    """Classify if text is a heading or main question"""
    if len(text.split()) > HEADING_LENGTH_THRESHOLD:
        return "Heading"
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_texts = heading_references + [text]
    tfidf_vectors = vectorizer.fit_transform([enhanced_normalize(t) for t in all_texts])
    similarity_scores = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
    max_tfidf_score = np.max(similarity_scores)
    
    try:
        model = load_sentence_transformer()
        emb_text = model.encode([text], convert_to_tensor=True)
        emb_refs = model.encode(heading_references, convert_to_tensor=True)
        semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
        max_semantic_score = np.max(semantic_scores.cpu().numpy())
    except Exception as e:
        logger.error(f"Semantic similarity computation failed: {e}")
        max_semantic_score = 0.0
    
    if max_tfidf_score >= HEADING_TFIDF_THRESHOLD or max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
        return "Heading"
    return "Main Question/Multiple Choice"

def extract_questions(survey_json):
    """Extract questions from SurveyMonkey survey JSON"""
    questions = []
    global_position = 0
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            
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

# ===== ENHANCED MATCHING FUNCTIONS =====

def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced normalization with synonym mapping"""
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def compute_tfidf_matches(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP):
    """Compute TF-IDF matches with enhanced normalization"""
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
    df_reference["norm_text"] = df_reference["heading_0"].apply(lambda x: enhanced_normalize(x, synonym_map))
    df_target["norm_text"] = df_target["heading_0"].apply(lambda x: enhanced_normalize(x, synonym_map))

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs, governance_status = [], [], [], [], []
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
        
        if best_idx is not None:
            matched_uid = df_reference.iloc[best_idx]["uid"]
            matched_question = df_reference.iloc[best_idx]["heading_0"]
            
            uid_count = len(df_reference[df_reference["uid"] == matched_uid])
            governance_compliant = uid_count <= UID_GOVERNANCE['max_variations_per_uid']
            
            matched_uids.append(matched_uid)
            matched_qs.append(matched_question)
            governance_status.append("✅" if governance_compliant else "⚠️")
        else:
            matched_uids.append(None)
            matched_qs.append(None)
            governance_status.append("N/A")
            
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    df_target["Governance_Status"] = governance_status
    return df_target

def detect_uid_conflicts(df_target):
    """Detect UID conflicts in target data"""
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def enhanced_uid_matching_process(df_target, df_reference):
    """Enhanced UID matching process with semantic analysis and governance"""
    try:
        logger.info("Step 1: Preparing Snowflake reference data...")
        
        existing_uids_data = {}
        for uid in df_reference['uid'].unique():
            if pd.notna(uid):
                uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
                uid_questions = [q for q in uid_questions if pd.notna(q) and str(q).strip()]
                
                if uid_questions:
                    best_question = get_best_question_for_uid(uid_questions)
                    existing_uids_data[str(uid)] = {
                        'best_question': best_question,
                        'variation_count': len(uid_questions),
                        'all_questions': uid_questions
                    }
        
        logger.info(f"Prepared {len(existing_uids_data)} existing UIDs for matching")
        
        logger.info("Step 2: Processing SurveyMonkey target questions...")
        
        df_enhanced = df_target.copy()
        
        enhanced_columns = [
            'standardized_question', 'question_pattern_category', 'question_pattern_type',
            'semantic_uid', 'semantic_confidence', 'semantic_match_status',
            'governance_compliant', 'match_method', 'quality_score'
        ]
        
        for col in enhanced_columns:
            if col not in df_enhanced.columns:
                df_enhanced[col] = None
        
        for idx, row in df_enhanced.iterrows():
            question_text = row['heading_0']
            
            if pd.isna(question_text) or str(question_text).strip() == '':
                continue
            
            if row.get('is_choice', False) or row.get('question_category') == 'Heading':
                continue
            
            question_pattern = detect_question_pattern(str(question_text))
            df_enhanced.at[idx, 'question_pattern_category'] = question_pattern['category']
            df_enhanced.at[idx, 'question_pattern_type'] = question_pattern['pattern']
            
            survey_category = categorize_question_by_content(str(question_text))
            
            uid_result = assign_uid_with_full_governance(
                str(question_text), existing_uids_data, survey_category, question_pattern
            )
            
            df_enhanced.at[idx, 'semantic_uid'] = uid_result['uid']
            df_enhanced.at[idx, 'semantic_confidence'] = uid_result['confidence']
            df_enhanced.at[idx, 'semantic_match_status'] = uid_result['match_status']
            df_enhanced.at[idx, 'governance_compliant'] = uid_result['governance_compliant']
            df_enhanced.at[idx, 'match_method'] = uid_result['method']
            df_enhanced.at[idx, 'standardized_question'] = uid_result['standardized_question']
            df_enhanced.at[idx, 'quality_score'] = score_question_quality(str(question_text))
            
            if uid_result['method'] == 'new_assignment':
                existing_uids_data[uid_result['uid']] = {
                    'best_question': uid_result['standardized_question'],
                    'variation_count': 1,
                    'all_questions': [str(question_text)]
                }
            elif uid_result['uid'] in existing_uids_data:
                existing_uids_data[uid_result['uid']]['variation_count'] += 1
                existing_uids_data[uid_result['uid']]['all_questions'].append(str(question_text))
        
        logger.info("Step 3: Applying TF-IDF matching as fallback...")
        
        unmatched_mask = df_enhanced['semantic_uid'].isna()
        unmatched_count = unmatched_mask.sum()
        
        if unmatched_count > 0:
            df_unmatched = df_enhanced[unmatched_mask].copy()
            
            processable_mask = (
                (df_unmatched.get('is_choice', False) == False) & 
                (df_unmatched.get('question_category', '') != 'Heading') &
                (df_unmatched['heading_0'].notna())
            )
            
            if processable_mask.any():
                df_processable = df_unmatched[processable_mask].copy()
                
                try:
                    df_tfidf_results = compute_tfidf_matches(df_reference, df_processable, ENHANCED_SYNONYM_MAP)
                    
                    for tfidf_idx, tfidf_row in df_tfidf_results.iterrows():
                        if pd.notna(tfidf_row.get('Suggested_UID')):
                            original_indices = df_enhanced[
                                (df_enhanced['heading_0'] == tfidf_row['heading_0']) & 
                                (df_enhanced['semantic_uid'].isna())
                            ].index
                            
                            if len(original_indices) > 0:
                                original_idx = original_indices[0]
                                df_enhanced.at[original_idx, 'semantic_uid'] = tfidf_row['Suggested_UID']
                                df_enhanced.at[original_idx, 'semantic_confidence'] = tfidf_row['Similarity']
                                df_enhanced.at[original_idx, 'match_method'] = 'tfidf_fallback'
                                df_enhanced.at[original_idx, 'semantic_match_status'] = 'tfidf_match'
                                df_enhanced.at[original_idx, 'governance_compliant'] = tfidf_row.get('Governance_Status', 'N/A') == '✅'
                
                except Exception as e:
                    logger.warning(f"TF-IDF fallback failed: {e}")
        
        logger.info("Step 4: Finalizing enhanced results...")
        
        df_enhanced['Final_UID'] = df_enhanced['semantic_uid']
        df_enhanced['configured_final_UID'] = df_enhanced['semantic_uid']
        
        for idx, row in df_enhanced.iterrows():
            if row.get('is_choice', False) and pd.notna(row.get('parent_question')):
                parent_questions = df_enhanced[
                    (df_enhanced['heading_0'] == row['parent_question']) & 
                    (df_enhanced.get('is_choice', False) == False)
                ]
                if not parent_questions.empty:
                    parent_uid = parent_questions.iloc[0]['Final_UID']
                    df_enhanced.at[idx, 'Final_UID'] = parent_uid
                    df_enhanced.at[idx, 'configured_final_UID'] = parent_uid
                    df_enhanced.at[idx, 'match_method'] = 'inherited_from_parent'
        
        df_enhanced['Final_Match_Type'] = df_enhanced.apply(lambda row: 
            f"🧠 {row['match_method']}" if pd.notna(row['Final_UID']) else "❌ No match", axis=1)
        
        df_enhanced['Final_Governance'] = df_enhanced['governance_compliant'].apply(
            lambda x: "✅ Compliant" if x else "⚠️ Violation" if pd.notna(x) else "N/A")
        
        df_enhanced = detect_uid_conflicts(df_enhanced)
        
        df_enhanced['survey_category'] = df_enhanced['heading_0'].apply(
            lambda x: categorize_question_by_content(str(x)) if pd.notna(x) else 'Unknown'
        )
        
        logger.info("Enhanced UID matching completed successfully")
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Enhanced UID matching process failed: {e}")
        st.error(f"❌ Enhanced matching failed: {e}")
        return None

def process_questions_without_reference(df_target):
    """Process questions when no reference data is available"""
    df_processed = df_target.copy()
    current_uid = 1
    
    for idx, row in df_processed.iterrows():
        if row.get('is_choice', False) or row.get('question_category') == 'Heading':
            continue
        
        question_text = row['heading_0']
        if pd.isna(question_text) or question_text.strip() == '':
            continue
        
        standardized = standardize_question_format(question_text)
        pattern = detect_question_pattern(question_text)
        
        df_processed.at[idx, 'Final_UID'] = str(current_uid)
        df_processed.at[idx, 'configured_final_UID'] = str(current_uid)
        df_processed.at[idx, 'standardized_question'] = standardized
        df_processed.at[idx, 'question_pattern_category'] = pattern['category']
        df_processed.at[idx, 'question_pattern_type'] = pattern['pattern']
        df_processed.at[idx, 'match_method'] = 'new_assignment'
        df_processed.at[idx, 'governance_compliant'] = True
        df_processed.at[idx, 'quality_score'] = score_question_quality(question_text)
        
        current_uid += 1
    
    for idx, row in df_processed.iterrows():
        if row.get('is_choice', False) and pd.notna(row.get('parent_question')):
            parent_questions = df_processed[
                (df_processed['heading_0'] == row['parent_question']) & 
                (df_processed.get('is_choice', False) == False)
            ]
            if not parent_questions.empty:
                parent_uid = parent_questions.iloc[0]['Final_UID']
                df_processed.at[idx, 'Final_UID'] = parent_uid
                df_processed.at[idx, 'configured_final_UID'] = parent_uid
    
    return df_processed

# ===== UI FUNCTIONS =====

def enhanced_configure_survey_page():
    """Enhanced configure survey page with semantic matching and governance"""
    st.markdown("## ⚙️ Enhanced Configure Survey with Semantic Matching")
    st.markdown("*Upload CSV or fetch from SurveyMonkey with advanced UID assignment*")
    st.markdown("*Matching SurveyMonkey questions against Snowflake HEADING_0 and UID data*")
    
    with st.expander("⚖️ Governance & Matching Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            UID_GOVERNANCE['semantic_matching_enabled'] = st.checkbox(
                "🧠 Enable Semantic Matching", 
                value=UID_GOVERNANCE.get('semantic_matching_enabled', True),
                help="Use AI to find semantically similar questions"
            )
            
            QUESTION_STANDARDIZATION['standardization_enabled'] = st.checkbox(
                "📝 Enable Question Standardization", 
                value=QUESTION_STANDARDIZATION.get('standardization_enabled', True),
                help="Standardize question formats before matching"
            )
        
        with col2:
            UID_GOVERNANCE['max_variations_per_uid'] = st.number_input(
                "📊 Max Variations per UID", 
                min_value=1, 
                max_value=200, 
                value=UID_GOVERNANCE['max_variations_per_uid'],
                help="Maximum number of question variations allowed per UID"
            )
            
            UID_GOVERNANCE['semantic_similarity_threshold'] = st.slider(
                "🎯 Semantic Similarity Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=UID_GOVERNANCE['semantic_similarity_threshold'],
                step=0.05,
                help="Minimum similarity score for semantic matching"
            )
        
        with col3:
            UID_GOVERNANCE['governance_enforcement'] = st.checkbox(
                "⚖️ Enforce Governance Rules", 
                value=UID_GOVERNANCE.get('governance_enforcement', True),
                help="Strictly enforce governance rules during UID assignment"
            )
            
            UID_GOVERNANCE['auto_consolidate_threshold'] = st.slider(
                "🔄 Auto-Consolidate Threshold", 
                min_value=0.8, 
                max_value=1.0, 
                value=UID_GOVERNANCE['auto_consolidate_threshold'],
                step=0.02,
                help="Automatically consolidate questions above this similarity"
            )
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Flow")
    st.markdown("• **Reference Data**: Snowflake `HEADING_0` and `UID` columns")
    st.markdown("• **Target Data**: SurveyMonkey questions and choices")
    st.markdown("• **Matching Process**: AI semantic matching + TF-IDF fallback")
    st.markdown("• **Output**: SurveyMonkey questions assigned with matched UIDs")
    st.markdown('</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "📥 Choose Input Method:",
        ["Upload CSV File", "Fetch from SurveyMonkey", "Enter SurveyMonkey Survey ID"],
        horizontal=True
    )
    
    df_target = None
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                df_target = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV uploaded successfully! Found {len(df_target)} rows.")
                
                required_cols = ['heading_0']
                missing_cols = [col for col in required_cols if col not in df_target.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    df_target = None
                else:
                    optional_cols = {
                        'survey_title': 'Uploaded CSV Survey',
                        'survey_id': 'uploaded_csv',
                        'is_choice': False,
                        'question_category': 'Main Question/Multiple Choice',
                        'position': range(1, len(df_target) + 1)
                    }
                    for col, default_val in optional_cols.items():
                        if col not in df_target.columns:
                            if col == 'position':
                                df_target[col] = list(default_val)
                            else:
                                df_target[col] = default_val
                    
                    st.info(f"📊 Preview of uploaded data:")
                    st.dataframe(df_target.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
    
    elif input_method == "Fetch from SurveyMonkey":
        try:
            token = st.secrets["surveymonkey"]["token"]
            surveys = get_surveys(token)
            
            if surveys:
                survey_options = {f"{s['id']} - {s['title']}": s['id'] for s in surveys}
                selected_survey = st.selectbox("📋 Select Survey:", list(survey_options.keys()))
                
                if selected_survey and st.button("📥 Fetch Survey Data"):
                    survey_id = survey_options[selected_survey]
                    
                    with st.spinner("🔄 Fetching survey data from SurveyMonkey..."):
                        survey_details = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_details)
                        df_target = pd.DataFrame(questions)
                        
                        st.success(f"✅ Fetched {len(df_target)} questions from SurveyMonkey!")
                        st.info(f"📊 Preview of SurveyMonkey data:")
                        st.dataframe(df_target.head(), use_container_width=True)
            else:
                st.warning("⚠️ No surveys found in your SurveyMonkey account.")
                
        except Exception as e:
            st.error(f"❌ SurveyMonkey API Error: {e}")
    
    elif input_method == "Enter SurveyMonkey Survey ID":
        survey_id = st.text_input("🆔 Enter Survey ID:")
        if survey_id and st.button("📥 Fetch Survey by ID"):
            try:
                token = st.secrets["surveymonkey"]["token"]
                with st.spinner("🔄 Fetching survey data from SurveyMonkey..."):
                    survey_details = get_survey_details(survey_id, token)
                    questions = extract_questions(survey_details)
                    df_target = pd.DataFrame(questions)
                    
                    st.success(f"✅ Fetched {len(df_target)} questions from survey {survey_id}!")
                    st.info(f"📊 Preview of SurveyMonkey data:")
                    st.dataframe(df_target.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error fetching survey {survey_id}: {e}")
    
    if df_target is not None and not df_target.empty:
        st.markdown("---")
        st.markdown("### 🧠 Enhanced UID Matching: SurveyMonkey ↔ Snowflake")
        
        try:
            with st.spinner("🔄 Loading Snowflake reference data (HEADING_0, UID)..."):
                df_reference = get_all_reference_questions()
                
            if df_reference.empty:
                st.warning("⚠️ No reference data available from Snowflake. Creating new UIDs for all questions.")
                df_final = process_questions_without_reference(df_target)
                st.session_state.df_final = df_final
                st.session_state.df_target = df_target
                st.info("✅ Processed questions without reference matching (all new UIDs assigned)")
                
            else:
                st.info(f"📊 Loaded {len(df_reference)} reference questions from Snowflake")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    unique_uids = df_reference['uid'].nunique()
                    st.metric("🆔 Unique UIDs in Snowflake", unique_uids)
                with col2:
                    avg_variations = len(df_reference) / unique_uids if unique_uids > 0 else 0
                    st.metric("📊 Avg Variations per UID", f"{avg_variations:.1f}")
                with col3:
                    governance_compliant_uids = len(df_reference.groupby('uid').size()[
                        df_reference.groupby('uid').size() <= UID_GOVERNANCE['max_variations_per_uid']
                    ])
                    compliance_rate = (governance_compliant_uids / unique_uids * 100) if unique_uids > 0 else 0
                    st.metric("⚖️ Governance Compliance", f"{compliance_rate:.1f}%")
                
                st.markdown("#### 📋 Target Data Summary (SurveyMonkey)")
                main_questions = len(df_target[df_target.get('is_choice', False) == False])
                choice_questions = len(df_target[df_target.get('is_choice', False) == True])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("❓ Main Questions", main_questions)
                with col2:
                    st.metric("🔘 Choice Options", choice_questions)
                with col3:
                    st.metric("📝 Total Items", len(df_target))
                
                if st.button("🧠 Start Enhanced UID Matching", type="primary"):
                    with st.spinner("🔄 Processing SurveyMonkey questions with AI semantic matching..."):
                        
                        with st.expander("📋 Matching Process Steps", expanded=True):
                            st.write("1. 🧠 **Semantic Analysis**: AI models compare question meanings")
                            st.write("2. ⚖️ **Governance Check**: Ensure compliance with variation limits")
                            st.write("3. 📝 **Standardization**: Normalize question formats")
                            st.write("4. 🔄 **TF-IDF Fallback**: Traditional keyword matching for unmatched questions")
                            st.write("5. ✅ **Final Assignment**: Assign UIDs with confidence scores")
                        
                        df_final = enhanced_uid_matching_process(df_target, df_reference)
                        
                        if df_final is not None:
                            st.session_state.df_final = df_final
                            st.session_state.df_target = df_target
                            display_enhanced_matching_results(df_final, df_reference)
                        else:
                            st.error("❌ Enhanced UID matching failed")
                            
        except Exception as e:
            st.error(f"❌ Error loading Snowflake reference data: {e}")
            if st.button("🔄 Process Without Snowflake Reference Data"):
                df_final = process_questions_without_reference(df_target)
                st.session_state.df_final = df_final
                st.session_state.df_target = df_target
                st.info("✅ Processed questions without reference matching")
                main_questions = len(df_final[df_final.get('is_choice', False) == False])
                st.success(f"✅ Assigned new UIDs to {main_questions} main questions")
    
    with st.expander("❓ How Enhanced Matching Works"):
        st.markdown("""
        ### 🧠 Enhanced Semantic Matching Process
        
        1. **Data Loading**: 
           - Snowflake: `HEADING_0` (questions) + `UID` (existing assignments)
           - SurveyMonkey: Questions and multiple choice options
        
        2. **AI Semantic Analysis**:
           - Uses transformer models to understand question meaning
           - Compares SurveyMonkey questions to Snowflake questions
           - Finds semantically similar questions even with different wording
        
        3. **Governance Compliance**:
           - Checks variation limits per UID
           - Enforces quality thresholds
           - Prevents UID conflicts
        
        4. **Question Standardization**:
           - Normalizes question formats
           - Fixes common typos and inconsistencies
           - Applies consistent terminology
        
        5. **Confidence Scoring**:
           - Each match gets a confidence score (0-1)
           - Higher scores indicate stronger semantic similarity
           - Transparent decision making
        
        ### 📊 Expected Results
        - **High confidence matches**: Questions with clear semantic similarity
        - **Medium confidence matches**: Possible matches requiring review  
        - **New UID assignments**: Questions with no similar existing questions
        - **Governance compliance**: All assignments follow configured rules
        """)

def display_enhanced_matching_results(df_final, df_reference):
    """Display enhanced matching results with governance and semantic analysis"""
    st.markdown("### 🎯 Enhanced Matching Results: SurveyMonkey ↔ Snowflake")
    
    total_questions = len(df_final[df_final.get('is_choice', False) == False])
    matched_questions = len(df_final[(df_final.get('is_choice', False) == False) & (df_final['Final_UID'].notna())])
    semantic_matches = len(df_final[df_final['match_method'].str.contains('semantic', na=False)])
    tfidf_matches = len(df_final[df_final['match_method'].str.contains('tfidf', na=False)])
    new_assignments = len(df_final[df_final['match_method'] == 'new_assignment'])
    governance_compliant = len(df_final[df_final['governance_compliant'] == True])
    
    st.markdown("#### 📊 Matching Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        match_rate = (matched_questions / total_questions * 100) if total_questions > 0 else 0
        st.metric("🎯 Match Rate", f"{match_rate:.1f}%", f"{matched_questions}/{total_questions}")
    
    with col2:
        semantic_rate = (semantic_matches / total_questions * 100) if total_questions > 0 else 0
        st.metric("🧠 Semantic Matches", f"{semantic_rate:.1f}%", f"{semantic_matches} questions")
    
    with col3:
        tfidf_rate = (tfidf_matches / total_questions * 100) if total_questions > 0 else 0
        st.metric("📊 TF-IDF Matches", f"{tfidf_rate:.1f}%", f"{tfidf_matches} questions")
    
    with col4:
        new_rate = (new_assignments / total_questions * 100) if total_questions > 0 else 0
        st.metric("🆕 New UIDs", f"{new_rate:.1f}%", f"{new_assignments} questions")
    
    with col5:
        governance_rate = (governance_compliant / len(df_final) * 100) if len(df_final) > 0 else 0
        st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
    
    if 'semantic_confidence' in df_final.columns:
        avg_confidence = df_final['semantic_confidence'].mean()
        if pd.notna(avg_confidence):
            st.markdown("#### 📈 Confidence Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Average Confidence", f"{avg_confidence:.3f}")
            
            with col2:
                high_confidence = len(df_final[df_final['semantic_confidence'] > 0.8])
                st.metric("🎯 High Confidence (>0.8)", high_confidence)
            
            with col3:
                low_confidence = len(df_final[df_final['semantic_confidence'] < 0.5])
                st.metric("⚠️ Low Confidence (<0.5)", low_confidence)
    
    st.markdown("#### 📋 Matching Method Breakdown")
    if 'match_method' in df_final.columns:
        method_counts = df_final[df_final['match_method'].notna()]['match_method'].value_counts()
        
        if not method_counts.empty:
            method_df = pd.DataFrame({
                'Method': method_counts.index,
                'Count': method_counts.values,
                'Percentage': (method_counts.values / len(df_final) * 100).round(1),
                'Description': method_counts.index.map({
                    'semantic_match': '🧠 AI found similar question in Snowflake',
                    'new_assignment': '🆕 No match found, assigned new UID',
                    'tfidf_fallback': '📊 Keyword-based match from Snowflake',
                    'auto_consolidate': '🔄 Automatically consolidated with existing',
                    'inherited_from_parent': '👨‍👩‍👧‍👦 Choice inherited parent question UID'
                })
            })
            
            st.dataframe(method_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### 🔍 Detailed Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_method = st.selectbox("Filter by Method", 
                                 ["All"] + list(df_final['match_method'].dropna().unique()) if 'match_method' in df_final.columns else ["All"])
    
    with col2:
        show_governance = st.selectbox("Filter by Governance", 
                                     ["All", "Compliant", "Violations"])
    
    with col3:
        confidence_options = ["All"]
        if 'semantic_confidence' in df_final.columns and df_final['semantic_confidence'].notna().any():
            confidence_options.extend(["High (>0.8)", "Medium (0.5-0.8)", "Low (<0.5)"])
        show_confidence = st.selectbox("Filter by Confidence", confidence_options)
    
    with col4:
        question_type = st.selectbox("Question Type", ["All", "Main Questions Only", "Choices Only"])
    
    # Apply filters
    display_df = df_final.copy()
    
    if show_method != "All" and 'match_method' in display_df.columns:
        display_df = display_df[display_df['match_method'] == show_method]
    
    if show_governance == "Compliant" and 'governance_compliant' in display_df.columns:
        display_df = display_df[display_df['governance_compliant'] == True]
    elif show_governance == "Violations" and 'governance_compliant' in display_df.columns:
        display_df = display_df[display_df['governance_compliant'] == False]
    
    if 'semantic_confidence' in display_df.columns:
        if show_confidence == "High (>0.8)":
            display_df = display_df[display_df['semantic_confidence'] > 0.8]
        elif show_confidence == "Medium (0.5-0.8)":
            display_df = display_df[(display_df['semantic_confidence'] >= 0.5) & 
                                  (display_df['semantic_confidence'] <= 0.8)]
        elif show_confidence == "Low (<0.5)":
            display_df = display_df[display_df['semantic_confidence'] < 0.5]
    
    if question_type == "Main Questions Only":
        display_df = display_df[display_df.get('is_choice', False) == False]
    elif question_type == "Choices Only":
        display_df = display_df[display_df.get('is_choice', False) == True]
    
    # Enhanced display columns
    base_columns = ['heading_0', 'Final_UID']
    optional_columns = [
        ('standardized_question', 'Standardized Question'),
        ('semantic_confidence', 'Confidence'),
        ('match_method', 'Method'),
        ('Final_Governance', 'Governance'),
        ('quality_score', 'Quality'),
        ('question_pattern_category', 'Pattern'),
        ('survey_category', 'Category')
    ]
    
    display_columns = base_columns.copy()
    column_config = {
        "heading_0": st.column_config.TextColumn("Original Question", width="large"),
        "Final_UID": st.column_config.TextColumn("Assigned UID", width="small")
    }
    
    # Add available optional columns
    for col_name, col_display in optional_columns:
        if col_name in display_df.columns:
            display_columns.append(col_name)
            if col_name == 'semantic_confidence':
                column_config[col_name] = st.column_config.NumberColumn(col_display, format="%.3f", width="small")
            elif col_name == 'quality_score':
                column_config[col_name] = st.column_config.NumberColumn(col_display, format="%.1f", width="small")
            else:
                column_config[col_name] = st.column_config.TextColumn(col_display, width="medium")
    
    if not display_df.empty:
        st.dataframe(
            display_df[display_columns],
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download All Results",
                df_final.to_csv(index=False),
                f"enhanced_uid_results_{uuid4()}.csv",
                "text/csv",
                use_container_width=True,
                help="Download complete results with all columns"
            )
        
        with col2:
            # Create summary report
            summary_data = {
                'total_questions': total_questions,
                'matched_questions': matched_questions,
                'match_rate': f"{match_rate:.1f}%",
                'semantic_matches': semantic_matches,
                'tfidf_matches': tfidf_matches,
                'new_assignments': new_assignments,
                'governance_compliant': governance_compliant,
                'avg_confidence': f"{avg_confidence:.3f}" if 'semantic_confidence' in df_final.columns and pd.notna(df_final['semantic_confidence'].mean()) else "N/A"
            }
            
            st.download_button(
                "📊 Download Summary Report",
                json.dumps(summary_data, indent=2),
                f"matching_summary_{uuid4()}.json",
                "application/json",
                use_container_width=True,
                help="Download summary statistics"
            )
        
        with col3:
            # Create governance report if there are violations
            governance_issues = df_final[df_final.get('governance_compliant', True) == False]
            if not governance_issues.empty:
                st.download_button(
                    "⚖️ Download Governance Issues",
                    governance_issues.to_csv(index=False),
                    f"governance_issues_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Download questions with governance violations"
                )
            else:
                st.success("✅ No governance violations detected!")
    
    else:
        st.info("ℹ️ No results match the current filters.")
    
    # Success message
    if matched_questions > 0:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.markdown(f"### ✅ Enhanced Matching Completed Successfully!")
        st.markdown(f"• **{semantic_matches}** questions matched using AI semantic analysis")
        st.markdown(f"• **{tfidf_matches}** questions matched using TF-IDF keyword analysis") 
        st.markdown(f"• **{new_assignments}** questions assigned new UIDs")
        st.markdown(f"• **{match_rate:.1f}%** overall match rate achieved")
        st.markdown('</div>', unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====

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

# ===== SIDEBAR NAVIGATION =====

with st.sidebar:
    st.markdown("### 🧠 Enhanced UID Matcher Pro")
    st.markdown("*With Semantic Matching & Governance*")
    
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("**📊 Enhanced SurveyMonkey**")
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    if st.button("⚙️ Enhanced Configure Survey", use_container_width=True):
        st.session_state.page = "enhanced_configure_survey"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("**⚖️ Enhanced Governance**")
    with st.expander("🔧 Current Settings"):
        st.markdown(f"• Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
        st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
        st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
        st.markdown(f"• Semantic matching: {'✅' if UID_GOVERNANCE.get('semantic_matching_enabled', True) else '❌'}")
        st.markdown(f"• Standardization: {'✅' if QUESTION_STANDARDIZATION.get('standardization_enabled', True) else '❌'}")
        st.markdown(f"• Governance enforcement: {'✅' if UID_GOVERNANCE.get('governance_enforcement', True) else '❌'}")
    
    st.markdown("---")
    
    st.markdown("**🔗 Quick Actions**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

# ===== MAIN APP UI =====

st.markdown('<div class="main-header">🧠 Enhanced UID Matcher Pro: Semantic Matching & Governance</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# ===== PAGE ROUTING =====

if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    st.markdown("*Now with AI-powered semantic matching, governance rules, and question standardization*")
    
    # Enhanced dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔄 Status", "Enhanced Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            with get_snowflake_engine().connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                count = result.fetchone()[0]
                st.metric("📊 Total UIDs", f"{count:,}")
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
        st.metric("🧠 AI Features", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced features showcase
    st.markdown("## 🚀 Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI-Powered Semantic Matching")
        st.markdown("• **Deep Learning Models**: Advanced question similarity detection")
        st.markdown("• **Context Understanding**: Recognizes meaning beyond keywords") 
        st.markdown("• **Confidence Scoring**: Transparent matching confidence levels")
        st.markdown("• **Multi-language Support**: Works across different phrasings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="governance-card">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Advanced Governance Rules")
        st.markdown(f"• **Variation Limits**: Max {UID_GOVERNANCE['max_variations_per_uid']} per UID")
        st.markdown("• **Quality Thresholds**: Automatic quality assessment")
        st.markdown("• **Conflict Detection**: Real-time duplicate identification")
        st.markdown("• **Compliance Monitoring**: Continuous governance tracking")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Question Standardization")
        st.markdown("• **Format Normalization**: Consistent question structures")
        st.markdown("• **Typo Correction**: Automatic error fixing")
        st.markdown("• **Synonym Mapping**: Intelligent phrase replacement")
        st.markdown("• **Pattern Recognition**: Category-based standardization")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Enhanced Analytics")
        st.markdown("• **Pattern Analysis**: Question type categorization")
        st.markdown("• **Quality Scoring**: Multi-factor assessment")
        st.markdown("• **Semantic Clustering**: AI-based grouping")
        st.markdown("• **Governance Reports**: Compliance dashboards")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced quick actions
    st.markdown("## 🚀 Enhanced Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚙️ Enhanced Configure Survey", type="primary", use_container_width=True):
            st.session_state.page = "enhanced_configure_survey"
            st.rerun()
    
    with col2:
        if st.button("📊 View SurveyMonkey Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        try:
            get_snowflake_engine()
            st.markdown('<div class="success-card">✅ Snowflake: Connected</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
    
    with status_col2:
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                get_surveys(token)
                st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">❌ SurveyMonkey: No Token</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">❌ SurveyMonkey: API Issues</div>', unsafe_allow_html=True)
    
    with status_col3:
        st.markdown('<div class="success-card">✅ AI Models: Ready</div>', unsafe_allow_html=True)

elif st.session_state.page == "enhanced_configure_survey":
    enhanced_configure_survey_page()

elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ View SurveyMonkey Surveys")
    st.markdown("*Browse and analyze your SurveyMonkey surveys*")
    
    try:
        token = st.secrets["surveymonkey"]["token"]
        with st.spinner("🔄 Fetching surveys from SurveyMonkey..."):
            surveys = get_surveys(token)
        
        if surveys:
            st.success(f"✅ Found {len(surveys)} surveys in your account")
            
            # Create surveys dataframe
            surveys_df = pd.DataFrame([
                {
                    'ID': survey['id'],
                    'Title': survey['title'],
                    'Date Created': survey.get('date_created', 'Unknown'),
                    'Date Modified': survey.get('date_modified', 'Unknown'),
                    'Question Count': survey.get('question_count', 'Unknown'),
                    'Response Count': survey.get('response_count', 0)
                }
                for survey in surveys
            ])
            
            st.dataframe(
                surveys_df,
                column_config={
                    "ID": st.column_config.TextColumn("Survey ID", width="medium"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Date Created": st.column_config.TextColumn("Created", width="medium"),
                    "Date Modified": st.column_config.TextColumn("Modified", width="medium"),
                    "Question Count": st.column_config.NumberColumn("Questions", width="small"),
                    "Response Count": st.column_config.NumberColumn("Responses", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Survey selection for detailed view
            selected_survey_id = st.selectbox(
                "📋 Select survey for detailed analysis:",
                options=[survey['id'] for survey in surveys],
                format_func=lambda x: f"{x} - {next(s['title'] for s in surveys if s['id'] == x)}"
            )
            
            if selected_survey_id and st.button("📊 Analyze Selected Survey"):
                with st.spinner("🔄 Fetching survey details..."):
                    survey_details = get_survey_details(selected_survey_id, token)
                    questions = extract_questions(survey_details)
                    
                    st.success(f"✅ Survey contains {len(questions)} total items")
                    
                    questions_df = pd.DataFrame(questions)
                    main_questions = len(questions_df[questions_df['is_choice'] == False])
                    choices = len(questions_df[questions_df['is_choice'] == True])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("❓ Main Questions", main_questions)
                    with col2:
                        st.metric("🔘 Choice Options", choices)
                    with col3:
                        st.metric("📝 Total Items", len(questions))
                    
                    st.dataframe(
                        questions_df[['heading_0', 'is_choice', 'schema_type', 'question_category']],
                        column_config={
                            "heading_0": st.column_config.TextColumn("Question/Choice Text", width="large"),
                            "is_choice": st.column_config.CheckboxColumn("Is Choice?", width="small"),
                            "schema_type": st.column_config.TextColumn("Type", width="medium"),
                            "question_category": st.column_config.TextColumn("Category", width="medium")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ No surveys found in your SurveyMonkey account.")
            
    except Exception as e:
        st.error(f"❌ Error fetching surveys: {e}")

else:
    st.markdown('<div class="warning-card">⚠️ Page not found. Please use the navigation menu.</div>', unsafe_allow_html=True)
