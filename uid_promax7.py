# ============= IMPORTS AND SETUP =============
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

# ============= STREAMLIT CONFIGURATION =============
st.set_page_config(
    page_title="UID Matcher Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# ============= CSS STYLES =============
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

# ============= LOGGING SETUP =============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= CONSTANTS AND CONFIGURATION =============
# Matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50

# Model and API settings
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
MAX_SURVEYS_PER_BATCH = 10

# Identity types for export filtering
IDENTITY_TYPES = [
    'full name', 'first name', 'last name', 'e-mail', 'company', 'gender', 
    'country', 'age', 'title', 'role', 'phone number', 'location', 
    'pin', 'passport', 'date of birth', 'uct', 'student number',
    'department', 'region', 'city', 'id number', 'marital status',
    'education level', 'english proficiency', 'email', 'surname',
    'name', 'contact', 'address', 'mobile', 'telephone', 'qualification',
    'degree', 'identification', 'birth', 'married', 'single', 'language',
    'sex', 'position', 'job', 'organization', 'organisation'
]

# Enhanced synonym mapping
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

# Survey categorization keywords
SURVEY_CATEGORIES = {
    "Application": ["application", "apply", "applying", "candidate", "candidacy", "admission"],
    "Pre programme": ["pre programme", "pre-programme", "pre program", "pre-program", "before programme", "preparation", "prep"],
    "Enrollment": ["enrollment", "enrolment", "enroll", "enrol", "registration", "register"],
    "Progress Review": ["progress", "review", "assessment", "evaluation", "mid-point", "checkpoint", "interim"],
    "Impact": ["impact", "outcome", "result", "effect", "change", "transformation", "benefit"],
    "GROW": ["GROW"],
    "Feedback": ["feedback", "comment", "suggestion", "opinion", "review", "rating"],
    "Pulse": ["pulse", "check-in", "checkin", "quick survey", "pulse survey"],
    "Your Experience": ["experience", "your experience", "participant experience", "learner experience", "journey"]
}

# UID governance rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'conflict_resolution_threshold': 10,
    'high_conflict_threshold': 100
}

# Heading reference texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda.",
    # ... (add other heading references)
    "Understanding your future plans and perspectives helps us anticipate trends and prepare for the evolving needs of the BDS sector.",
    "Thank You for Your Participation",
    "Your participation is a significant step towards creating a more robust, responsive, and effective BDS ecosystem that can drive sustainable MSME growth and contribute to Rwanda's economic development."
]

# ============= SESSION STATE INITIALIZATION =============
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "df_target": None,
        "df_final": None,
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "question_bank": None,
        "question_bank_with_authority": None,
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
        "fetched_survey_ids": [],
        "categorized_questions": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============= CACHED RESOURCES =============
@st.cache_resource
def load_sentence_transformer():
    """Load the sentence transformer model"""
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_snowflake_engine():
    """Create and cache Snowflake connection engine"""
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
                "UID matching is disabled, but you can use SurveyMonkey features."
            )
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    """Create and cache TF-IDF vectors"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# ============= UTILITY FUNCTIONS =============
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
    """Enhanced question quality scoring with English structure preference"""
    try:
        if not isinstance(question, str) or len(question.strip()) < 5:
            return 0
        
        score = 0
        text = question.lower().strip()
        original_text = question.strip()
        
        # Length scoring (optimal range 10-100 characters)
        length = len(question)
        if 10 <= length <= 100:
            score += 25
        elif 5 <= length <= 150:
            score += 15
        elif length < 5:
            score -= 25
        
        # Question format scoring
        if original_text.endswith('?'):
            score += 30
        
        # English question words at the beginning
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
        first_three_words = text.split()[:3]
        if any(word in first_three_words for word in question_words):
            score += 25
        
        # Proper capitalization
        if question and question[0].isupper():
            score += 15
        
        # Grammar and structure bonuses
        if ' is ' in text or ' are ' in text:
            score += 10
        
        # Complete sentence structure
        word_count = len(question.split())
        if 3 <= word_count <= 15:
            score += 20
        elif word_count < 3:
            score -= 20
        
        # Avoid common artifacts
        bad_patterns = [
            'click here', 'please select', '...', 'n/a', 'other', 
            'select one', 'choose all', 'privacy policy', 'thank you',
            'contact us', 'submit', 'continue', '<div', '<span', 'html'
        ]
        if any(pattern in text for pattern in bad_patterns):
            score -= 30
        
        # Avoid HTML content
        if '<' in question and '>' in question:
            score -= 40
        
        # Bonus for well-formed questions
        if original_text.endswith('?') and any(word in first_three_words for word in question_words):
            score += 15
        
        return max(0, score)
        
    except Exception as e:
        logger.error(f"Error scoring question quality: {e}")
        return 0

def get_best_question_for_uid(variants, occurrence_counts=None):
    """Select the best quality question from variants"""
    try:
        if not variants:
            return None
        
        valid_variants = [v for v in variants if isinstance(v, str) and len(v.strip()) > 3]
        if not valid_variants:
            return None
        
        # If occurrence counts provided, prioritize by highest authority count first
        if occurrence_counts and isinstance(occurrence_counts, dict):
            variant_scores = []
            for variant in valid_variants:
                count = occurrence_counts.get(variant, 0)
                quality = score_question_quality(variant)
                variant_scores.append((variant, count, quality))
            
            variant_scores.sort(key=lambda x: (-x[1], -x[2], len(x[0])))
            return variant_scores[0][0]
        
        # Fallback to quality-based selection
        scored_variants = [(v, score_question_quality(v)) for v in valid_variants]
        
        def sort_key(item):
            question, score = item
            has_question_mark = question.strip().endswith('?')
            has_question_word = any(question.lower().strip().startswith(word) for word in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should'])
            proper_structure_bonus = 1000 if (has_question_mark and has_question_word) else 0
            
            return (-score - proper_structure_bonus, len(question))
        
        scored_variants.sort(key=sort_key)
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
        
        # Check against heading references
        for ref in heading_references:
            if text.strip() in ref or ref in text.strip():
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

def calculate_matched_percentage(df_final):
    """Calculate percentage of matched questions"""
    if df_final is None or df_final.empty:
        return 0.0
    df_main = df_final[df_final["is_choice"] == False].copy()
    privacy_filter = ~df_main["question_text"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["question_text"].str.contains(html_pattern, case=False, na=False, regex=True)
    eligible_questions = df_main[privacy_filter & html_filter]
    if eligible_questions.empty:
        return 0.0
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    return round((len(matched_questions) / len(eligible_questions)) * 100, 2)

# ============= IDENTITY DETECTION FUNCTIONS =============
def contains_identity_info(text):
    """Check if question/choice text contains identity information"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower().strip()
    
    # Check for direct matches
    for identity_type in IDENTITY_TYPES:
        if identity_type in text_lower:
            return True
    
    # Additional patterns for identity detection
    identity_patterns = [
        r'\b(name|surname|firstname|lastname)\b',
        r'\b(email|e-mail|mail)\b',
        r'\b(company|organization|organisation)\b',
        r'\b(phone|mobile|telephone|contact)\b',
        r'\b(address|location)\b',
        r'\b(age|gender|sex)\b',
        r'\b(title|position|role|job)\b',
        r'\b(country|region|city|department)\b',
        r'\b(id|identification|passport|pin)\b',
        r'\b(student number|uct)\b',
        r'\b(date of birth|dob|birth)\b',
        r'\b(marital status|married|single)\b',
        r'\b(education|qualification|degree)\b',
        r'\b(english proficiency|language)\b'
    ]
    
    for pattern in identity_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def determine_identity_type(text):
    """Determine the specific identity type from question text"""
    if not isinstance(text, str):
        return 'Unknown'
    
    text_lower = text.lower().strip()
    
    # Priority order for identity type detection
    if any(name in text_lower for name in ['first name', 'firstname']):
        return 'First Name'
    elif any(name in text_lower for name in ['last name', 'lastname', 'surname']):
        return 'Last Name'
    elif any(name in text_lower for name in ['full name']) or ('name' in text_lower and 'first' not in text_lower and 'last' not in text_lower and 'company' not in text_lower):
        return 'Full Name'
    elif any(email in text_lower for email in ['email', 'e-mail', 'mail']):
        return 'E-Mail'
    elif any(company in text_lower for company in ['company', 'organization', 'organisation']):
        return 'Company'
    elif any(phone in text_lower for phone in ['phone', 'mobile', 'telephone']):
        return 'Phone Number'
    elif 'gender' in text_lower or 'sex' in text_lower:
        return 'Gender'
    elif 'age' in text_lower:
        return 'Age'
    elif any(title in text_lower for title in ['title', 'position', 'role', 'job']):
        return 'Title/Role'
    elif 'country' in text_lower:
        return 'Country'
    elif 'region' in text_lower:
        return 'Region'
    elif 'city' in text_lower:
        return 'City'
    elif 'department' in text_lower:
        return 'Department'
    elif any(loc in text_lower for loc in ['location', 'address']):
        return 'Location'
    elif any(id_type in text_lower for id_type in ['id number', 'identification']):
        return 'ID Number'
    elif 'passport' in text_lower or 'pin' in text_lower:
        return 'PIN/ Passport'
    elif 'student number' in text_lower:
        return 'Student Number'
    elif 'uct' in text_lower:
        return 'UCT'
    elif any(dob in text_lower for dob in ['date of birth', 'dob', 'birth']):
        return 'Date of Birth'
    elif 'marital' in text_lower:
        return 'Marital Status'
    elif any(edu in text_lower for edu in ['education', 'qualification', 'degree']):
        return 'Education level'
    elif 'english proficiency' in text_lower or ('language' in text_lower and 'proficiency' in text_lower):
        return 'English Proficiency'
    else:
        return 'Other'

# ============= SURVEY CATEGORIZATION FUNCTIONS =============
def categorize_survey_by_title(title):
    """Categorize survey based on title keywords"""
    if not isinstance(title, str):
        return "Uncategorized"
    
    title_lower = title.lower().strip()
    
    # Check for exact GROW match first (case sensitive)
    if "GROW" in title:
        return "GROW"
    
    # Check other categories
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == "GROW":  # Skip GROW as we already checked
            continue
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    
    return "Uncategorized"

def clean_question_text(text):
    """Clean question text by removing year specifications and extracting core question"""
    if not isinstance(text, str):
        return text
    
    # Remove year specifications like (i.e. 1 Jan. 2024 - 31 Dec. 2024)
    text = re.sub(r'\(i\.e\.\s*\d{1,2}\s+\w+\.?\s+\d{4}\s*-\s*\d{1,2}\s+\w+\.?\s+\d{4}\)', '', text)
    
    # Remove other date patterns
    text = re.sub(r'\(\d{1,2}\s+\w+\.?\s+\d{4}\s*-\s*\d{1,2}\s+\w+\.?\s+\d{4}\)', '', text)
    
    # For mobile number patterns, extract the main question part
    if 'Your mobile number' in text and '<br>' in text:
        parts = text.split('<br>')
        if parts:
            return parts[0].strip()
    
    # For questions with HTML formatting, extract the main question
    if '<br>' in text:
        parts = text.split('<br>')
        for part in parts:
            clean_part = re.sub(r'<[^>]+>', '', part).strip()
            if len(clean_part) > 10 and not clean_part.startswith('Country area'):
                return clean_part
    
    # Clean up extra spaces and HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_question_for_grouping(text):
    """Normalize question text for grouping similar questions"""
    if not isinstance(text, str):
        return text
    
    cleaned = clean_question_text(text)
    normalized = cleaned.lower().strip()
    normalized = re.sub(r'\s*-\s*', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

# ============= CACHE MANAGEMENT =============
def load_cached_survey_data():
    """Load cached survey data if available and recent"""
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
    """Save survey data to cache"""
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

# ============= SURVEYMONKEY API FUNCTIONS =============
def get_surveymonkey_token():
    """Get SurveyMonkey API token from secrets"""
    try:
        return st.secrets["surveymonkey"]["access_token"]
    except Exception as e:
        logger.error(f"Failed to get SurveyMonkey token: {e}")
        return None

def check_surveymonkey_connection():
    """Check SurveyMonkey API connection status"""
    try:
        token = get_surveymonkey_token()
        if not token:
            return False, "No access token available"
        
        # Test API call
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://api.surveymonkey.com/v3/surveys?per_page=1", headers=headers)
        
        if response.status_code == 200:
            return True, "Connected successfully"
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

@st.cache_data
def get_surveys_cached(token):
    """Get all surveys from SurveyMonkey API"""
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
    """Get detailed survey information with retry logic"""
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests")
    response.raise_for_status()
    return response.json()

def extract_questions(survey_json):
    """Extract questions from SurveyMonkey survey JSON"""
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
                    "question_text": q_text,
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
                            "question_text": f"{q_text} - {choice_text}",
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

# ============= SNOWFLAKE DATABASE FUNCTIONS =============
def check_snowflake_connection():
    """Check Snowflake database connection status"""
    try:
        engine = get_snowflake_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT CURRENT_VERSION()"))
            version = result.fetchone()[0]
            return True, f"Connected to Snowflake version {version}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def run_snowflake_reference_query(limit=10000, offset=0):
    """Run basic Snowflake reference query for question bank"""
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
        result = result.rename(columns={'heading_0': 'HEADING_0', 'uid': 'UID'})
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account is locked. UID matching is disabled.")
        raise

@st.cache_data(ttl=600)
def get_question_bank_with_authority_count():
    """Fetch question bank with authority count for Question Bank Viewer"""
    query = """
    SELECT 
        HEADING_0, 
        UID, 
        COUNT(*) as AUTHORITY_COUNT
    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
    WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
    AND TRIM(HEADING_0) != ''
    GROUP BY HEADING_0, UID
    ORDER BY UID, AUTHORITY_COUNT DESC
    """
    
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        
        result.columns = result.columns.str.upper()
        logger.info(f"Question bank with authority count fetched: {len(result)} records")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch question bank with authority count: {e}")
        # Fallback to simple query
        try:
            simple_query = """
            SELECT HEADING_0, MAX(UID) AS UID, 1 AS AUTHORITY_COUNT
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL
            GROUP BY HEADING_0
            ORDER BY CAST(UID AS INTEGER) ASC
            """
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(simple_query), conn)
            
            result.columns = result.columns.str.upper()
            logger.info(f"Question bank fallback query successful: {len(result)} records")
            return result
            
        except Exception as e2:
            logger.error(f"Both enhanced and fallback queries failed: {e2}")
            return pd.DataFrame()

@st.cache_data(ttl=600)
def get_all_reference_questions_from_snowflake():
    """Fetch ALL reference questions from Snowflake for optimization"""
    all_data = []
    limit = 10000
    offset = 0
    
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
                
            result.columns = result.columns.str.upper()
            all_data.append(result)
            offset += limit
            
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            # Fall back to simple query
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
                result.columns = result.columns.str.upper()
                result['OCCURRENCE_COUNT'] = 1
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
        if 'OCCURRENCE_COUNT' not in final_df.columns:
            final_df['OCCURRENCE_COUNT'] = 1
        logger.info(f"Total reference questions fetched from Snowflake: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched from Snowflake")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_configured_surveys_from_snowflake():
    """Get distinct survey IDs that are configured in Snowflake"""
    query = """
        SELECT DISTINCT SURVEY_ID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
        GROUP BY SURVEY_ID 
        ORDER BY SURVEY_ID
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        return result['SURVEY_ID'].tolist()
    except Exception as e:
        logger.error(f"Failed to get configured surveys: {e}")
        return []

def count_configured_surveys_from_surveymonkey(surveys):
    """Count how many SurveyMonkey surveys are configured in Snowflake"""
    try:
        configured_surveys = get_configured_surveys_from_snowflake()
        surveymonkey_survey_ids = [str(survey['id']) for survey in surveys]
        configured_count = len([sid for sid in surveymonkey_survey_ids if sid in configured_surveys])
        return configured_count
    except Exception as e:
        logger.error(f"Failed to count configured surveys: {e}")
        return 0

# ============= DATA PROCESSING FUNCTIONS =============
def get_unique_questions_by_category():
    """Extract unique questions per category from ALL cached survey data"""
    # First try to load from cache
    if st.session_state.all_questions is None or st.session_state.all_questions.empty:
        cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
        if cached_questions is not None and not cached_questions.empty:
            st.session_state.all_questions = cached_questions
            st.session_state.dedup_questions = cached_dedup_questions
            st.session_state.dedup_choices = cached_dedup_choices
            st.session_state.fetched_survey_ids = cached_questions["survey_id"].unique().tolist()
    
    # If still no data, fetch ALL surveys from SurveyMonkey directly
    if st.session_state.all_questions is None or st.session_state.all_questions.empty:
        token = get_surveymonkey_token()
        if token:
            try:
                with st.spinner("🔄 Fetching ALL surveys from SurveyMonkey for categorization..."):
                    surveys = get_surveys_cached(token)
                    combined_questions = []
                    
                    # Fetch all surveys (limit to reasonable number for performance)
                    surveys_to_process = surveys[:50]  # Limit to first 50 surveys
                    
                    progress_bar = st.progress(0)
                    for i, survey in enumerate(surveys_to_process):
                        survey_id = survey['id']
                        try:
                            survey_json = get_survey_details_with_retry(survey_id, token)
                            questions = extract_questions(survey_json)
                            combined_questions.extend(questions)
                            time.sleep(REQUEST_DELAY)
                            progress_bar.progress((i + 1) / len(surveys_to_process))
                        except Exception as e:
                            logger.error(f"Failed to fetch survey {survey_id}: {e}")
                            continue
                    
                    progress_bar.empty()
                    
                    if combined_questions:
                        st.session_state.all_questions = pd.DataFrame(combined_questions)
                        st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                            st.session_state.all_questions["is_choice"] == False
                        ]["question_text"].unique().tolist())
                        st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                            st.session_state.all_questions["is_choice"] == True
                        ]["question_text"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
                        
                        # Save to cache
                        save_cached_survey_data(
                            st.session_state.all_questions,
                            st.session_state.dedup_questions,
                            st.session_state.dedup_choices
                        )
                        
                        st.success(f"✅ Fetched {len(combined_questions)} questions from {len(surveys_to_process)} surveys")
                    
            except Exception as e:
                logger.error(f"Failed to fetch surveys for categorization: {e}")
                st.error(f"❌ Failed to fetch surveys: {str(e)}")
                return pd.DataFrame()
    
    all_questions_df = st.session_state.all_questions
    
    if all_questions_df is None or all_questions_df.empty:
        return pd.DataFrame()
    
    try:
        # Add category column based on survey title
        all_questions_df['survey_category'] = all_questions_df['survey_title'].apply(categorize_survey_by_title)
        
        # Clean and normalize question text
        all_questions_df['cleaned_question_text'] = all_questions_df['question_text'].apply(clean_question_text)
        all_questions_df['normalized_question'] = all_questions_df['cleaned_question_text'].apply(normalize_question_for_grouping)
        
        # Group by category and get unique questions
        category_questions = []
        
        for category in list(SURVEY_CATEGORIES.keys()) + ["Uncategorized"]:
            category_df = all_questions_df[all_questions_df['survey_category'] == category]
            
            if not category_df.empty:
                # Get unique main questions (not choices) by normalized text
                main_questions_df = category_df[category_df['is_choice'] == False].copy()
                if not main_questions_df.empty:
                    unique_main_questions = main_questions_df.groupby('normalized_question').first()
                    
                    # Add main questions
                    for norm_question, question_data in unique_main_questions.iterrows():
                        # Count surveys for this normalized question
                        survey_count = len(main_questions_df[
                            main_questions_df['normalized_question'] == norm_question
                        ]['survey_id'].unique())
                        
                        category_questions.append({
                            'survey_category': category,
                            'question_text': question_data['cleaned_question_text'],
                            'schema_type': question_data.get('schema_type'),
                            'is_choice': False,
                            'parent_question': None,
                            'survey_count': survey_count,
                            'Final_UID': None,
                            'configured_final_UID': None,
                            'Change_UID': None,
                            'required': False
                        })
                
                # Get unique choices by normalized text
                choices_df = category_df[category_df['is_choice'] == True].copy()
                if not choices_df.empty:
                    unique_choices = choices_df.groupby('normalized_question').first()
                    
                    # Add choices
                    for norm_question, choice_data in unique_choices.iterrows():
                        # Count surveys for this normalized choice
                        survey_count = len(choices_df[
                            choices_df['normalized_question'] == norm_question
                        ]['survey_id'].unique())
                        
                        category_questions.append({
                            'survey_category': category,
                            'question_text': choice_data['cleaned_question_text'],
                            'schema_type': choice_data.get('schema_type'),
                            'is_choice': True,
                            'parent_question': choice_data.get('parent_question'),
                            'survey_count': survey_count,
                            'Final_UID': None,
                            'configured_final_UID': None,
                            'Change_UID': None,
                            'required': False
                        })
        
        return pd.DataFrame(category_questions)
        
    except Exception as e:
        logger.error(f"Error in get_unique_questions_by_category: {e}")
        st.error(f"❌ Error processing categorized questions: {str(e)}")
        return pd.DataFrame()

# ============= UID MATCHING FUNCTIONS =============
def run_uid_match(question_bank, df_target):
    """Run UID matching algorithm between question bank and target questions"""
    try:
        # Prepare question bank with normalized text
        question_bank_norm = question_bank.copy()
        question_bank_norm["norm_text"] = question_bank_norm["HEADING_0"].apply(enhanced_normalize)
        
        # Prepare target questions
        df_target_norm = df_target.copy()
        df_target_norm["norm_text"] = df_target_norm["question_text"].apply(enhanced_normalize)
        
        # Get TF-IDF vectors
        vectorizer, reference_vectors = get_tfidf_vectors(question_bank_norm)
        
        # Initialize results
        df_target_norm["Final_UID"] = None
        df_target_norm["Match_Confidence"] = None
        df_target_norm["Final_Match_Type"] = None
        
        # Load sentence transformer for semantic matching
        model = load_sentence_transformer()
        
        # Process in batches
        for start_idx in range(0, len(df_target_norm), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(df_target_norm))
            batch_df = df_target_norm.iloc[start_idx:end_idx].copy()
            
            # Vectorize batch
            batch_vectors = vectorizer.transform(batch_df["norm_text"])
            
            # Calculate TF-IDF similarities
            tfidf_similarities = cosine_similarity(batch_vectors, reference_vectors)
            
            # Process each question in batch
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                tfidf_scores = tfidf_similarities[i]
                max_tfidf_idx = np.argmax(tfidf_scores)
                max_tfidf_score = tfidf_scores[max_tfidf_idx]
                
                # TF-IDF matching
                if max_tfidf_score >= TFIDF_HIGH_CONFIDENCE:
                    matched_uid = question_bank_norm.iloc[max_tfidf_idx]["UID"]
                    df_target_norm.at[idx, "Final_UID"] = matched_uid
                    df_target_norm.at[idx, "Match_Confidence"] = "✅ High"
                    df_target_norm.at[idx, "Final_Match_Type"] = "✅ High"
                elif max_tfidf_score >= TFIDF_LOW_CONFIDENCE:
                    matched_uid = question_bank_norm.iloc[max_tfidf_idx]["UID"]
                    df_target_norm.at[idx, "Final_UID"] = matched_uid
                    df_target_norm.at[idx, "Match_Confidence"] = "⚠️ Low"
                    df_target_norm.at[idx, "Final_Match_Type"] = "⚠️ Low"
                else:
                    # Try semantic matching
                    try:
                        question_embedding = model.encode([row["question_text"]], convert_to_tensor=True)
                        reference_embeddings = model.encode(question_bank_norm["HEADING_0"].tolist(), convert_to_tensor=True)
                        semantic_scores = util.cos_sim(question_embedding, reference_embeddings)[0]
                        max_semantic_score = torch.max(semantic_scores).item()
                        
                        if max_semantic_score >= SEMANTIC_THRESHOLD:
                            max_semantic_idx = torch.argmax(semantic_scores).item()
                            matched_uid = question_bank_norm.iloc[max_semantic_idx]["UID"]
                            df_target_norm.at[idx, "Final_UID"] = matched_uid
                            df_target_norm.at[idx, "Match_Confidence"] = "🧠 Semantic"
                            df_target_norm.at[idx, "Final_Match_Type"] = "🧠 Semantic"
                        else:
                            df_target_norm.at[idx, "Final_UID"] = None
                            df_target_norm.at[idx, "Match_Confidence"] = "❌ No match"
                            df_target_norm.at[idx, "Final_Match_Type"] = "❌ No match"
                    except Exception as e:
                        logger.error(f"Semantic matching failed for question {idx}: {e}")
                        df_target_norm.at[idx, "Final_UID"] = None
                        df_target_norm.at[idx, "Match_Confidence"] = "❌ No match"
                        df_target_norm.at[idx, "Final_Match_Type"] = "❌ No match"
        
        # Remove normalization column before returning
        df_target_norm = df_target_norm.drop(columns=["norm_text"])
        
        return df_target_norm
        
    except Exception as e:
        logger.error(f"UID matching failed: {e}")
        # Return original dataframe with empty UID columns
        df_target["Final_UID"] = None
        df_target["Match_Confidence"] = "❌ Error"
        df_target["Final_Match_Type"] = "❌ Error"
        return df_target

# ============= OPTIMIZATION FUNCTIONS =============
def build_optimized_1to1_question_bank(df_reference):
    """Build optimized 1:1 question bank with conflict resolution"""
    try:
        if df_reference.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Group by UID and collect all questions for each UID
        uid_groups = df_reference.groupby('UID')
        optimized_questions = []
        conflicts_data = []
        
        for uid, group in uid_groups:
            # Get all unique questions for this UID
            questions = group['HEADING_0'].unique()
            occurrence_counts = group.groupby('HEADING_0')['OCCURRENCE_COUNT'].sum().to_dict()
            
            if len(questions) == 1:
                # No conflict - single question for UID
                best_question = questions[0]
                authority_count = occurrence_counts.get(best_question, 1)
            else:
                # Conflict detected - multiple questions for same UID
                # Select best question based on occurrence count and quality
                best_question = get_best_question_for_uid(questions, occurrence_counts)
                authority_count = occurrence_counts.get(best_question, 1)
                
                # Record conflict details
                sorted_questions = sorted(occurrence_counts.items(), key=lambda x: x[1], reverse=True)
                winner = sorted_questions[0]
                top_competitor = sorted_questions[1] if len(sorted_questions) > 1 else ("", 0)
                
                total_occurrences = sum(occurrence_counts.values())
                winner_percentage = (winner[1] / total_occurrences) * 100
                competitor_percentage = (top_competitor[1] / total_occurrences) * 100 if top_competitor[1] > 0 else 0
                authority_difference = winner_percentage - competitor_percentage
                
                conflict_severity = len(questions)
                is_high_conflict = (
                    conflict_severity >= UID_GOVERNANCE['high_conflict_threshold'] or
                    authority_difference < 20  # Less than 20% authority difference
                )
                
                conflicts_data.append({
                    'UID': uid,
                    'question': best_question[:100] + "..." if len(best_question) > 100 else best_question,
                    'winner_UID': uid,
                    'winner_count': winner[1],
                    'winner_percentage': winner_percentage,
                    'top_competitor_UID': uid,
                    'top_competitor_count': top_competitor[1],
                    'top_competitor_percentage': competitor_percentage,
                    'authority_difference': authority_difference,
                    'conflict_severity': conflict_severity,
                    'competing_uids': len(questions),
                    'is_high_conflict': is_high_conflict
                })
            
            # Add to optimized question bank
            optimized_questions.append({
                'UID': uid,
                'HEADING_0': best_question,
                'AUTHORITY_COUNT': authority_count,
                'CONFLICT_COUNT': len(questions) - 1,
                'QUALITY_SCORE': score_question_quality(best_question)
            })
        
        optimized_df = pd.DataFrame(optimized_questions)
        conflicts_df = pd.DataFrame(conflicts_data)
        
        # Store in session state
        st.session_state.optimized_question_bank = optimized_df
        st.session_state.uid_conflicts_summary = conflicts_df
        st.session_state.primary_matching_reference = optimized_df
        
        logger.info(f"Built optimized question bank: {len(optimized_df)} questions, {len(conflicts_df)} conflicts")
        
        return optimized_df, conflicts_df
        
    except Exception as e:
        logger.error(f"Failed to build optimized question bank: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ============= EXPORT FUNCTIONS =============
def prepare_export_data(df_final):
    """Prepare export data split into identity and non-identity tables"""
    try:
        if df_final is None or df_final.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter for main questions only (not choices)
        main_questions = df_final[df_final["is_choice"] == False].copy()
        
        if main_questions.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Add identity classification
        main_questions['is_identity'] = main_questions['question_text'].apply(contains_identity_info)
        main_questions['identity_type'] = main_questions['question_text'].apply(determine_identity_type)
        
        # Split into identity and non-identity questions
        identity_questions = main_questions[main_questions['is_identity'] == True].copy()
        non_identity_questions = main_questions[main_questions['is_identity'] == False].copy()
        
        # Prepare non-identity export (Table 1)
        export_df_non_identity = pd.DataFrame()
        if not non_identity_questions.empty:
            export_df_non_identity = non_identity_questions[[
                'question_uid', 'question_text', 'schema_type', 'Final_UID', 'required'
            ]].copy()
            export_df_non_identity.columns = ['question_id', 'question_text', 'question_type', 'UID', 'required']
        
        # Prepare identity export (Table 2)
        export_df_identity = pd.DataFrame()
        if not identity_questions.empty:
            export_df_identity = identity_questions[[
                'question_uid', 'question_text', 'schema_type', 'identity_type', 'Final_UID', 'required'
            ]].copy()
            export_df_identity.columns = ['question_id', 'question_text', 'question_type', 'identity_type', 'UID', 'required']
        
        return export_df_non_identity, export_df_identity
        
    except Exception as e:
        logger.error(f"Failed to prepare export data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def upload_to_snowflake_tables(export_df_non_identity, export_df_identity):
    """Upload both export tables to Snowflake"""
    try:
        engine = get_snowflake_engine()
        
        with st.spinner("🚀 Uploading tables to Snowflake..."):
            # Upload non-identity questions
            if not export_df_non_identity.empty:
                table_name_1 = f"uid_matcher_non_identity_{uuid4().hex[:8]}"
                export_df_non_identity.to_sql(
                    table_name_1, 
                    engine, 
                    if_exists='replace', 
                    index=False,
                    method='multi'
                )
                st.success(f"✅ Non-identity questions uploaded to: {table_name_1}")
            
            # Upload identity questions
            if not export_df_identity.empty:
                table_name_2 = f"uid_matcher_identity_{uuid4().hex[:8]}"
                export_df_identity.to_sql(
                    table_name_2, 
                    engine, 
                    if_exists='replace', 
                    index=False,
                    method='multi'
                )
                st.success(f"✅ Identity questions uploaded to: {table_name_2}")
            
            st.success("🎉 Both tables uploaded successfully to Snowflake!")
        
    except Exception as e:
        logger.error(f"Failed to upload to Snowflake: {e}")
        st.error(f"❌ Failed to upload to Snowflake: {str(e)}")

# ============= INITIALIZATION =============
# Initialize session state
initialize_session_state()

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
        st.markdown("• question_text → SurveyMonkey question/choice text")
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
    
      # Survey Categorization
    st.markdown("**📂 Survey Categorization**")
    if st.button("📊 Survey Categories", use_container_width=True):
        st.session_state.page = "survey_categorization"
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
            st.session_state.question_bank = pd.DataFrame(columns=["HEADING_0", "UID"])

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
        if sf_status and st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
            unique_uids = st.session_state.question_bank["UID"].nunique()
            st.metric("❄️ Snowflake UIDs", f"{unique_uids:,}")
        else:
            st.metric("❄️ Snowflake UIDs", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 SM Surveys", len(surveys))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if sf_status and surveys:
            try:
                configured_count = count_configured_surveys_from_surveymonkey(surveys)
                st.metric("🎯 Configured Surveys", f"{configured_count}")
            except:
                st.metric("🎯 Configured Surveys", "Error")
        else:
            st.metric("🎯 Configured Surveys", "No Connection")
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
        st.markdown("### 2️⃣ Survey Categorization")
        st.markdown("Categorize by survey purpose:")
        st.markdown("• Filter by survey categories")
        st.markdown("• Group by question types")
        st.markdown("• Assign UIDs by category")
        
        if st.button("📊 Start Survey Categories", use_container_width=True):
            st.session_state.page = "survey_categorization"
            st.rerun()
    
    with col3:
        st.markdown("### 3️⃣ UID Matching")
        st.markdown("Match questions to UIDs:")
        st.markdown("• Optimized 1:1 matching")
        st.markdown("• Configure assignments")
        st.markdown("• Resolve conflicts")
        
        if st.button("🔧 Start UID Matching", use_container_width=True):
            st.session_state.page = "uid_matching"
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
            ]["question_text"].unique().tolist())
            st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == True
            ]["question_text"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
            
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
                display_columns = ["question_uid", "question_text", "schema_type", "is_choice", "survey_title"]
                available_columns = [col for col in display_columns if col in display_df.columns]
                st.dataframe(display_df[available_columns], height=400)
                
                # Next step buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Proceed to Survey Categories", type="primary", use_container_width=True):
                        st.session_state.page = "survey_categorization"
                        st.rerun()
                with col2:
                    if st.button("🔧 Proceed to UID Matching", use_container_width=True):
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

elif st.session_state.page == "survey_categorization":
    st.markdown("## 📊 Survey Categorization")
    st.markdown('<div class="data-source-info">📂 <strong>Data Source:</strong> SurveyMonkey questions/choices - Independent categorization using cached survey data</div>', unsafe_allow_html=True)
    
    # Category overview
    st.markdown("### 📂 Survey Categories Overview")
    
    with st.expander("📋 Category Definitions", expanded=False):
        for category, keywords in SURVEY_CATEGORIES.items():
            st.markdown(f"**{category}:** {', '.join(keywords)}")
        st.markdown("**Uncategorized:** Surveys that don't match any category")
    
    # Generate categorized questions from ALL cached survey data
    with st.spinner("📊 Analyzing all survey categories (independent of selection)..."):
        categorized_df = get_unique_questions_by_category()
    
    if categorized_df.empty:
        st.markdown('<div class="warning-card">⚠️ No survey data available for categorization.</div>', unsafe_allow_html=True)
        st.markdown("**This page is now truly independent and will:**")
        st.markdown("• First try to load cached survey data")
        st.markdown("• If no cache exists, automatically fetch surveys from SurveyMonkey")
        st.markdown("• Process all available surveys for categorization")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Force Refresh All Survey Data", use_container_width=True):
                # Clear all cached data and force fresh fetch
                st.session_state.all_questions = None
                st.session_state.dedup_questions = []
                st.session_state.dedup_choices = []
                st.session_state.fetched_survey_ids = []
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                st.rerun()
        
        with col2:
            if st.button("📋 Go to Survey Selection (Optional)", use_container_width=True):
                st.session_state.page = "survey_selection"
                st.rerun()
        st.stop()
    
    # Show data cleaning info
    st.markdown("### 🧹 Data Cleaning Applied")
    with st.expander("ℹ️ Question Text Cleaning Rules", expanded=False):
        st.markdown("**Automatic cleaning applied to questions:**")
        st.markdown("• Removed year specifications like `(i.e. 1 Jan. 2024 - 31 Dec. 2024)`")
        st.markdown("• For mobile number questions: Extracted main question `Your mobile number`")
        st.markdown("• Removed HTML formatting tags")
        st.markdown("• Grouped similar questions together")
        st.markdown("• **Example:** `Your mobile number <br>Country area code - +255 (Tanzania)` → `Your mobile number`")
    
    # Category metrics
    st.markdown("### 📊 Category Metrics")
    
    category_stats = categorized_df.groupby('survey_category').agg({
        'question_text': 'count',
        'survey_count': 'sum'
    }).rename(columns={'question_text': 'question_count'}).reset_index()
    
    # Display metrics in columns
    cols = st.columns(min(len(category_stats), 4))
    for idx, (_, row) in enumerate(category_stats.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"📂 {row['survey_category']}", 
                f"{row['question_count']} questions",
                f"From {row['survey_count']} surveys"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Category filter and display
    st.markdown("### 🔍 Questions by Survey Category")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        survey_category_filter = st.multiselect(
            "Filter by survey category:",
            list(SURVEY_CATEGORIES.keys()) + ["Uncategorized"],
            default=list(SURVEY_CATEGORIES.keys()) + ["Uncategorized"],
            key="cat_survey_category_filter"
        )
    
    with col2:
        schema_filter = st.multiselect(
            "Filter by question type:",
            ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
            default=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
            key="cat_schema_filter"
        )
    
    with col3:
        question_type_filter = st.selectbox(
            "Filter by question classification:",
            ["All", "Main Question", "Choice", "Heading"],
            index=0,
            key="cat_question_type_filter"
        )
    
    # Apply filters
    filtered_df = categorized_df.copy()
    
    if survey_category_filter:
        filtered_df = filtered_df[filtered_df['survey_category'].isin(survey_category_filter)]
    
    if schema_filter:
        filtered_df = filtered_df[filtered_df['schema_type'].isin(schema_filter)]
    
    if question_type_filter != "All":
        if question_type_filter == "Main Question":
            filtered_df = filtered_df[filtered_df['is_choice'] == False]
            if 'question_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['question_category'] != "Heading"]
        elif question_type_filter == "Choice":
            filtered_df = filtered_df[filtered_df['is_choice'] == True]
        elif question_type_filter == "Heading":
            if 'question_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['question_category'] == "Heading"]
            else:
                filtered_df['temp_classification'] = filtered_df['question_text'].apply(classify_question)
                filtered_df = filtered_df[filtered_df['temp_classification'] == "Heading"]
                filtered_df = filtered_df.drop('temp_classification', axis=1)
    
    # UID Assignment Section
    if not filtered_df.empty:
        st.markdown("### 🔧 UID Assignment")
        
        # Prepare UID options
        uid_options = [None]
        if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
            uid_options.extend([f"{row['UID']} - {row['HEADING_0']}" for _, row in st.session_state.question_bank.iterrows()])
        
        # Display and edit questions
        display_columns = [
            "survey_category", "question_text", "schema_type", 
            "is_choice", "survey_count", "Final_UID", "Change_UID", "required"
        ]
        
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        edited_categorized_df = st.data_editor(
            filtered_df[available_columns],
            column_config={
                "survey_category": st.column_config.TextColumn("Survey Category", width="medium"),
                "question_text": st.column_config.TextColumn("Question/Choice", width="large"),
                "schema_type": st.column_config.TextColumn("Type", width="medium"),
                "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                "survey_count": st.column_config.NumberColumn("Survey Count", width="small"),
                "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                "Change_UID": st.column_config.SelectboxColumn(
                    "Assign UID",
                    options=uid_options,
                    default=None,
                    width="large"
                ),
                "required": st.column_config.CheckboxColumn("Required", width="small")
            },
            disabled=["survey_category", "question_text", "schema_type", "is_choice", "survey_count", "Final_UID"],
            hide_index=True,
            height=500,
            key="categorized_editor"
        )
        
        # Process UID changes
        uid_changes_made = False
        for idx, row in edited_categorized_df.iterrows():
            if pd.notnull(row.get("Change_UID")):
                new_uid = row["Change_UID"].split(" - ")[0]
                categorized_df.at[idx, "Final_UID"] = new_uid
                categorized_df.at[idx, "configured_final_UID"] = new_uid
                uid_changes_made = True
        
        # Action buttons
        st.markdown("### 🚀 Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save UID Assignments", use_container_width=True):
                if uid_changes_made:
                    st.session_state.df_final = categorized_df.copy()
                    st.session_state.df_target = categorized_df.copy()
                    st.session_state.categorized_questions = categorized_df.copy()
                    st.markdown('<div class="success-card">✅ UID assignments saved successfully!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-card">⚠️ No UID changes to save.</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("🔧 Proceed to UID Matching", use_container_width=True):
                st.session_state.df_target = categorized_df.copy()
                st.session_state.page = "uid_matching"
                st.rerun()
        
        with col3:
            if st.button("📥 Export Category Data", use_container_width=True):
                csv_data = categorized_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"categorized_questions_{uuid4()}.csv",
                    "text/csv",
                    key="cat_download"
                )
        
        # Summary by survey category
        st.markdown("### 📊 Assignment Summary by Survey Category")
        
        assignment_summary = categorized_df.groupby('survey_category').agg({
            'question_text': 'count',
            'Final_UID': lambda x: x.notna().sum()
        }).rename(columns={
            'question_text': 'Total Questions',
            'Final_UID': 'Assigned UIDs'
        })
        assignment_summary['Assignment Rate %'] = (
            assignment_summary['Assigned UIDs'] / assignment_summary['Total Questions'] * 100
        ).round(2)
        
        st.dataframe(assignment_summary, use_container_width=True)
        
    else:
        st.info("ℹ️ No questions match the selected filters")
    
    # Survey title analysis
    if st.expander("📋 Survey Title Analysis", expanded=False):
        st.markdown("### 📊 How Surveys Were Categorized")
        
        if st.session_state.all_questions is not None:
            survey_analysis = st.session_state.all_questions.groupby(['survey_title', 'survey_id']).first().reset_index()
            survey_analysis['survey_category'] = survey_analysis['survey_title'].apply(categorize_survey_by_title)
            
            st.dataframe(
                survey_analysis[['survey_title', 'survey_category', 'survey_id']],
                column_config={
                    "survey_title": st.column_config.TextColumn("Survey Title", width="large"),
                    "survey_category": st.column_config.TextColumn("Assigned Category", width="medium"),
                    "survey_id": st.column_config.TextColumn("Survey ID", width="medium")
                },
                use_container_width=True,
                height=300
            )

elif st.session_state.page == "question_bank":
    st.markdown("## 📚 Question Bank Viewer")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Snowflake - Current question bank with authority counts</div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("📊 Loading question bank from Snowflake..."):
            question_bank_with_authority = get_question_bank_with_authority_count()
        
        if question_bank_with_authority.empty:
            st.warning("⚠️ No question bank data available from Snowflake")
            st.stop()
        
        st.session_state.question_bank_with_authority = question_bank_with_authority
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_records = len(question_bank_with_authority)
            st.metric("📊 Total Records", f"{total_records:,}")
        with col2:
            unique_uids = question_bank_with_authority['UID'].nunique()
            st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        with col3:
            unique_questions = question_bank_with_authority['HEADING_0'].nunique()
            st.metric("📝 Unique Questions", f"{unique_questions:,}")
        with col4:
            conflicts = total_records - unique_uids
            st.metric("⚠️ Conflicts", f"{conflicts:,}")
        
        # Filters
        st.markdown("### 🔍 Filter and Search")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uid_filter = st.text_input("🆔 Filter by UID (exact match):")
        with col2:
            min_authority = st.number_input("📊 Minimum Authority Count:", min_value=1, value=1)
        with col3:
            search_text = st.text_input("🔍 Search question text:")
        
        # Apply filters
        filtered_df = question_bank_with_authority.copy()
        
        if uid_filter:
            filtered_df = filtered_df[filtered_df['UID'].astype(str) == uid_filter]
        
        if min_authority > 1:
            filtered_df = filtered_df[filtered_df['AUTHORITY_COUNT'] >= min_authority]
        
        if search_text:
            filtered_df = filtered_df[
                filtered_df['HEADING_0'].str.contains(search_text, case=False, na=False)
            ]
        
        # Display results
        st.markdown(f"### 📋 Question Bank ({len(filtered_df):,} records)")
        
        if not filtered_df.empty:
            # Sort by UID, then by authority count
            display_df = filtered_df.sort_values(['UID', 'AUTHORITY_COUNT'], ascending=[True, False])
            
            st.dataframe(
                display_df,
                column_config={
                    "UID": st.column_config.NumberColumn("UID", width="small"),
                    "HEADING_0": st.column_config.TextColumn("Question Text", width="large"),
                    "AUTHORITY_COUNT": st.column_config.NumberColumn("Authority Count", width="medium")
                },
                use_container_width=True,
                height=500
            )
            
            # Download option
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Question Bank",
                csv_data,
                f"question_bank_filtered_{uuid4()}.csv",
                "text/csv"
            )
            
            # Conflict analysis
            if len(display_df) > display_df['UID'].nunique():
                st.markdown("### ⚠️ UID Conflicts Detected")
                
                conflict_uids = display_df.groupby('UID').size()
                conflict_uids = conflict_uids[conflict_uids > 1]
                
                if not conflict_uids.empty:
                    st.write(f"**Found {len(conflict_uids)} UIDs with multiple questions:**")
                    
                    for uid in conflict_uids.index[:10]:  # Show first 10 conflicts
                        uid_questions = display_df[display_df['UID'] == uid]
                        with st.expander(f"UID {uid} - {len(uid_questions)} questions"):
                            for _, row in uid_questions.iterrows():
                                st.write(f"• **Authority: {row['AUTHORITY_COUNT']}** - {row['HEADING_0'][:100]}...")
        else:
            st.info("ℹ️ No questions match the selected filters")
    
    except Exception as e:
        logger.error(f"Failed to load question bank: {e}")
        st.error(f"❌ Failed to load question bank: {str(e)}")

elif st.session_state.page == "uid_matching":
    st.markdown("## 🔧 UID Matching & Configuration")
    st.markdown('<div class="data-source-info">🔄 <strong>Process:</strong> Match survey questions → Snowflake references → Assign UIDs</div>', unsafe_allow_html=True)
    
    if st.session_state.df_target is None or st.session_state.df_target.empty:
        st.markdown('<div class="warning-card">⚠️ No survey data selected. Please select surveys first.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Go to Survey Selection"):
                st.session_state.page = "survey_selection"
                st.rerun()
        with col2:
            if st.button("📊 Go to Survey Categories"):
                st.session_state.page = "survey_categorization"
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
            result_df = result_df[result_df["question_text"].str.contains(search_query, case=False, na=False)]
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
                uid_options.extend([f"{row['UID']} - {row['HEADING_0']}" for _, row in st.session_state.question_bank.iterrows()])
            
            # Create required column if it doesn't exist
            if "required" not in result_df.columns:
                result_df["required"] = False
            
            display_columns = ["question_uid", "question_text", "schema_type", "is_choice"]
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
                    "question_text": st.column_config.TextColumn("Question/Choice", width="large"),
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
                disabled=["question_uid", "question_text", "schema_type", "is_choice", "Final_UID"],
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
        
        # Prepare export data - now returns two tables
        export_df_non_identity, export_df_identity = prepare_export_data(st.session_state.df_final)
        
        if not export_df_non_identity.empty or not export_df_identity.empty:
            
            # Show metrics for both tables
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Non-Identity Questions", len(export_df_non_identity))
            with col2:
                st.metric("🔐 Identity Questions", len(export_df_identity))
            with col3:
                total_records = len(export_df_non_identity) + len(export_df_identity)
                st.metric("📋 Total Records", total_records)
            
            # Preview both tables
            st.markdown("#### 👁️ Preview Data for Export")
            
            # Non-Identity Questions Preview
            if not export_df_non_identity.empty:
                st.markdown("**📊 Non-Identity Questions (Table 1)**")
                st.dataframe(export_df_non_identity.head(10), use_container_width=True)
            
            # Identity Questions Preview  
            if not export_df_identity.empty:
                st.markdown("**🔐 Identity Questions (Table 2)**")
                st.dataframe(export_df_identity.head(10), use_container_width=True)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not export_df_non_identity.empty:
                    csv_data_non_identity = export_df_non_identity.to_csv(index=False)
                    st.download_button(
                        "📥 Download Non-Identity CSV",
                        csv_data_non_identity,
                        f"non_identity_questions_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if not export_df_identity.empty:
                    csv_data_identity = export_df_identity.to_csv(index=False)
                    st.download_button(
                        "📥 Download Identity CSV", 
                        csv_data_identity,
                        f"identity_questions_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("🚀 Upload Both Tables to Snowflake", use_container_width=True):
                    upload_to_snowflake_tables(export_df_non_identity, export_df_identity)

        else:
            st.warning("⚠️ No data available for export")

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
            unique_uids = df_reference['UID'].nunique()
            st.metric("🆔 Unique UIDs", f"{unique_uids:,}")
        with col3:
            unique_questions = df_reference['HEADING_0'].nunique()
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
                st.write(f"**Winner UID:** {row['winner_UID']} ({row['winner_count']} occurrences, {row['winner_percentage']:.1f}%)")
                st.write(f"**Top Competitor:** UID {row['top_competitor_UID']} ({row['top_competitor_count']} occurrences, {row['top_competitor_percentage']:.1f}%)")
                st.write(f"**Authority Difference:** {row['authority_difference']:.1f}%")
                st.write(f"**Conflict Severity:** {row['conflict_severity']} competing records")
                if row['is_high_conflict']:
                    st.write("🚨 **HIGH SEVERITY CONFLICT**")
    
    else:
        st.info("ℹ️ No conflicts match the selected filters")

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
            st.session_state.edited_df = pd.DataFrame(columns=["question_text", "schema_type", "is_choice", "required"])

        edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config={
                "question_text": st.column_config.SelectboxColumn(
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
            non_standard = edited_df[~edited_df["question_text"].isin(st.session_state.question_bank["HEADING_0"])]
            if not non_standard.empty:
                st.markdown('<div class="warning-card">⚠️ Non-standard questions detected:</div>', unsafe_allow_html=True)
                st.dataframe(non_standard[["question_text"]], use_container_width=True)
                st.markdown("[📝 Submit New Questions](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
            else:
                st.markdown('<div class="success-card">✅ All questions are validated!</div>', unsafe_allow_html=True)
        
        if preview_btn or create_btn:
            if not survey_title or edited_df.empty:
                st.markdown('<div class="warning-card">⚠️ Survey title and questions are required.</div>', unsafe_allow_html=True)
            else:
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
    
    # Show configured surveys count
    if sf_status and surveys:
        try:
            configured_count = count_configured_surveys_from_surveymonkey(surveys)
            st.write(f"Configured: {configured_count}")
        except:
            st.write("Configured: Error")

# ============= END OF SCRIPT =============


