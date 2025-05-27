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
    
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.125rem;
    }
    
    .category-application { background: #e3f2fd; color: #1976d2; }
    .category-pre-programme { background: #f3e5f5; color: #7b1fa2; }
    .category-enrollment { background: #e8f5e8; color: #388e3c; }
    .category-progress { background: #fff3e0; color: #f57c00; }
    .category-impact { background: #ffebee; color: #d32f2f; }
    .category-grow { background: #fce4ec; color: #c2185b; }
    .category-feedback { background: #e0f2f1; color: #00695c; }
    .category-pulse { background: #f1f8e9; color: #689f38; }
    .category-other { background: #f5f5f5; color: #616161; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.65  # Increased for better accuracy
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# Enhanced UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'standardization_enabled': True,
    'category_based_assignment': True
}

# Enhanced Survey Categories with more keywords
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join', 'register', 'applicant'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline', 'pre-program', 'pre program'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start', 'enrolled', 'joining'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment', 'evaluation', 'progress review'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation', 'impact assessment'],
    'GROW': ['GROW'],  # Exact match for CAPS
    'Feedback': ['feedback', 'evaluation', 'rating', 'satisfaction', 'opinion', 'survey feedback'],
    'Pulse': ['pulse', 'quick', 'brief', 'snapshot', 'check-in', 'pulse survey']
}

# Enhanced Synonym Mapping for standardization
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
    "which of the following": "what",
    "select all that apply": "what applies",
    "choose one": "what is",
    "pick one": "what is",
    "indicate your": "what is your",
    "specify your": "what is your"
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# Enhanced Question Standardization
def standardize_question_format(question_text):
    """
    Standardize question format before UID assignment
    """
    if not question_text or pd.isna(question_text):
        return ""
    
    text = str(question_text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize common question formats
    text = re.sub(r'^please\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^kindly\s+', '', text, flags=re.IGNORECASE)
    
    # Standardize punctuation
    if text and not text.endswith(('?', '.', '!')):
        if any(word in text.lower().split()[:3] for word in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'are', 'is', 'can', 'will']):
            text += '?'
        else:
            text += '.'
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

# Enhanced Survey Categorization
def categorize_survey(survey_title):
    """
    Enhanced categorization with better keyword matching
    """
    if not survey_title:
        return "Unknown"
    
    title_lower = survey_title.lower()
    
    # Check GROW first (exact match for CAPS)
    if 'GROW' in survey_title:
        return 'GROW'
    
    # Check other categories with weighted scoring
    category_scores = {}
    
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == 'GROW':  # Already checked
            continue
        
        score = 0
        for keyword in keywords:
            if keyword.lower() in title_lower:
                # Give higher score for exact word matches
                if keyword.lower() in title_lower.split():
                    score += 2
                else:
                    score += 1
        
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score
    if category_scores:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    return "Other"

# Enhanced Semantic UID Assignment with Governance (Updated for Snowflake data)
def enhanced_semantic_matching_with_governance(question_text, existing_uids_data, category=None, threshold=0.75):
    """
    Enhanced semantic matching with governance rules and category awareness
    Note: existing_uids_data structure updated for Snowflake data
    """
    if not existing_uids_data:
        return None, 0.0, "new_assignment"
    
    try:
        model = load_sentence_transformer()
        standardized_question = standardize_question_format(question_text)
        
        # Get embeddings
        question_embedding = model.encode([standardized_question], convert_to_tensor=True)
        
        # Filter existing questions by category if provided and available
        if category and category != "Unknown":
            category_filtered_uids = {
                uid: data for uid, data in existing_uids_data.items() 
                if data.get('category') == category
            }
            if category_filtered_uids:
                existing_uids_data = category_filtered_uids
        
        existing_questions = [data['best_question'] for data in existing_uids_data.values()]
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(question_embedding, existing_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        if best_score >= threshold:
            best_uid = list(existing_uids_data.keys())[best_idx]
            
            # Check governance compliance
            current_variations = existing_uids_data[best_uid].get('variation_count', 0)
            if current_variations < UID_GOVERNANCE['max_variations_per_uid']:
                return best_uid, best_score, "semantic_match"
            else:
                logger.warning(f"UID {best_uid} exceeds max variations, suggesting new UID")
                return None, best_score, "governance_violation"
        
        return None, best_score, "below_threshold"
            
    except Exception as e:
        logger.error(f"Enhanced semantic matching failed: {e}")
        return None, 0.0, "error"

# Enhanced UID Assignment with Category and Governance (Updated for Snowflake data)
def assign_uid_with_enhanced_governance(question_text, existing_uids_data, survey_category=None):
    """
    Assign UID with enhanced governance rules, category awareness, and standardization
    Note: Updated to work with SurveyMonkey questions vs Snowflake HEADING_0 data
    """
    # Standardize question format first
    standardized_question = standardize_question_format(question_text)
    
    # Try semantic matching with category preference (if we have category info)
    matched_uid, confidence, match_type = enhanced_semantic_matching_with_governance(
        standardized_question, existing_uids_data, survey_category
    )
    
    if matched_uid and match_type == "semantic_match":
        return {
            'uid': matched_uid,
            'method': 'semantic_match',
            'confidence': confidence,
            'governance_compliant': True,
            'category': survey_category,
            'standardized_question': standardized_question
        }
    
    # If no match or governance violation, create new UID
    if existing_uids_data:
        max_uid = max([int(uid) for uid in existing_uids_data.keys() if uid.isdigit()])
        new_uid = str(max_uid + 1)
    else:
        new_uid = "1"
    
    return {
        'uid': new_uid,
        'method': 'new_assignment',
        'confidence': 1.0 if match_type == "below_threshold" else 0.8,
        'governance_compliant': True,
        'category': survey_category,
        'standardized_question': standardized_question,
        'reason': f"No suitable match found (best score: {confidence:.3f})" if match_type == "below_threshold" else "Governance compliance"
    }

# Cached Resources
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
            st.warning(
                "🔒 Snowflake connection failed: User account is locked. "
                "UID matching is disabled, but you can edit questions, search, and use Google Forms."
            )
        raise

@st.cache_data
def get_all_reference_questions():
    """Fetch ALL reference questions from Snowflake - only HEADING_0 and UID"""
    all_data = []
    limit = 10000
    offset = 0
    
    while True:
        query = """
            SELECT HEADING_0, UID
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
            
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed: {e}")
            raise
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Enhanced Unique Questions Bank Creation (Updated for Snowflake data)
def create_enhanced_unique_questions_bank(df_reference):
    """
    Create enhanced unique questions bank with categories and governance
    Note: df_reference only contains HEADING_0 and UID from Snowflake
    """
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_reference)} reference questions for enhanced unique bank")
    
    unique_questions = []
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        uid_questions = group['heading_0'].tolist()
        
        # Standardize and score questions
        scored_questions = []
        for q in uid_questions:
            standardized = standardize_question_format(q)
            score = score_question_quality(standardized)
            scored_questions.append((q, standardized, score))
        
        # Get best question
        best_question_data = max(scored_questions, key=lambda x: x[2])
        best_original, best_standardized, best_score = best_question_data
        
        # Since we don't have survey titles from Snowflake, we'll categorize based on question content
        question_category = categorize_question_by_content(best_original)
        
        unique_questions.append({
            'uid': uid,
            'best_question': best_original,
            'standardized_question': best_standardized,
            'total_variants': len(uid_questions),
            'question_length': len(str(best_original)),
            'question_words': len(str(best_original).split()),
            'survey_category': question_category,  # Categorized from question content
            'survey_titles': 'From Snowflake Data',  # Placeholder since we don't have survey titles
            'quality_score': best_score,
            'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid'],
            'all_categories': question_category,
            'category_count': 1,
            'all_variants': uid_questions
        })
    
    unique_df = pd.DataFrame(unique_questions)
    logger.info(f"Created enhanced unique questions bank with {len(unique_df)} UIDs")
    
    # Sort by UID
    if not unique_df.empty:
        try:
            unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
            unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
            unique_df = unique_df.drop('uid_numeric', axis=1)
        except:
            unique_df = unique_df.sort_values('uid')
    
    return unique_df

# New function to categorize questions by content (since we don't have survey titles from Snowflake)
def categorize_question_by_content(question_text):
    """
    Categorize questions based on their content when survey title is not available
    """
    if not question_text:
        return "Unknown"
    
    text_lower = str(question_text).lower()
    
    # Application-related keywords
    if any(keyword in text_lower for keyword in ['apply', 'application', 'register', 'join', 'signup', 'eligibility']):
        return 'Application'
    
    # Pre-programme keywords
    if any(keyword in text_lower for keyword in ['baseline', 'preparation', 'readiness', 'before', 'pre-']):
        return 'Pre programme'
    
    # Enrollment keywords
    if any(keyword in text_lower for keyword in ['enrollment', 'enroll', 'onboard', 'welcome', 'start']):
        return 'Enrollment'
    
    # Progress keywords
    if any(keyword in text_lower for keyword in ['progress', 'milestone', 'review', 'assessment', 'evaluation']):
        return 'Progress Review'
    
    # Impact keywords
    if any(keyword in text_lower for keyword in ['impact', 'outcome', 'result', 'change', 'improvement', 'benefit']):
        return 'Impact'
    
    # GROW (check for CAPS)
    if 'GROW' in question_text:
        return 'GROW'
    
    # Feedback keywords
    if any(keyword in text_lower for keyword in ['feedback', 'rating', 'satisfaction', 'opinion', 'rate']):
        return 'Feedback'
    
    # Pulse keywords
    if any(keyword in text_lower for keyword in ['pulse', 'quick', 'brief', 'check']):
        return 'Pulse'
    
    # Demographic/profile questions
    if any(keyword in text_lower for keyword in ['name', 'age', 'gender', 'location', 'company', 'role', 'position', 'department']):
        return 'Profile'
    
    return "Other"

# Enhanced question quality scoring
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
    
    # Standardization bonus
    if text[0].isupper() and (text.endswith('?') or text.endswith('.')):
        score += 5
    
    return score

# SurveyMonkey API functions (keeping existing ones)
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

def extract_questions(survey_json):
    questions = []
    global_position = 0
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            subtype = question.get("subtype", None)
            
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

def classify_question(text, heading_references=HEADING_REFERENCES):
    # Length-based heuristic
    if len(text.split()) > HEADING_LENGTH_THRESHOLD:
        return "Heading"
    
    # Check against heading references
    try:
        model = load_sentence_transformer()
        emb_text = model.encode([text], convert_to_tensor=True)
        emb_refs = model.encode(heading_references, convert_to_tensor=True)
        semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
        max_semantic_score = np.max(semantic_scores.cpu().numpy())
        
        if max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
            return "Heading"
    except Exception as e:
        logger.error(f"Classification failed: {e}")
    
    return "Main Question/Multiple Choice"

# Enhanced matching functions
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    # Apply enhanced synonym mapping
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

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
if "unique_questions_bank" not in st.session_state:
    st.session_state.unique_questions_bank = None

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 UID Matcher Pro")
    st.markdown("Enhanced with Semantic Matching & Governance")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # SurveyMonkey section
    st.markdown("**📊 SurveyMonkey**")
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    if st.button("⚙️ Configure Survey (Enhanced)", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()
    if st.button("📊 Survey Categorization", use_container_width=True):
        st.session_state.page = "survey_categorization"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank section
    st.markdown("**📚 Question Bank**")
    if st.button("⭐ Enhanced Unique Bank", use_container_width=True):
        st.session_state.page = "unique_question_bank"
        st.rerun()
    if st.button("📊 Categorized Questions", use_container_width=True):
        st.session_state.page = "categorized_questions"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Governance section
    st.markdown("**⚖️ Enhanced Governance**")
    st.markdown(f"• Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.markdown(f"• Standardization: {'✅' if UID_GOVERNANCE['standardization_enabled'] else '❌'}")
    st.markdown(f"• Category-based: {'✅' if UID_GOVERNANCE['category_based_assignment'] else '❌'}")

# App Header
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Semantic Matching & Governance</div>', unsafe_allow_html=True)

# Home Page
if st.session_state.page == "home":
    st.markdown("## 🏠 Enhanced UID Matcher Pro Dashboard")
    
    # Enhanced features showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Enhanced Features")
        st.markdown("• **Semantic UID Matching**: AI-powered question similarity")
        st.markdown("• **Question Standardization**: Automatic format normalization")
        st.markdown("• **Governance Rules**: Max variations per UID enforcement")
        st.markdown("• **Survey Categorization**: Auto-categorize by survey titles")
        st.markdown("• **Category-based Assignment**: Smart UID assignment by category")
        
    with col2:
        st.markdown("### 📊 Survey Categories")
        for category in SURVEY_CATEGORIES.keys():
            badge_class = f"category-{category.lower().replace(' ', '-')}"
            st.markdown(f'<span class="category-badge {badge_class}">{category}</span>', unsafe_allow_html=True)

# Enhanced Configure Survey Page
elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Enhanced Survey Configuration with Semantic Matching")
    st.markdown("*Configure surveys with enhanced UID matching, governance rules, and standardization*")
    
    try:
        token = st.secrets["surveymonkey"]["token"]
        surveys = get_surveys(token)
        
        if not surveys:
            st.warning("⚠️ No surveys found in your SurveyMonkey account.")
        else:
            # Load unique questions bank for semantic matching
            if st.session_state.unique_questions_bank is None:
                with st.spinner("🔄 Loading enhanced question bank for semantic matching..."):
                    df_reference = get_all_reference_questions()
                    st.session_state.unique_questions_bank = create_enhanced_unique_questions_bank(df_reference)
            
            # Survey selection with category detection
            st.markdown("### 📋 Select Survey for Enhanced Configuration")
            
            survey_options = []
            for survey in surveys:
                title = survey.get("title", "Untitled")
                survey_id = survey.get("id", "")
                category = categorize_survey(title)
                
                # Add category badge to title
                badge_class = f"category-{category.lower().replace(' ', '-')}"
                display_title = f"{title} [{category}]"
                survey_options.append((display_title, survey_id, title, category))
            
            # Sort by category then by title
            survey_options.sort(key=lambda x: (x[3], x[2]))
            
            selected_survey = st.selectbox(
                "Choose survey to configure:",
                options=survey_options,
                format_func=lambda x: x[0]
            )
            
            if selected_survey:
                survey_id = selected_survey[1]
                survey_title = selected_survey[2]
                survey_category = selected_survey[3]
                
                st.markdown(f"**Selected Survey:** {survey_title}")
                st.markdown(f"**Detected Category:** `{survey_category}`")
                
                # Enhanced configuration options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    semantic_threshold = st.slider(
                        "🧠 Semantic Matching Threshold", 
                        0.5, 0.95, 0.75, 0.05,
                        help="Higher values require more similarity for UID matching"
                    )
                
                with col2:
                    enable_standardization = st.checkbox(
                        "📝 Enable Question Standardization", 
                        value=True,
                        help="Standardize question formats before matching"
                    )
                
                with col3:
                    category_preference = st.checkbox(
                        "📊 Prefer Category-based Matching", 
                        value=True,
                        help="Prioritize matching within the same survey category"
                    )
                
                if st.button("⚙️ Configure with Enhanced Matching", type="primary"):
                    with st.spinner(f"🔄 Configuring survey with enhanced semantic matching..."):
                        # Get survey details
                        survey_details = get_survey_details(survey_id, token)
                        questions_data = extract_questions(survey_details)
                        
                        if not questions_data:
                            st.error("❌ No questions found in this survey.")
                        else:
                            df_questions = pd.DataFrame(questions_data)
                            
                            # Add survey category to all questions
                            df_questions['survey_category'] = survey_category
                            
                            # Prepare existing UIDs data for semantic matching
                            # Note: This comes from Snowflake (HEADING_0, UID) and we categorize by question content
                            unique_bank = st.session_state.unique_questions_bank
                            existing_uids_data = {}
                            
                            if not unique_bank.empty:
                                for _, row in unique_bank.iterrows():
                                    existing_uids_data[row['uid']] = {
                                        'best_question': row['standardized_question'] if enable_standardization else row['best_question'],
                                        'variation_count': row['total_variants'],
                                        'category': row['survey_category'],  # This is categorized from question content
                                        'quality_score': row['quality_score']
                                    }
                            
                            # Enhanced UID assignment
                            enhanced_assignments = []
                            assignment_stats = {
                                'semantic_matches': 0,
                                'new_assignments': 0,
                                'governance_violations': 0,
                                'standardized_questions': 0
                            }
                            
                            for _, question_row in df_questions.iterrows():
                                if question_row['is_choice'] or question_row['question_category'] == "Heading":
                                    # Skip choices and headings for direct UID assignment
                                    enhanced_assignments.append({
                                        'original_question': question_row['heading_0'],
                                        'standardized_question': question_row['heading_0'],
                                        'assigned_uid': None,
                                        'method': 'skipped',
                                        'confidence': 0.0,
                                        'category': survey_category,
                                        'governance_compliant': True
                                    })
                                    continue
                                
                                # Get enhanced UID assignment
                                assignment_result = assign_uid_with_enhanced_governance(
                                    question_row['heading_0'],
                                    existing_uids_data,
                                    survey_category if category_preference else None
                                )
                                
                                enhanced_assignments.append({
                                    'original_question': question_row['heading_0'],
                                    'standardized_question': assignment_result['standardized_question'],
                                    'assigned_uid': assignment_result['uid'],
                                    'method': assignment_result['method'],
                                    'confidence': assignment_result['confidence'],
                                    'category': assignment_result['category'],
                                    'governance_compliant': assignment_result['governance_compliant'],
                                    'reason': assignment_result.get('reason', '')
                                })
                                
                                # Update stats
                                if assignment_result['method'] == 'semantic_match':
                                    assignment_stats['semantic_matches'] += 1
                                elif assignment_result['method'] == 'new_assignment':
                                    assignment_stats['new_assignments'] += 1
                                
                                if enable_standardization and assignment_result['standardized_question'] != question_row['heading_0']:
                                    assignment_stats['standardized_questions'] += 1
                            
                            # Create enhanced results dataframe
                            df_enhanced = df_questions.copy()
                            
                            # Add enhanced assignment data
                            for i, assignment in enumerate(enhanced_assignments):
                                if i < len(df_enhanced):
                                    df_enhanced.loc[i, 'Enhanced_UID'] = assignment['assigned_uid']
                                    df_enhanced.loc[i, 'Standardized_Question'] = assignment['standardized_question']
                                    df_enhanced.loc[i, 'Assignment_Method'] = assignment['method']
                                    df_enhanced.loc[i, 'Confidence_Score'] = assignment['confidence']
                                    df_enhanced.loc[i, 'Governance_Compliant'] = assignment['governance_compliant']
                                    df_enhanced.loc[i, 'Assignment_Reason'] = assignment.get('reason', '')
                            
                            # Handle choices (inherit from parent questions)
                            for i, row in df_enhanced.iterrows():
                                if row['is_choice'] and pd.notna(row['parent_question']):
                                    parent_uid = df_enhanced[df_enhanced['heading_0'] == row['parent_question']]['Enhanced_UID'].iloc[0] if len(df_enhanced[df_enhanced['heading_0'] == row['parent_question']]) > 0 else None
                                    df_enhanced.loc[i, 'Enhanced_UID'] = parent_uid
                                    df_enhanced.loc[i, 'Assignment_Method'] = 'inherited_from_parent'
                            
                            st.session_state.df_final = df_enhanced
                            
                            # Display enhanced results
                            st.success("✅ Enhanced UID assignment completed!")
                            
                            # Show assignment statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("🧠 Semantic Matches", assignment_stats['semantic_matches'])
                            with col2:
                                st.metric("🆕 New Assignments", assignment_stats['new_assignments'])
                            with col3:
                                st.metric("📝 Standardized", assignment_stats['standardized_questions'])
                            with col4:
                                governance_rate = len([a for a in enhanced_assignments if a['governance_compliant']]) / len(enhanced_assignments) * 100
                                st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
                            
                            st.markdown("---")
                            
                            # Enhanced results table
                            st.markdown("### 📊 Enhanced UID Assignment Results")
                            
                            # Filter and display options
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                show_method = st.selectbox(
                                    "Filter by method:",
                                    ["All", "semantic_match", "new_assignment", "skipped"]
                                )
                            
                            with col2:
                                show_choices = st.checkbox("Show choice questions", value=False)
                            
                            with col3:
                                show_standardized = st.checkbox("Show standardized questions", value=True)
                            
                            # Apply filters
                            display_df = df_enhanced.copy()
                            
                            if not show_choices:
                                display_df = display_df[display_df['is_choice'] == False]
                            
                            if show_method != "All":
                                display_df = display_df[display_df['Assignment_Method'] == show_method]
                            
                            # Prepare display columns
                            display_columns = {
                                'heading_0': 'Original Question',
                                'Enhanced_UID': 'Assigned UID',
                                'Assignment_Method': 'Method',
                                'Confidence_Score': 'Confidence',
                                'question_category': 'Type',
                                'survey_category': 'Category',
                                'Governance_Compliant': 'Governance'
                            }
                            
                            if show_standardized:
                                display_columns['Standardized_Question'] = 'Standardized Question'
                            
                            display_df_filtered = display_df[list(display_columns.keys())].copy()
                            display_df_filtered = display_df_filtered.rename(columns=display_columns)
                            
                            # Add icons for better visualization
                            display_df_filtered['Method'] = display_df_filtered['Method'].map({
                                'semantic_match': '🧠 Semantic',
                                'new_assignment': '🆕 New',
                                'skipped': '⏭️ Skipped',
                                'inherited_from_parent': '👨‍👩‍👧‍👦 Inherited'
                            })
                            
                            display_df_filtered['Governance'] = display_df_filtered['Governance'].map({
                                True: '✅',
                                False: '❌'
                            })
                            
                            st.dataframe(
                                display_df_filtered,
                                column_config={
                                    "Original Question": st.column_config.TextColumn("Original Question", width="large"),
                                    "Assigned UID": st.column_config.TextColumn("UID", width="small"),
                                    "Method": st.column_config.TextColumn("Method", width="medium"),
                                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
                                    "Type": st.column_config.TextColumn("Type", width="medium"),
                                    "Category": st.column_config.TextColumn("Category", width="medium"),
                                    "Governance": st.column_config.TextColumn("Gov", width="small"),
                                    "Standardized Question": st.column_config.TextColumn("Standardized Question", width="large") if show_standardized else None
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Download options
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.download_button(
                                    "📥 Download Enhanced Results",
                                    df_enhanced.to_csv(index=False),
                                    f"enhanced_survey_config_{survey_id}_{uuid4().hex[:8]}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Create assignment summary
                                summary_data = []
                                for method, count in Counter([a['method'] for a in enhanced_assignments]).items():
                                    summary_data.append({
                                        'Method': method,
                                        'Count': count,
                                        'Percentage': f"{count/len(enhanced_assignments)*100:.1f}%"
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.download_button(
                                    "📊 Download Assignment Summary",
                                    summary_df.to_csv(index=False),
                                    f"assignment_summary_{survey_id}_{uuid4().hex[:8]}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            
                            with col3:
                                if assignment_stats['standardized_questions'] > 0:
                                    standardized_df = pd.DataFrame([
                                        {
                                            'Original': a['original_question'],
                                            'Standardized': a['standardized_question']
                                        }
                                        for a in enhanced_assignments 
                                        if a['original_question'] != a['standardized_question']
                                    ])
                                    
                                    st.download_button(
                                        "📝 Download Standardizations",
                                        standardized_df.to_csv(index=False),
                                        f"standardizations_{survey_id}_{uuid4().hex[:8]}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                            
                            # Interactive UID editing
                            st.markdown("---")
                            st.markdown("### ✏️ Interactive UID Editing")
                            
                            if st.checkbox("Enable UID editing", value=False):
                                main_questions = df_enhanced[df_enhanced['is_choice'] == False].copy()
                                
                                for idx, row in main_questions.iterrows():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        st.text(f"{row['heading_0'][:100]}...")
                                    
                                    with col2:
                                        current_uid = row.get('Enhanced_UID', '')
                                        new_uid = st.text_input(
                                            f"UID for Q{idx+1}",
                                            value=str(current_uid) if pd.notna(current_uid) else "",
                                            key=f"uid_edit_{idx}"
                                        )
                                        
                                        if new_uid != str(current_uid):
                                            st.session_state.uid_changes[idx] = new_uid
                                    
                                    with col3:
                                        if row.get('Assignment_Method') == 'semantic_match':
                                            st.success("🧠")
                                        elif row.get('Assignment_Method') == 'new_assignment':
                                            st.info("🆕")
                                        else:
                                            st.warning("⏭️")
                                
                                if st.session_state.uid_changes:
                                    if st.button("💾 Save UID Changes"):
                                        for idx, new_uid in st.session_state.uid_changes.items():
                                            if idx < len(df_enhanced):
                                                df_enhanced.loc[idx, 'Enhanced_UID'] = new_uid
                                                df_enhanced.loc[idx, 'Assignment_Method'] = 'manual_edit'
                                        
                                        st.session_state.df_final = df_enhanced
                                        st.session_state.uid_changes = {}
                                        st.success("✅ UID changes saved!")
                                        st.rerun()
    
    except Exception as e:
        logger.error(f"Configure survey failed: {e}")
        st.error(f"❌ Error configuring survey: {e}")

# New Survey Categorization Page
elif st.session_state.page == "survey_categorization":
    st.markdown("## 📊 Survey Categorization & UID Assignment")
    st.markdown("*Categorize SurveyMonkey surveys and assign UIDs based on Snowflake HEADING_0 data*")
    
    # Data source explanation
    st.info("**📋 Data Sources:** Survey categories come from SurveyMonkey survey titles. Unique questions and UIDs come from Snowflake HEADING_0 data, categorized by question content.")
    
    try:
        # Load surveys and unique questions bank
        token = st.secrets["surveymonkey"]["token"]
        surveys = get_surveys(token)
        
        if st.session_state.unique_questions_bank is None:
            with st.spinner("🔄 Loading unique questions bank..."):
                df_reference = get_all_reference_questions()
                st.session_state.unique_questions_bank = create_enhanced_unique_questions_bank(df_reference)
        
        unique_bank = st.session_state.unique_questions_bank
        
        if not surveys:
            st.warning("⚠️ No surveys found in your SurveyMonkey account.")
        else:
            # Categorize all surveys
            categorized_surveys = []
            for survey in surveys:
                title = survey.get("title", "Untitled")
                survey_id = survey.get("id", "")
                category = categorize_survey(title)
                
                categorized_surveys.append({
                    'survey_id': survey_id,
                    'survey_title': title,
                    'category': category,
                    'created_at': survey.get('date_created', ''),
                    'modified_at': survey.get('date_modified', ''),
                    'num_responses': survey.get('response_count', 0)
                })
            
            df_surveys = pd.DataFrame(categorized_surveys)
            
            # Category overview
            st.markdown("### 📊 Survey Category Overview")
            
            category_stats = df_surveys.groupby('category').agg({
                'survey_id': 'count',
                'num_responses': 'sum'
            }).rename(columns={'survey_id': 'survey_count', 'num_responses': 'total_responses'})
            
            # Display category metrics
            categories = list(SURVEY_CATEGORIES.keys()) + ['Other', 'Unknown', 'Mixed']
            cols = st.columns(min(4, len(categories)))
            
            for i, category in enumerate(categories):
                if category in category_stats.index:
                    count = category_stats.loc[category, 'survey_count']
                    badge_class = f"category-{category.lower().replace(' ', '-')}"
                    
                    with cols[i % 4]:
                        st.markdown(f'<div class="metric-card"><span class="category-badge {badge_class}">{category}</span><br><strong>{count} surveys</strong></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Category filter and detailed view
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_category = st.selectbox(
                    "📊 Filter by Category:",
                    ["All Categories"] + sorted(df_surveys['category'].unique().tolist())
                )
            
            with col2:
                show_responses = st.checkbox("Show response counts", value=True)
            
            # Filter surveys by category
            if selected_category == "All Categories":
                filtered_surveys = df_surveys.copy()
            else:
                filtered_surveys = df_surveys[df_surveys['category'] == selected_category].copy()
            
            st.markdown(f"### 📋 {selected_category} Surveys ({len(filtered_surveys)} items)")
            
            if not filtered_surveys.empty:
                # Display surveys table
                display_columns = ['survey_title', 'category']
                if show_responses:
                    display_columns.append('num_responses')
                
                display_df = filtered_surveys[display_columns + ['survey_id']].copy()
                
                # Add category badges
                display_df['category_badge'] = display_df['category'].apply(lambda x: f'<span class="category-badge category-{x.lower().replace(" ", "-")}">{x}</span>')
                
                st.dataframe(
                    display_df.drop(['survey_id'], axis=1),
                    column_config={
                        "survey_title": st.column_config.TextColumn("Survey Title", width="large"),
                        "category": st.column_config.TextColumn("Category", width="medium"),
                        "category_badge": st.column_config.TextColumn("Category", width="medium"),
                        "num_responses": st.column_config.NumberColumn("Responses", width="small") if show_responses else None
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Unique questions by category
                st.markdown(f"### 🆔 Unique Questions in {selected_category}")
                
                if not unique_bank.empty:
                    if selected_category == "All Categories":
                        category_questions = unique_bank.copy()
                    else:
                        category_questions = unique_bank[unique_bank['survey_category'] == selected_category].copy()
                    
                    if not category_questions.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("🆔 Unique UIDs", len(category_questions))
                        with col2:
                            st.metric("📝 Total Variants", category_questions['total_variants'].sum())
                        with col3:
                            avg_quality = category_questions['quality_score'].mean()
                            st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
                        with col4:
                            governance_rate = (category_questions['governance_compliant'] == True).sum() / len(category_questions) * 100
                            st.metric("⚖️ Governance", f"{governance_rate:.1f}%")
                        
                        # UID assignment interface
                        st.markdown("#### 🔧 UID Assignment Interface")
                        
                        # Search and filter
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            search_term = st.text_input("🔍 Search questions", placeholder="Type to filter...")
                        
                        with col2:
                            sort_by = st.selectbox("Sort by", ["UID", "Quality Score", "Total Variants"])
                        
                        # Apply search filter
                        if search_term:
                            category_questions = category_questions[
                                category_questions['best_question'].str.contains(search_term, case=False, na=False)
                            ]
                        
                        # Apply sorting
                        if sort_by == "UID":
                            try:
                                category_questions['uid_numeric'] = pd.to_numeric(category_questions['uid'], errors='coerce')
                                category_questions = category_questions.sort_values(['uid_numeric', 'uid'], na_position='last')
                                category_questions = category_questions.drop('uid_numeric', axis=1)
                            except:
                                category_questions = category_questions.sort_values('uid')
                        elif sort_by == "Quality Score":
                            category_questions = category_questions.sort_values('quality_score', ascending=False)
                        elif sort_by == "Total Variants":
                            category_questions = category_questions.sort_values('total_variants', ascending=False)
                        
                        # Display questions with UID assignment capability
                        st.markdown("#### 📋 Questions with UID Assignment")
                        
                        assignment_changes = {}
                        
                        for idx, row in category_questions.head(20).iterrows():  # Show top 20 for performance
                            with st.expander(f"🆔 UID {row['uid']} - Quality: {row['quality_score']:.1f} - Variants: {row['total_variants']}"):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Question:** {row['best_question']}")
                                    if row['standardized_question'] != row['best_question']:
                                        st.markdown(f"**Standardized:** {row['standardized_question']}")
                                    
                                    st.markdown(f"**Category:** {row['survey_category']}")
                                    st.markdown(f"**Governance:** {'✅ Compliant' if row['governance_compliant'] else '❌ Violation'}")
                                
                                with col2:
                                    current_uid = row['uid']
                                    new_uid = st.text_input(
                                        f"New UID",
                                        value=str(current_uid),
                                        key=f"category_uid_{idx}"
                                    )
                                    
                                    if new_uid != str(current_uid):
                                        assignment_changes[idx] = new_uid
                                    
                                    # Show assignment options
                                    if st.button(f"📋 View All Variants", key=f"variants_{idx}"):
                                        st.write("**All Variants:**")
                                        for i, variant in enumerate(row['all_variants'][:5], 1):
                                            st.write(f"{i}. {variant}")
                                        if len(row['all_variants']) > 5:
                                            st.write(f"... and {len(row['all_variants']) - 5} more variants")
                        
                        # Save assignment changes
                        if assignment_changes:
                            if st.button("💾 Save UID Changes", type="primary"):
                                # Update the unique bank with new UIDs
                                for idx, new_uid in assignment_changes.items():
                                    if idx in category_questions.index:
                                        st.session_state.unique_questions_bank.loc[
                                            st.session_state.unique_questions_bank['uid'] == category_questions.loc[idx, 'uid'], 
                                            'uid'
                                        ] = new_uid
                                
                                st.success(f"✅ Updated {len(assignment_changes)} UID assignments!")
                                st.rerun()
                        
                        # Download options
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                f"📥 Download {selected_category} Questions",
                                category_questions.to_csv(index=False),
                                f"{selected_category.lower().replace(' ', '_')}_questions_{uuid4().hex[:8]}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Create UID mapping file
                            uid_mapping = category_questions[['uid', 'best_question', 'survey_category']].copy()
                            uid_mapping = uid_mapping.rename(columns={
                                'uid': 'UID',
                                'best_question': 'Question',
                                'survey_category': 'Category'
                            })
                            
                            st.download_button(
                                "🆔 Download UID Mapping",
                                uid_mapping.to_csv(index=False),
                                f"uid_mapping_{selected_category.lower().replace(' ', '_')}_{uuid4().hex[:8]}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Survey assignment template
                            survey_template = filtered_surveys[['survey_id', 'survey_title', 'category']].copy()
                            survey_template['recommended_uids'] = survey_template['category'].apply(
                                lambda cat: ', '.join(
                                    unique_bank[unique_bank['survey_category'] == cat]['uid'].head(5).tolist()
                                ) if not unique_bank[unique_bank['survey_category'] == cat].empty else ''
                            )
                            
                            st.download_button(
                                "📊 Download Survey Template",
                                survey_template.to_csv(index=False),
                                f"survey_template_{selected_category.lower().replace(' ', '_')}_{uuid4().hex[:8]}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                    
                    else:
                        st.info(f"ℹ️ No unique questions found for category: {selected_category}")
                
                else:
                    st.warning("⚠️ No unique questions bank available. Please load the question bank first.")
            
            else:
                st.info("ℹ️ No surveys found in the selected category.")
    
    except Exception as e:
        logger.error(f"Survey categorization failed: {e}")
        st.error(f"❌ Error: {e}")

# Enhanced Unique Questions Bank Page
elif st.session_state.page == "unique_question_bank":
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*Enhanced with standardization, governance, and category-based organization from Snowflake HEADING_0 data*")
    
    # Data source clarification
    st.info("**📊 Data Source:** Questions and UIDs from Snowflake HEADING_0 data. Categories determined by question content analysis since survey titles are not available in Snowflake.")
    
    try:
        with st.spinner("🔄 Loading enhanced unique questions bank..."):
            if st.session_state.unique_questions_bank is None:
                df_reference = get_all_reference_questions()
                st.session_state.unique_questions_bank = create_enhanced_unique_questions_bank(df_reference)
            
            unique_questions_df = st.session_state.unique_questions_bank
        
        if unique_questions_df.empty:
            st.warning("⚠️ No unique questions found in the database.")
        else:
            # Enhanced summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🆔 Unique UIDs", len(unique_questions_df))
            with col2:
                st.metric("📝 Total Variants", unique_questions_df['total_variants'].sum())
            with col3:
                governance_compliant = len(unique_questions_df[unique_questions_df['governance_compliant'] == True])
                st.metric("⚖️ Governance", f"{governance_compliant}/{len(unique_questions_df)}")
            with col4:
                avg_quality = unique_questions_df['quality_score'].mean()
                st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
            with col5:
                categories = unique_questions_df['survey_category'].nunique()
                st.metric("📊 Categories", categories)
            
            st.markdown("---")
            
            # Enhanced search and filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            
            with col2:
                category_filter = st.selectbox("📊 Filter by Category", ["All"] + sorted(unique_questions_df['survey_category'].unique().tolist()))
            
            with col3:
                quality_filter = st.selectbox("🎯 Quality Filter", ["All", "High (>15)", "Medium (5-15)", "Low (<5)"])
            
            # Additional filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                governance_filter = st.selectbox("⚖️ Governance", ["All", "Compliant Only", "Violations Only"])
            
            with col2:
                variants_filter = st.selectbox("📊 Variants", ["All", "High (>10)", "Medium (5-10)", "Low (<5)"])
            
            with col3:
                show_standardized = st.checkbox("Show standardized questions", value=True)
            
            # Apply filters
            filtered_df = unique_questions_df.copy()
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['best_question'].str.contains(search_term, case=False, na=False) |
                    filtered_df['standardized_question'].str.contains(search_term, case=False, na=False)
                ]
            
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['survey_category'] == category_filter]
            
            if quality_filter == "High (>15)":
                filtered_df = filtered_df[filtered_df['quality_score'] > 15]
            elif quality_filter == "Medium (5-15)":
                filtered_df = filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] <= 15)]
            elif quality_filter == "Low (<5)":
                filtered_df = filtered_df[filtered_df['quality_score'] < 5]
            
            if governance_filter == "Compliant Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == True]
            elif governance_filter == "Violations Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == False]
            
            if variants_filter == "High (>10)":
                filtered_df = filtered_df[filtered_df['total_variants'] > 10]
            elif variants_filter == "Medium (5-10)":
                filtered_df = filtered_df[(filtered_df['total_variants'] >= 5) & (filtered_df['total_variants'] <= 10)]
            elif variants_filter == "Low (<5)":
                filtered_df = filtered_df[filtered_df['total_variants'] < 5]
            
            st.markdown(f"### 📋 Showing {len(filtered_df)} unique questions")
            
            # Display the enhanced unique questions
            if not filtered_df.empty:
                display_df = filtered_df.copy()
                
                # Prepare display columns
                display_columns = {
                    'uid': 'UID',
                    'best_question': 'Best Question',
                    'survey_category': 'Category',
                    'total_variants': 'Variants',
                    'quality_score': 'Quality',
                    'governance_compliant': 'Governance',
                    'question_length': 'Length',
                    'question_words': 'Words'
                }
                
                if show_standardized:
                    display_columns['standardized_question'] = 'Standardized Question'
                
                display_df = display_df.rename(columns=display_columns)
                
                # Add governance and category icons
                display_df['Governance'] = display_df['Governance'].apply(lambda x: "✅" if x else "❌")
                
                # Add category badges
                display_df['Category_Badge'] = display_df['Category'].apply(
                    lambda x: f'<span class="category-badge category-{x.lower().replace(" ", "-")}">{x}</span>'
                )
                
                st.dataframe(
                    display_df.drop(['Category'], axis=1),
                    column_config={
                        "UID": st.column_config.TextColumn("UID", width="small"),
                        "Best Question": st.column_config.TextColumn("Best Question", width="large"),
                        "Category_Badge": st.column_config.TextColumn("Category", width="medium"),
                        "Variants": st.column_config.NumberColumn("Variants", width="small"),
                        "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                        "Governance": st.column_config.TextColumn("Gov", width="small"),
                        "Length": st.column_config.NumberColumn("Length", width="small"),
                        "Words": st.column_config.NumberColumn("Words", width="small"),
                        "Standardized Question": st.column_config.TextColumn("Standardized Question", width="large") if show_standardized else None
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Enhanced download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Filtered Results",
                        display_df.to_csv(index=False),
                        f"enhanced_unique_questions_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Create UID mapping for external use
                    uid_mapping = filtered_df[['uid', 'standardized_question', 'survey_category', 'quality_score']].copy()
                    uid_mapping = uid_mapping.rename(columns={
                        'uid': 'UID',
                        'standardized_question': 'Standard_Question',
                        'survey_category': 'Category',
                        'quality_score': 'Quality_Score'
                    })
                    
                    st.download_button(
                        "🆔 Download UID Mapping",
                        uid_mapping.to_csv(index=False),
                        f"uid_mapping_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    # Generate category-specific exports
                    if category_filter != "All":
                        st.download_button(
                            f"📊 Download {category_filter} Only",
                            filtered_df.to_csv(index=False),
                            f"{category_filter.lower().replace(' ', '_')}_questions_{uuid4().hex[:8]}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                # Interactive UID management
                st.markdown("---")
                st.markdown("### 🔧 Interactive UID Management")
                
                if st.checkbox("Enable UID editing and management", value=False):
                    st.markdown("#### ✏️ Bulk UID Operations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bulk category reassignment
                        st.markdown("**Bulk Category Reassignment**")
                        new_category = st.selectbox(
                            "Assign new category to filtered questions:",
                            list(SURVEY_CATEGORIES.keys()) + ['Other']
                        )
                        
                        if st.button("🔄 Apply Category Change"):
                            for idx in filtered_df.index:
                                st.session_state.unique_questions_bank.loc[idx, 'survey_category'] = new_category
                            st.success(f"✅ Updated {len(filtered_df)} questions to category: {new_category}")
                            st.rerun()
                    
                    with col2:
                        # UID consolidation
                        st.markdown("**UID Consolidation**")
                        target_uid = st.text_input("Target UID for consolidation:")
                        
                        if target_uid and st.button("🔗 Consolidate Selected UIDs"):
                            consolidated_count = 0
                            for idx in filtered_df.index:
                                if st.session_state.unique_questions_bank.loc[idx, 'uid'] != target_uid:
                                    st.session_state.unique_questions_bank.loc[idx, 'uid'] = target_uid
                                    consolidated_count += 1
                            
                            if consolidated_count > 0:
                                st.success(f"✅ Consolidated {consolidated_count} questions to UID: {target_uid}")
                                st.rerun()
                    
                    # Individual UID editing
                    st.markdown("#### 📝 Individual UID Editing")
                    
                    edit_uid_changes = {}
                    
                    for idx, row in filtered_df.head(10).iterrows():  # Show top 10 for performance
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.text(f"{row['best_question'][:80]}...")
                        
                        with col2:
                            current_uid = row['uid']
                            new_uid = st.text_input(
                                f"UID",
                                value=str(current_uid),
                                key=f"edit_uid_{idx}"
                            )
                            
                            if new_uid != str(current_uid):
                                edit_uid_changes[idx] = new_uid
                        
                        with col3:
                            current_category = row['survey_category']
                            new_category = st.selectbox(
                                "Category",
                                list(SURVEY_CATEGORIES.keys()) + ['Other'],
                                index=list(SURVEY_CATEGORIES.keys()).index(current_category) if current_category in SURVEY_CATEGORIES.keys() else 0,
                                key=f"edit_cat_{idx}"
                            )
                            
                            if new_category != current_category:
                                edit_uid_changes[f"cat_{idx}"] = new_category
                        
                        with col4:
                            quality_score = row['quality_score']
                            governance = "✅" if row['governance_compliant'] else "❌"
                            st.text(f"Q:{quality_score:.1f} {governance}")
                    
                    # Save individual changes
                    if edit_uid_changes:
                        if st.button("💾 Save Individual Changes", type="primary"):
                            for key, value in edit_uid_changes.items():
                                if key.startswith("cat_"):
                                    idx = int(key.split("_")[1])
                                    st.session_state.unique_questions_bank.loc[idx, 'survey_category'] = value
                                else:
                                    st.session_state.unique_questions_bank.loc[key, 'uid'] = value
                            
                            st.success(f"✅ Applied {len(edit_uid_changes)} changes!")
                            st.rerun()
            
            else:
                st.info("ℹ️ No questions match your current filters.")
                
    except Exception as e:
        logger.error(f"Enhanced unique questions bank failed: {e}")
        st.error(f"❌ Error: {e}")

# Enhanced Categorized Questions Page
elif st.session_state.page == "categorized_questions":
    st.markdown("## 📊 Enhanced Categorized Questions")
    st.markdown("*Advanced categorization with standardization and governance insights*")
    
    try:
        with st.spinner("🔄 Loading categorized questions..."):
            if st.session_state.unique_questions_bank is None:
                df_reference = get_all_reference_questions()
                st.session_state.unique_questions_bank = create_enhanced_unique_questions_bank(df_reference)
            
            unique_questions_df = st.session_state.unique_questions_bank
        
        if unique_questions_df.empty:
            st.warning("⚠️ No categorized questions found.")
        else:
            # Enhanced category overview with governance and quality metrics
            category_stats = unique_questions_df.groupby('survey_category').agg({
                'uid': 'count',
                'total_variants': 'sum',
                'quality_score': ['mean', 'std'],
                'governance_compliant': lambda x: (x == True).sum(),
                'question_length': 'mean'
            }).round(2)
            
            category_stats.columns = ['Questions', 'Total_Variants', 'Avg_Quality', 'Quality_Std', 'Governance_Compliant', 'Avg_Length']
            category_stats['Governance_Rate'] = (category_stats['Governance_Compliant'] / category_stats['Questions'] * 100).round(1)
            category_stats = category_stats.sort_values('Questions', ascending=False)
            
            st.markdown("### 📊 Enhanced Category Overview")
            
            # Visual category display with enhanced metrics
            categories = list(SURVEY_CATEGORIES.keys()) + ['Other', 'Unknown', 'Mixed']
            
            for category in categories:
                if category in category_stats.index:
                    stats = category_stats.loc[category]
                    badge_class = f"category-{category.lower().replace(' ', '-')}"
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><span class="category-badge {badge_class}">{category}</span><br><strong>{int(stats["Questions"])} questions</strong></div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("📝 Total Variants", int(stats['Total_Variants']))
                    
                    with col3:
                        st.metric("🎯 Avg Quality", f"{stats['Avg_Quality']:.1f} ±{stats['Quality_Std']:.1f}")
                    
                    with col4:
                        st.metric("⚖️ Governance", f"{stats['Governance_Rate']:.1f}%")
                    
                    st.markdown("---")
            
            # Detailed category analysis
            st.markdown("### 📈 Detailed Category Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_category = st.selectbox(
                    "📊 Select Category",
                    ["All"] + sorted(unique_questions_df['survey_category'].unique().tolist())
                )
            
            with col2:
                sort_by = st.selectbox(
                    "🔄 Sort by",
                    ["UID", "Quality Score", "Total Variants", "Question Length", "Governance Status"]
                )
            
            with col3:
                analysis_type = st.selectbox(
                    "📊 Analysis Type",
                    ["Question List", "Statistical Summary", "Quality Distribution", "Governance Report"]
                )
            
            # Filter by selected category
            if selected_category == "All":
                filtered_df = unique_questions_df.copy()
            else:
                filtered_df = unique_questions_df[unique_questions_df['survey_category'] == selected_category].copy()
            
            # Apply sorting
            if sort_by == "UID":
                try:
                    filtered_df['uid_numeric'] = pd.to_numeric(filtered_df['uid'], errors='coerce')
                    filtered_df = filtered_df.sort_values(['uid_numeric', 'uid'], na_position='last')
                    filtered_df = filtered_df.drop('uid_numeric', axis=1)
                except:
                    filtered_df = filtered_df.sort_values('uid')
            elif sort_by == "Quality Score":
                filtered_df = filtered_df.sort_values('quality_score', ascending=False)
            elif sort_by == "Total Variants":
                filtered_df = filtered_df.sort_values('total_variants', ascending=False)
            elif sort_by == "Question Length":
                filtered_df = filtered_df.sort_values('question_length', ascending=False)
            elif sort_by == "Governance Status":
                filtered_df = filtered_df.sort_values('governance_compliant', ascending=False)
            
            st.markdown(f"### 📋 {selected_category} Analysis ({len(filtered_df)} items)")
            
            if not filtered_df.empty:
                # Display based on analysis type
                if analysis_type == "Question List":
                    # Standard question list with enhanced columns
                    display_df = filtered_df[['uid', 'best_question', 'standardized_question', 'survey_category', 'total_variants', 'quality_score', 'governance_compliant']].copy()
                    
                    display_df['governance_compliant'] = display_df['governance_compliant'].apply(lambda x: "✅" if x else "❌")
                    display_df['category_badge'] = display_df['survey_category'].apply(lambda x: f'<span class="category-badge category-{x.lower().replace(" ", "-")}">{x}</span>')
                    
                    display_df = display_df.rename(columns={
                        'uid': 'UID',
                        'best_question': 'Original Question',
                        'standardized_question': 'Standardized Question',
                        'total_variants': 'Variants',
                        'quality_score': 'Quality',
                        'governance_compliant': 'Governance'
                    })
                    
                    st.dataframe(
                        display_df.drop(['survey_category'], axis=1),
                        column_config={
                            "UID": st.column_config.TextColumn("UID", width="small"),
                            "Original Question": st.column_config.TextColumn("Original Question", width="large"),
                            "Standardized Question": st.column_config.TextColumn("Standardized Question", width="large"),
                            "category_badge": st.column_config.TextColumn("Category", width="medium"),
                            "Variants": st.column_config.NumberColumn("Variants", width="small"),
                            "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                            "Governance": st.column_config.TextColumn("Gov", width="small")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                elif analysis_type == "Statistical Summary":
                    # Statistical summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Basic Statistics")
                        st.metric("Total Questions", len(filtered_df))
                        st.metric("Total Variants", filtered_df['total_variants'].sum())
                        st.metric("Avg Variants per UID", f"{filtered_df['total_variants'].mean():.1f}")
                        st.metric("Max Variants", filtered_df['total_variants'].max())
                        
                    with col2:
                        st.markdown("#### 🎯 Quality Metrics")
                        st.metric("Avg Quality Score", f"{filtered_df['quality_score'].mean():.1f}")
                        st.metric("Quality Std Dev", f"{filtered_df['quality_score'].std():.1f}")
                        st.metric("Governance Rate", f"{(filtered_df['governance_compliant'] == True).sum() / len(filtered_df) * 100:.1f}%")
                        st.metric("Avg Question Length", f"{filtered_df['question_length'].mean():.0f} chars")
                    
                    # Distribution charts would go here (using st.bar_chart, etc.)
                    st.markdown("#### 📈 Quality Score Distribution")
                    quality_bins = pd.cut(filtered_df['quality_score'], bins=5)
                    quality_dist = quality_bins.value_counts().sort_index()
                    st.bar_chart(quality_dist)
                
                elif analysis_type == "Quality Distribution":
                    # Quality distribution analysis
                    st.markdown("#### 🎯 Quality Score Analysis")
                    
                    quality_ranges = {
                        'Excellent (>20)': len(filtered_df[filtered_df['quality_score'] > 20]),
                        'Good (15-20)': len(filtered_df[(filtered_df['quality_score'] >= 15) & (filtered_df['quality_score'] <= 20)]),
                        'Fair (10-15)': len(filtered_df[(filtered_df['quality_score'] >= 10) & (filtered_df['quality_score'] < 15)]),
                        'Poor (5-10)': len(filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] < 10)]),
                        'Very Poor (<5)': len(filtered_df[filtered_df['quality_score'] < 5])
                    }
                    
                    quality_df = pd.DataFrame(list(quality_ranges.items()), columns=['Quality Range', 'Count'])
                    quality_df['Percentage'] = (quality_df['Count'] / len(filtered_df) * 100).round(1)
                    
                    st.dataframe(quality_df, use_container_width=True)
                    
                    # Show examples from each quality range
                    st.markdown("#### 📝 Quality Examples")
                    
                    for range_name, count in quality_ranges.items():
                        if count > 0:
                            with st.expander(f"{range_name} - {count} questions"):
                                if range_name == 'Excellent (>20)':
                                    examples = filtered_df[filtered_df['quality_score'] > 20].head(3)
                                elif range_name == 'Good (15-20)':
                                    examples = filtered_df[(filtered_df['quality_score'] >= 15) & (filtered_df['quality_score'] <= 20)].head(3)
                                elif range_name == 'Fair (10-15)':
                                    examples = filtered_df[(filtered_df['quality_score'] >= 10) & (filtered_df['quality_score'] < 15)].head(3)
                                elif range_name == 'Poor (5-10)':
                                    examples = filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] < 10)].head(3)
                                else:
                                    examples = filtered_df[filtered_df['quality_score'] < 5].head(3)
                                
                                for _, example in examples.iterrows():
                                    st.write(f"**UID {example['uid']} (Score: {example['quality_score']:.1f}):** {example['best_question']}")
                
                elif analysis_type == "Governance Report":
                    # Governance compliance report
                    st.markdown("#### ⚖️ Governance Compliance Report")
                    
                    compliant = filtered_df[filtered_df['governance_compliant'] == True]
                    violations = filtered_df[filtered_df['governance_compliant'] == False]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### ✅ Compliant UIDs")
                        st.metric("Compliant Count", len(compliant))
                        st.metric("Compliance Rate", f"{len(compliant) / len(filtered_df) * 100:.1f}%")
                        
                        if not compliant.empty:
                            st.markdown("**Best Performing UIDs:**")
                            top_compliant = compliant.nlargest(5, 'quality_score')[['uid', 'quality_score', 'total_variants']]
                            st.dataframe(top_compliant, hide_index=True)
                    
                    with col2:
                        st.markdown("##### ❌ Governance Violations")
                        st.metric("Violation Count", len(violations))
                        st.metric("Violation Rate", f"{len(violations) / len(filtered_df) * 100:.1f}%")
                        
                        if not violations.empty:
                            st.markdown("**Most Problematic UIDs:**")
                            worst_violations = violations.nlargest(5, 'total_variants')[['uid', 'total_variants', 'quality_score']]
                            st.dataframe(worst_violations, hide_index=True)
                    
                    # Detailed violation analysis
                    if not violations.empty:
                        st.markdown("##### 🔍 Detailed Violation Analysis")
                        
                        for _, violation in violations.head(5).iterrows():
                            with st.expander(f"UID {violation['uid']} - {violation['total_variants']} variants (Max: {UID_GOVERNANCE['max_variations_per_uid']})"):
                                st.write(f"**Best Question:** {violation['best_question']}")
                                st.write(f"**Quality Score:** {violation['quality_score']:.1f}")
                                st.write(f"**Category:** {violation['survey_category']}")
                                st.write(f"**Total Variants:** {violation['total_variants']}")
                                
                                # Show some variants
                                variants = violation['all_variants'][:5]
                                st.write("**Sample Variants:**")
                                for i, variant in enumerate(variants, 1):
                                    st.write(f"{i}. {variant}")
                
                # Enhanced download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        f"📥 Download {selected_category} Data",
                        filtered_df.to_csv(index=False),
                        f"{selected_category.lower().replace(' ', '_')}_analysis_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Category-specific statistics
                    if selected_category != "All":
                        category_report = category_stats.loc[[selected_category]] if selected_category in category_stats.index else pd.DataFrame()
                        
                        if not category_report.empty:
                            st.download_button(
                                f"📊 Download {selected_category} Stats",
                                category_report.to_csv(),
                                f"{selected_category.lower().replace(' ', '_')}_stats_{uuid4().hex[:8]}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                
                with col3:
                    # Cross-category comparison
                    comparison_df = category_stats.copy()
                    comparison_df['Category'] = comparison_df.index
                    
                    st.download_button(
                        "📈 Download Cross-Category Report",
                        comparison_df.to_csv(index=False),
                        f"cross_category_comparison_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            else:
                st.info("ℹ️ No questions found in the selected category.")
                
    except Exception as e:
        logger.error(f"Enhanced categorized questions failed: {e}")
        st.error(f"❌ Error: {e}")

# View Surveys Page (keeping existing functionality)
elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ View SurveyMonkey Surveys")
    
    try:
        token = st.secrets["surveymonkey"]["token"]
        surveys = get_surveys(token)
        
        if not surveys:
            st.warning("⚠️ No surveys found in your SurveyMonkey account.")
        else:
            st.success(f"✅ Found {len(surveys)} surveys in your account")
            
            # Enhanced survey display with categories
            surveys_data = []
            for survey in surveys:
                title = survey.get("title", "Untitled")
                survey_id = survey.get("id", "")
                category = categorize_survey(title)
                
                surveys_data.append({
                    'Survey ID': survey_id,
                    'Title': title,
                    'Category': category,
                    'Date Created': survey.get('date_created', ''),
                    'Date Modified': survey.get('date_modified', ''),
                    'Response Count': survey.get('response_count', 0)
                })
            
            df_surveys = pd.DataFrame(surveys_data)
            
            # Add category badges
            df_surveys['Category_Badge'] = df_surveys['Category'].apply(
                lambda x: f'<span class="category-badge category-{x.lower().replace(" ", "-")}">{x}</span>'
            )
            
            st.dataframe(
                df_surveys.drop(['Category'], axis=1),
                column_config={
                    "Survey ID": st.column_config.TextColumn("Survey ID", width="medium"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Category_Badge": st.column_config.TextColumn("Category", width="medium"),
                    "Date Created": st.column_config.TextColumn("Created", width="medium"),
                    "Date Modified": st.column_config.TextColumn("Modified", width="medium"),
                    "Response Count": st.column_config.NumberColumn("Responses", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
            
    except Exception as e:
        logger.error(f"View surveys failed: {e}")
        st.error(f"❌ Error loading surveys: {e}")

else:
    st.error("🚨 Unknown page. Please use the sidebar navigation.")

# Footer
st.markdown("---")
st.markdown("### 🧠 UID Matcher Pro - Enhanced Edition")
st.markdown("*Powered by Semantic Matching, Governance Rules, and Survey Categorization*")
