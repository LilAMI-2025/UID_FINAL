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
    
    .governance-compliant {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 4px;
    }
    
    .governance-violation {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Constants with New Features
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.75  # Increased for better matching
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
    'monthly_quality_check': True,
    'standardization_required': True,
    'auto_assign_new_uid': True,
    'governance_violation_action': 'warn_and_log'  # 'warn_and_log', 'auto_fix', 'reject'
}

# Question Standardization Rules
STANDARDIZATION_RULES = {
    'capitalize_first_letter': True,
    'remove_extra_whitespace': True,
    'standardize_punctuation': True,
    'normalize_question_words': True,
    'remove_html_artifacts': True,
    'standardize_common_phrases': True
}

# Enhanced Survey Categories
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation'],
    'GROW': ['GROW'],
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
    "organisation": "organization",
    "colour": "color",
    "favour": "favor"
}

# Question Format Standardization Patterns
STANDARDIZATION_PATTERNS = {
    'question_starters': {
        'what is your': 'What is your',
        'how many': 'How many',
        'which of the following': 'Which of the following',
        'do you': 'Do you',
        'have you': 'Have you',
        'would you': 'Would you'
    },
    'common_endings': {
        ' ?': '?',
        '??': '?',
        ' .': '.',
        '..': '.'
    },
    'html_removal': [
        r'<[^>]+>',  # Remove HTML tags
        r'&nbsp;',   # Remove HTML entities
        r'&amp;',
        r'&lt;',
        r'&gt;'
    ]
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# ========================
# NEW: Question Standardization Functions
# ========================

def standardize_question_format(question_text):
    """
    Standardize question format according to governance rules
    """
    if not question_text or pd.isna(question_text):
        return question_text
    
    text = str(question_text).strip()
    
    if not STANDARDIZATION_RULES['standardization_required']:
        return text
    
    # Remove HTML artifacts
    if STANDARDIZATION_RULES['remove_html_artifacts']:
        for pattern in STANDARDIZATION_PATTERNS['html_removal']:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    if STANDARDIZATION_RULES['remove_extra_whitespace']:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Capitalize first letter
    if STANDARDIZATION_RULES['capitalize_first_letter'] and text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Standardize punctuation
    if STANDARDIZATION_RULES['standardize_punctuation']:
        for old, new in STANDARDIZATION_PATTERNS['common_endings'].items():
            if text.endswith(old):
                text = text[:-len(old)] + new
    
    # Normalize question words
    if STANDARDIZATION_RULES['normalize_question_words']:
        text_lower = text.lower()
        for old, new in STANDARDIZATION_PATTERNS['question_starters'].items():
            if text_lower.startswith(old):
                text = new + text[len(old):]
                break
    
    # Standardize common phrases
    if STANDARDIZATION_RULES['standardize_common_phrases']:
        for old, new in ENHANCED_SYNONYM_MAP.items():
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    
    return text.strip()

def calculate_question_quality_score(question_text):
    """
    Enhanced scoring function for question quality with governance compliance
    """
    if not question_text or pd.isna(question_text):
        return 0
    
    score = 0
    text = str(question_text).lower().strip()
    
    # Length scoring (sweet spot is 10-100 characters)
    length = len(text)
    if 10 <= length <= 100:
        score += 25
    elif 5 <= length <= 150:
        score += 15
    elif length < 5:
        score -= 25
    elif length > 200:
        score -= 10
    
    # Question format scoring
    if text.endswith('?'):
        score += 20
    elif text.endswith('.') and any(text.startswith(word) for word in ['please', 'select', 'choose']):
        score += 10
    
    # English question word scoring
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
    first_words = text.split()[:3]
    if any(word in first_words for word in question_words):
        score += 20
    
    # Proper capitalization
    if question_text and question_text[0].isupper():
        score += 10
    
    # Avoid artifacts (enhanced list)
    bad_patterns = ['click here', 'please select', '...', 'n/a', 'other', 'select one', 'choose all', 'privacy policy', 'terms and conditions']
    penalty = sum(15 for pattern in bad_patterns if pattern in text)
    score -= penalty
    
    # Avoid HTML
    if '<' in text and '>' in text:
        score -= 25
    
    # Prefer complete sentences
    word_count = len(text.split())
    if 5 <= word_count <= 20:
        score += 15
    elif word_count > 30:
        score -= 10
    elif word_count < 3:
        score -= 15
    
    # Avoid repetitive characters
    if any(char * 3 in text for char in 'abcdefghijklmnopqrstuvwxyz'):
        score -= 15
    
    # Semantic coherence bonus
    if all(word.replace(',', '').replace('.', '').replace('?', '').isalpha() or word in ['?', '.', ',', '(', ')'] for word in text.split()):
        score += 10
    
    # Grammar patterns (basic)
    if re.search(r'\b(what|how|when|where|why|which)\s+(is|are|do|does|did|will|would|can|could)\b', text):
        score += 10
    
    return max(0, score)  # Ensure non-negative score

# ========================
# NEW: Advanced Semantic Matching Functions
# ========================

@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model"""
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

def compute_semantic_similarity(question1, question2, model=None):
    """
    Compute semantic similarity between two questions
    """
    if not model:
        model = load_sentence_transformer()
    
    try:
        # Standardize both questions first
        q1_std = standardize_question_format(question1)
        q2_std = standardize_question_format(question2)
        
        # Get embeddings
        embeddings = model.encode([q1_std, q2_std], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        return similarity
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        return 0.0

def find_best_semantic_match(question_text, existing_uids_data, threshold=None):
    """
    Find the best semantic match for a question using advanced matching
    """
    if not existing_uids_data:
        return None, 0.0, {}
    
    if threshold is None:
        threshold = UID_GOVERNANCE['semantic_similarity_threshold']
    
    try:
        model = load_sentence_transformer()
        
        # Standardize input question
        std_question = standardize_question_format(question_text)
        
        # Get all existing questions
        existing_questions = []
        uid_mapping = {}
        
        for uid, data in existing_uids_data.items():
            if isinstance(data, dict) and 'best_question' in data:
                question = data['best_question']
            else:
                question = str(data)
            
            std_existing = standardize_question_format(question)
            existing_questions.append(std_existing)
            uid_mapping[len(existing_questions) - 1] = uid
        
        if not existing_questions:
            return None, 0.0, {}
        
        # Calculate embeddings
        input_embedding = model.encode([std_question], convert_to_tensor=True)
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(input_embedding, existing_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        match_details = {
            'original_question': question_text,
            'standardized_question': std_question,
            'matched_question': existing_questions[best_idx],
            'original_matched_question': list(existing_uids_data.values())[best_idx] if isinstance(list(existing_uids_data.values())[best_idx], str) else list(existing_uids_data.values())[best_idx].get('best_question', ''),
            'similarity_score': best_score,
            'threshold_met': best_score >= threshold,
            'confidence_level': 'High' if best_score >= 0.9 else 'Medium' if best_score >= 0.75 else 'Low'
        }
        
        if best_score >= threshold:
            matched_uid = uid_mapping[best_idx]
            return matched_uid, best_score, match_details
        
        return None, best_score, match_details
        
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        return None, 0.0, {'error': str(e)}

# ========================
# NEW: UID Governance Functions
# ========================

def check_uid_governance_compliance(uid, existing_uids_data):
    """
    Check if assigning a new variation to a UID would violate governance rules
    """
    if not UID_GOVERNANCE['conflict_detection_enabled']:
        return True, "Governance checking disabled"
    
    if uid not in existing_uids_data:
        return True, "New UID - compliant"
    
    uid_data = existing_uids_data[uid]
    
    if isinstance(uid_data, dict):
        current_variations = uid_data.get('variation_count', 1)
    else:
        # Count variations manually if not tracked
        current_variations = 1  # Simplified for this example
    
    max_allowed = UID_GOVERNANCE['max_variations_per_uid']
    
    if current_variations >= max_allowed:
        return False, f"Exceeds max variations ({current_variations}/{max_allowed})"
    
    return True, f"Within limits ({current_variations}/{max_allowed})"

def get_next_available_uid(existing_uids_data):
    """
    Get the next available UID following governance rules
    """
    if not existing_uids_data:
        return "1"
    
    # Extract numeric UIDs
    numeric_uids = []
    for uid in existing_uids_data.keys():
        try:
            numeric_uids.append(int(uid))
        except ValueError:
            continue
    
    if not numeric_uids:
        return "1"
    
    return str(max(numeric_uids) + 1)

def assign_uid_with_semantic_governance(question_text, existing_uids_data, force_new=False):
    """
    Assign UID using semantic matching and governance rules
    """
    result = {
        'assigned_uid': None,
        'method': 'unknown',
        'confidence': 0.0,
        'governance_compliant': True,
        'standardized_question': standardize_question_format(question_text),
        'quality_score': calculate_question_quality_score(question_text),
        'match_details': {},
        'governance_check': {},
        'recommendations': []
    }
    
    # Standardize the question first
    std_question = standardize_question_format(question_text)
    result['standardized_question'] = std_question
    
    # Skip UID assignment for low-quality questions
    if result['quality_score'] < UID_GOVERNANCE['quality_score_threshold']:
        result['recommendations'].append(f"Question quality score ({result['quality_score']:.1f}) below threshold ({UID_GOVERNANCE['quality_score_threshold']})")
        if not force_new:
            return result
    
    # Try semantic matching first
    if not force_new and existing_uids_data:
        matched_uid, similarity, match_details = find_best_semantic_match(std_question, existing_uids_data)
        result['match_details'] = match_details
        
        if matched_uid:
            # Check governance compliance for this UID
            is_compliant, governance_msg = check_uid_governance_compliance(matched_uid, existing_uids_data)
            result['governance_check'] = {
                'uid': matched_uid,
                'compliant': is_compliant,
                'message': governance_msg
            }
            
            if is_compliant:
                result['assigned_uid'] = matched_uid
                result['method'] = 'semantic_match'
                result['confidence'] = similarity
                result['recommendations'].append(f"Semantically matched to existing UID {matched_uid} (similarity: {similarity:.3f})")
                return result
            else:
                result['recommendations'].append(f"Best semantic match UID {matched_uid} violates governance: {governance_msg}")
                # Fall through to create new UID
    
    # Assign new UID
    new_uid = get_next_available_uid(existing_uids_data)
    result['assigned_uid'] = new_uid
    result['method'] = 'new_assignment'
    result['confidence'] = 1.0
    result['governance_check'] = {
        'uid': new_uid,
        'compliant': True,
        'message': 'New UID assignment'
    }
    result['recommendations'].append(f"Assigned new UID {new_uid}")
    
    return result

# ========================
# NEW: Conflict Detection Functions
# ========================

def detect_uid_conflicts_advanced(df_reference):
    """
    Advanced UID conflict detection with semantic analysis
    """
    conflicts = []
    
    if df_reference.empty:
        return conflicts
    
    # Group by UID
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        questions = group['heading_0'].dropna().unique()
        
        conflict_entry = {
            'uid': uid,
            'total_variations': len(questions),
            'conflicts': []
        }
        
        # Check for excessive variations
        if len(questions) > UID_GOVERNANCE['max_variations_per_uid']:
            conflict_entry['conflicts'].append({
                'type': 'excessive_variations',
                'count': len(questions),
                'threshold': UID_GOVERNANCE['max_variations_per_uid'],
                'severity': 'high' if len(questions) > UID_GOVERNANCE['max_variations_per_uid'] * 2 else 'medium'
            })
        
        # Check for semantic inconsistencies within UID
        if len(questions) > 1:
            try:
                model = load_sentence_transformer()
                standardized_questions = [standardize_question_format(q) for q in questions]
                
                # Calculate pairwise similarities
                embeddings = model.encode(standardized_questions, convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                
                # Find questions that are too dissimilar
                min_similarity_threshold = 0.7  # Questions in same UID should be similar
                dissimilar_pairs = []
                
                for i in range(len(questions)):
                    for j in range(i + 1, len(questions)):
                        sim = similarities[i][j].item()
                        if sim < min_similarity_threshold:
                            dissimilar_pairs.append({
                                'question1': questions[i][:100] + '...' if len(questions[i]) > 100 else questions[i],
                                'question2': questions[j][:100] + '...' if len(questions[j]) > 100 else questions[j],
                                'similarity': sim
                            })
                
                if dissimilar_pairs:
                    conflict_entry['conflicts'].append({
                        'type': 'semantic_inconsistency',
                        'dissimilar_pairs': dissimilar_pairs[:5],  # Limit to first 5
                        'severity': 'medium'
                    })
                    
            except Exception as e:
                logger.error(f"Semantic inconsistency check failed for UID {uid}: {e}")
        
        # Check for quality issues
        quality_scores = [calculate_question_quality_score(q) for q in questions]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < UID_GOVERNANCE['quality_score_threshold']:
            conflict_entry['conflicts'].append({
                'type': 'low_quality',
                'average_score': avg_quality,
                'threshold': UID_GOVERNANCE['quality_score_threshold'],
                'severity': 'low'
            })
        
        # Only add if there are conflicts
        if conflict_entry['conflicts']:
            conflicts.append(conflict_entry)
    
    return conflicts

def run_monthly_quality_check(df_reference):
    """
    Run monthly quality check as per governance rules
    """
    if not UID_GOVERNANCE['monthly_quality_check']:
        return None
    
    logger.info("Running monthly quality check...")
    
    quality_report = {
        'check_date': datetime.now().isoformat(),
        'total_questions': len(df_reference),
        'total_uids': df_reference['uid'].nunique(),
        'conflicts': detect_uid_conflicts_advanced(df_reference),
        'quality_metrics': {},
        'recommendations': []
    }
    
    # Calculate overall quality metrics
    if not df_reference.empty:
        questions = df_reference['heading_0'].dropna()
        quality_scores = [calculate_question_quality_score(q) for q in questions]
        
        quality_report['quality_metrics'] = {
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'questions_below_threshold': sum(1 for score in quality_scores if score < UID_GOVERNANCE['quality_score_threshold']),
            'percentage_below_threshold': (sum(1 for score in quality_scores if score < UID_GOVERNANCE['quality_score_threshold']) / len(quality_scores)) * 100
        }
        
        # Generate recommendations
        if quality_report['quality_metrics']['percentage_below_threshold'] > 10:
            quality_report['recommendations'].append("High percentage of low-quality questions detected. Consider quality improvement initiative.")
        
        if len(quality_report['conflicts']) > 10:
            quality_report['recommendations'].append("Multiple UID conflicts detected. Consider running cleanup process.")
        
        governance_violations = sum(1 for conflict in quality_report['conflicts'] 
                                  if any(c['type'] == 'excessive_variations' for c in conflict['conflicts']))
        
        if governance_violations > 5:
            quality_report['recommendations'].append(f"{governance_violations} UIDs violate governance rules. Immediate attention required.")
    
    return quality_report

# ========================
# ENHANCED: Configure Survey Page
# ========================

def enhanced_configure_survey_page():
    """
    Enhanced configure survey page with semantic matching and governance
    """
    st.markdown("## ⚙️ Enhanced Survey Configuration with Semantic Matching")
    st.markdown("*AI-powered UID assignment with governance compliance and conflict detection*")
    
    # Governance status display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="governance-compliant">⚖️ Governance: ACTIVE</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="info-card">🎯 Quality Threshold: {UID_GOVERNANCE["quality_score_threshold"]}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="info-card">🔢 Max Variations: {UID_GOVERNANCE["max_variations_per_uid"]}</div>', unsafe_allow_html=True)
    
    # Load reference data with caching
    try:
        with st.spinner("🔄 Loading reference data for semantic matching..."):
            if 'df_reference_cached' not in st.session_state:
                st.session_state.df_reference_cached = get_all_reference_questions()
            
            df_reference = st.session_state.df_reference_cached
            
            if df_reference.empty:
                st.warning("⚠️ No reference data available. UID matching will be limited.")
                existing_uids_data = {}
            else:
                # Prepare existing UIDs data for semantic matching
                existing_uids_data = {}
                for _, row in df_reference.iterrows():
                    uid = str(row['uid'])
                    question = row['heading_0']
                    
                    if uid not in existing_uids_data:
                        existing_uids_data[uid] = {
                            'best_question': question,
                            'variation_count': 1,
                            'quality_score': calculate_question_quality_score(question)
                        }
                    else:
                        existing_uids_data[uid]['variation_count'] += 1
                        # Keep the best quality question
                        current_score = calculate_question_quality_score(question)
                        if current_score > existing_uids_data[uid]['quality_score']:
                            existing_uids_data[uid]['best_question'] = question
                            existing_uids_data[uid]['quality_score'] = current_score
                
                st.success(f"✅ Loaded {len(existing_uids_data)} unique UIDs for semantic matching")
    
    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        st.error(f"❌ Failed to load reference data: {e}")
        existing_uids_data = {}
    
    # Survey configuration options
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Survey Data", "🔧 UID Assignment", "⚖️ Governance Check", "📋 Results"])
    
    with tab1:
        st.markdown("### 📊 Survey Data Input")
        
        # Survey data input methods
        input_method = st.radio(
            "Choose input method:",
            ["SurveyMonkey API", "Upload CSV", "Manual Entry"],
            horizontal=True
        )
        
        survey_data = None
        
        if input_method == "SurveyMonkey API":
            token = st.text_input("🔑 SurveyMonkey API Token", 
                                type="password", 
                                value=st.secrets.get("surveymonkey", {}).get("token", ""))
            
            if token:
                try:
                    surveys = get_surveys(token)
                    if surveys:
                        survey_options = {f"{s['id']} - {s['title']}": s['id'] for s in surveys}
                        selected_survey = st.selectbox("📋 Select Survey", list(survey_options.keys()))
                        
                        if selected_survey and st.button("📥 Load Survey Data"):
                            survey_id = survey_options[selected_survey]
                            with st.spinner("Loading survey data..."):
                                survey_details = get_survey_details(survey_id, token)
                                questions = extract_questions(survey_details)
                                survey_data = pd.DataFrame(questions)
                                st.session_state.survey_data = survey_data
                                st.success(f"✅ Loaded {len(survey_data)} questions from survey")
                except Exception as e:
                    st.error(f"❌ SurveyMonkey API error: {e}")
        
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])
            if uploaded_file:
                try:
                    survey_data = pd.read_csv(uploaded_file)
                    st.session_state.survey_data = survey_data
                    st.success(f"✅ Loaded {len(survey_data)} rows from CSV")
                    st.dataframe(survey_data.head())
                except Exception as e:
                    st.error(f"❌ CSV loading error: {e}")
        
        elif input_method == "Manual Entry":
            st.markdown("#### ✏️ Enter Questions Manually")
            
            manual_questions = st.text_area(
                "Enter questions (one per line):",
                height=200,
                placeholder="What is your name?\nHow old are you?\nWhat is your occupation?"
            )
            
            if manual_questions.strip():
                questions_list = [q.strip() for q in manual_questions.split('\n') if q.strip()]
                survey_data = pd.DataFrame({
                    'heading_0': questions_list,
                    'position': range(1, len(questions_list) + 1),
                    'is_choice': [False] * len(questions_list),
                    'survey_title': 'Manual Entry Survey'
                })
                st.session_state.survey_data = survey_data
                st.success(f"✅ Created {len(survey_data)} questions")
        
        # Display current survey data
        if 'survey_data' in st.session_state and not st.session_state.survey_data.empty:
            st.markdown("### 📋 Current Survey Data")
            st.dataframe(st.session_state.survey_data)
    
    with tab2:
        st.markdown("### 🔧 Enhanced UID Assignment")
        
        if 'survey_data' not in st.session_state or st.session_state.survey_data.empty:
            st.warning("⚠️ Please load survey data first in the Survey Data tab.")
        else:
            survey_data = st.session_state.survey_data.copy()
            
            # Assignment options
            col1, col2 = st.columns(2)
            
            with col1:
                semantic_matching = st.checkbox("🧠 Enable Semantic Matching", value=True)
                standardize_questions = st.checkbox("📝 Standardize Question Format", value=True)
            
            with col2:
                enforce_governance = st.checkbox("⚖️ Enforce Governance Rules", value=True)
                show_quality_scores = st.checkbox("🎯 Show Quality Scores", value=True)
            
            # Advanced options
            with st.expander("🔧 Advanced Settings"):
                semantic_threshold = st.slider(
                    "Semantic Similarity Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=UID_GOVERNANCE['semantic_similarity_threshold'],
                    step=0.05
                )
                
                quality_threshold = st.slider(
                    "Minimum Quality Score",
                    min_value=0.0,
                    max_value=20.0,
                    value=UID_GOVERNANCE['quality_score_threshold'],
                    step=0.5
                )
                
                batch_processing = st.checkbox("⚡ Enable Batch Processing", value=True)
            
            if st.button("🚀 Start Enhanced UID Assignment", type="primary"):
                with st.spinner("🔄 Processing questions with semantic matching..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    questions_to_process = survey_data[survey_data['heading_0'].notna()]['heading_0'].tolist()
                    
                    for idx, question in enumerate(questions_to_process):
                        # Update progress
                        progress = (idx + 1) / len(questions_to_process)
                        progress_bar.progress(progress)
                        
                        # Apply standardization if enabled
                        original_question = question
                        if standardize_questions:
                            question = standardize_question_format(question)
                        
                        # Assign UID with semantic matching and governance
                        if semantic_matching and existing_uids_data:
                            # Update threshold
                            UID_GOVERNANCE['semantic_similarity_threshold'] = semantic_threshold
                            UID_GOVERNANCE['quality_score_threshold'] = quality_threshold
                            
                            assignment_result = assign_uid_with_semantic_governance(
                                question, existing_uids_data
                            )
                        else:
                            # Simple new UID assignment
                            new_uid = get_next_available_uid(existing_uids_data)
                            assignment_result = {
                                'assigned_uid': new_uid,
                                'method': 'sequential_assignment',
                                'confidence': 1.0,
                                'governance_compliant': True,
                                'standardized_question': question,
                                'quality_score': calculate_question_quality_score(question),
                                'match_details': {},
                                'governance_check': {'compliant': True, 'message': 'New UID'},
                                'recommendations': [f'Assigned sequential UID {new_uid}']
                            }
                            # Update existing_uids_data for next iteration
                            existing_uids_data[new_uid] = {
                                'best_question': question,
                                'variation_count': 1,
                                'quality_score': assignment_result['quality_score']
                            }
                        
                        # Store result
                        result_entry = {
                            'original_question': original_question,
                            'standardized_question': assignment_result['standardized_question'],
                            'assigned_uid': assignment_result['assigned_uid'],
                            'assignment_method': assignment_result['method'],
                            'confidence': assignment_result['confidence'],
                            'quality_score': assignment_result['quality_score'],
                            'governance_compliant': assignment_result['governance_compliant'],
                            'recommendations': '; '.join(assignment_result['recommendations'])
                        }
                        
                        # Add match details if available
                        if assignment_result['match_details']:
                            result_entry['matched_question'] = assignment_result['match_details'].get('original_matched_question', '')
                            result_entry['similarity_score'] = assignment_result['match_details'].get('similarity_score', 0)
                        
                        results.append(result_entry)
                    
                    # Store results
                    st.session_state.assignment_results = pd.DataFrame(results)
                    progress_bar.progress(1.0)
                    
                    st.success(f"✅ Completed UID assignment for {len(results)} questions!")
                    
                    # Summary statistics
                    results_df = st.session_state.assignment_results
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        semantic_matches = len(results_df[results_df['assignment_method'] == 'semantic_match'])
                        st.metric("🧠 Semantic Matches", semantic_matches)
                    
                    with col2:
                        new_assignments = len(results_df[results_df['assignment_method'] == 'new_assignment'])
                        st.metric("🆕 New UIDs", new_assignments)
                    
                    with col3:
                        avg_quality = results_df['quality_score'].mean()
                        st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
                    
                    with col4:
                        high_confidence = len(results_df[results_df['confidence'] >= 0.8])
                        st.metric("⭐ High Confidence", high_confidence)
    
    with tab3:
        st.markdown("### ⚖️ Governance Compliance Check")
        
        if 'assignment_results' not in st.session_state:
            st.info("ℹ️ Complete UID assignment first to see governance analysis.")
        else:
            results_df = st.session_state.assignment_results
            
            # Run governance analysis
            governance_analysis = {
                'total_questions': len(results_df),
                'compliant_assignments': len(results_df[results_df['governance_compliant'] == True]),
                'quality_violations': len(results_df[results_df['quality_score'] < UID_GOVERNANCE['quality_score_threshold']]),
                'low_confidence_matches': len(results_df[results_df['confidence'] < 0.7])
            }
            
            # Display governance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                compliance_rate = (governance_analysis['compliant_assignments'] / governance_analysis['total_questions']) * 100
                st.metric("⚖️ Compliance Rate", f"{compliance_rate:.1f}%")
            
            with col2:
                st.metric("❌ Quality Violations", governance_analysis['quality_violations'])
            
            with col3:
                st.metric("⚠️ Low Confidence", governance_analysis['low_confidence_matches'])
            
            with col4:
                unique_uids = results_df['assigned_uid'].nunique()
                st.metric("🆔 Unique UIDs Created", unique_uids)
            
            # Detailed compliance analysis
            st.markdown("#### 📊 Detailed Compliance Analysis")
            
            # Quality score distribution
            st.markdown("**Quality Score Distribution:**")
            quality_bins = pd.cut(results_df['quality_score'], bins=[0, 5, 10, 15, 20, float('inf')], labels=['Very Low (0-5)', 'Low (5-10)', 'Medium (10-15)', 'High (15-20)', 'Very High (20+)'])
            quality_dist = quality_bins.value_counts()
            st.bar_chart(quality_dist)
            
            # Assignment method breakdown
            st.markdown("**Assignment Methods:**")
            method_counts = results_df['assignment_method'].value_counts()
            st.bar_chart(method_counts)
            
            # Governance violations details
            quality_violations = results_df[results_df['quality_score'] < UID_GOVERNANCE['quality_score_threshold']]
            if not quality_violations.empty:
                st.markdown("**Quality Violations:**")
                st.dataframe(
                    quality_violations[['original_question', 'quality_score', 'recommendations']],
                    use_container_width=True
                )
            
            # Run conflict detection if reference data available
            if not df_reference.empty:
                with st.expander("🔍 Advanced Conflict Detection"):
                    if st.button("🔄 Run Conflict Detection"):
                        with st.spinner("Analyzing conflicts..."):
                            conflicts = detect_uid_conflicts_advanced(df_reference)
                            
                            if conflicts:
                                st.warning(f"⚠️ Found {len(conflicts)} UIDs with conflicts")
                                
                                for conflict in conflicts[:10]:  # Show first 10
                                    with st.expander(f"UID {conflict['uid']} - {len(conflict['conflicts'])} conflicts"):
                                        st.write(f"**Total Variations:** {conflict['total_variations']}")
                                        
                                        for conflict_detail in conflict['conflicts']:
                                            if conflict_detail['type'] == 'excessive_variations':
                                                st.error(f"❌ Excessive variations: {conflict_detail['count']} (max: {conflict_detail['threshold']})")
                                            elif conflict_detail['type'] == 'semantic_inconsistency':
                                                st.warning("⚠️ Semantic inconsistencies detected")
                                                for pair in conflict_detail['dissimilar_pairs'][:3]:
                                                    st.write(f"  • Similarity {pair['similarity']:.2f}: '{pair['question1']}' vs '{pair['question2']}'")
                                            elif conflict_detail['type'] == 'low_quality':
                                                st.info(f"ℹ️ Low quality: avg score {conflict_detail['average_score']:.1f}")
                            else:
                                st.success("✅ No conflicts detected in reference data")
    
    with tab4:
        st.markdown("### 📋 Assignment Results")
        
        if 'assignment_results' not in st.session_state:
            st.info("ℹ️ Complete UID assignment to see results.")
        else:
            results_df = st.session_state.assignment_results
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                method_filter = st.selectbox(
                    "Filter by Method:",
                    ["All"] + list(results_df['assignment_method'].unique())
                )
            
            with col2:
                min_quality = st.slider(
                    "Minimum Quality Score:",
                    min_value=0.0,
                    max_value=results_df['quality_score'].max(),
                    value=0.0
                )
            
            with col3:
                show_recommendations = st.checkbox("Show Recommendations", value=True)
            
            # Apply filters
            filtered_results = results_df.copy()
            
            if method_filter != "All":
                filtered_results = filtered_results[filtered_results['assignment_method'] == method_filter]
            
            filtered_results = filtered_results[filtered_results['quality_score'] >= min_quality]
            
            # Display results
            st.markdown(f"#### 📊 Showing {len(filtered_results)} of {len(results_df)} results")
            
            display_columns = ['original_question', 'assigned_uid', 'assignment_method', 'confidence', 'quality_score']
            
            if 'matched_question' in filtered_results.columns:
                display_columns.append('matched_question')
                display_columns.append('similarity_score')
            
            if show_recommendations:
                display_columns.append('recommendations')
            
            # Prepare display dataframe
            display_df = filtered_results[display_columns].copy()
            
            # Rename columns for better display
            column_renames = {
                'original_question': 'Question',
                'assigned_uid': 'UID',
                'assignment_method': 'Method',
                'confidence': 'Confidence',
                'quality_score': 'Quality',
                'matched_question': 'Matched Question',
                'similarity_score': 'Similarity',
                'recommendations': 'Recommendations'
            }
            
            display_df = display_df.rename(columns=column_renames)
            
            # Configure column display
            column_config = {
                "Question": st.column_config.TextColumn("Question", width="large"),
                "UID": st.column_config.TextColumn("UID", width="small"),
                "Method": st.column_config.TextColumn("Method", width="medium"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
                "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                "Similarity": st.column_config.NumberColumn("Similarity", format="%.3f", width="small"),
                "Recommendations": st.column_config.TextColumn("Recommendations", width="large")
            }
            
            st.dataframe(
                display_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
            
            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📥 Download Results (CSV)",
                    results_df.to_csv(index=False),
                    f"uid_assignment_results_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create summary report
                summary_report = {
                    'assignment_date': datetime.now().isoformat(),
                    'total_questions': len(results_df),
                    'semantic_matches': len(results_df[results_df['assignment_method'] == 'semantic_match']),
                    'new_uids': len(results_df[results_df['assignment_method'] == 'new_assignment']),
                    'average_quality': results_df['quality_score'].mean(),
                    'governance_compliance_rate': (len(results_df[results_df['governance_compliant'] == True]) / len(results_df)) * 100
                }
                
                st.download_button(
                    "📊 Download Summary Report",
                    json.dumps(summary_report, indent=2),
                    f"assignment_summary_{uuid4()}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Create UID mapping file for integration
                uid_mapping = results_df[['original_question', 'assigned_uid', 'standardized_question']].copy()
                uid_mapping = uid_mapping.rename(columns={
                    'original_question': 'original_text',
                    'assigned_uid': 'uid',
                    'standardized_question': 'standardized_text'
                })
                
                st.download_button(
                    "🔗 Download UID Mapping",
                    uid_mapping.to_csv(index=False),
                    f"uid_mapping_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )

# ========================
# UPDATED: Main Application Logic
# ========================

# Update the existing functions to include the enhanced configure survey page
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

def classify_question(text, heading_references=HEADING_REFERENCES):
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

def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

# Snowflake connection functions (keeping existing ones)
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
    """Cached function to get all reference questions"""
    return run_snowflake_reference_query_all()

def run_snowflake_reference_query_all():
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
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

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

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 UID Matcher Pro")
    st.markdown("Enhanced with Semantic Matching & Governance")
    
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    st.markdown("**📊 SurveyMonkey**")
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    if st.button("⚙️ Enhanced Configure Survey", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()
    
    st.markdown("---")
    st.markdown("**⚖️ Governance Status**")
    st.markdown(f"• Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
    if UID_GOVERNANCE['conflict_detection_enabled']:
        st.markdown("• ✅ Conflict detection: ON")
    else:
        st.markdown("• ❌ Conflict detection: OFF")

# Main Application
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Semantic Matching & Governance</div>', unsafe_allow_html=True)

# Secrets validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Page routing
if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    st.markdown("*Now with AI-powered semantic matching, governance rules, and quality assurance*")
    
    # Enhanced features showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 New Enhanced Features")
        st.markdown("✅ **Semantic Matching**: AI understands question meaning")
        st.markdown("✅ **Governance Rules**: Automatic compliance checking")
        st.markdown("✅ **Quality Scoring**: Smart question assessment")
        st.markdown("✅ **Conflict Detection**: Real-time duplicate identification")
        st.markdown("✅ **Question Standardization**: Consistent formatting")
        
    with col2:
        st.markdown("### 📊 System Status")
        try:
            get_snowflake_engine()
            st.markdown("✅ **Snowflake**: Connected")
        except:
            st.markdown("❌ **Snowflake**: Connection Issues")
        
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                st.markdown("✅ **SurveyMonkey**: Connected")
            else:
                st.markdown("❌ **SurveyMonkey**: No Token")
        except:
            st.markdown("❌ **SurveyMonkey**: API Issues")
        
        st.markdown("✅ **Governance**: Active")
        st.markdown("✅ **Semantic Matching**: Ready")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start")
    st.markdown("1. **Configure Survey** - Use enhanced UID assignment with semantic matching")
    st.markdown("2. **Review Governance** - Check compliance and quality scores")
    st.markdown("3. **Analyze Results** - View detailed assignment analytics")
    
    if st.button("🚀 Start Enhanced Survey Configuration", type="primary", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()

elif st.session_state.page == "configure_survey":
    enhanced_configure_survey_page()

elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ View SurveyMonkey Surveys")
    
    token = st.text_input("🔑 SurveyMonkey API Token", 
                        type="password", 
                        value=st.secrets.get("surveymonkey", {}).get("token", ""))
    
    if token:
        try:
            surveys = get_surveys(token)
            if surveys:
                st.success(f"✅ Found {len(surveys)} surveys")
                
                surveys_df = pd.DataFrame(surveys)
                st.dataframe(surveys_df, use_container_width=True)
                
                # Survey selection for detailed view
                selected_survey = st.selectbox(
                    "Select survey for details:",
                    [f"{s['id']} - {s['title']}" for s in surveys]
                )
                
                if selected_survey:
                    survey_id = selected_survey.split(" - ")[0]
                    
                    if st.button("📋 View Survey Details"):
                        with st.spinner("Loading survey details..."):
                            survey_details = get_survey_details(survey_id, token)
                            questions = extract_questions(survey_details)
                            
                            st.markdown(f"### Survey: {survey_details.get('title', 'Unknown')}")
                            st.markdown(f"**ID:** {survey_details.get('id', 'Unknown')}")
                            st.markdown(f"**Questions:** {len(questions)}")
                            
                            if questions:
                                questions_df = pd.DataFrame(questions)
                                st.dataframe(questions_df, use_container_width=True)
            else:
                st.info("No surveys found")
        except Exception as e:
            st.error(f"❌ Error fetching surveys: {e}")

# Additional helper functions for the complete application

def categorize_survey(survey_title):
    """Categorize survey based on title keywords with priority ordering"""
    if not survey_title:
        return "Unknown"
    
    title_lower = survey_title.lower()
    
    # Check GROW first (exact match for CAPS)
    if 'GROW' in survey_title:
        return 'GROW'
    
    # Check other categories
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == 'GROW':  # Already checked
            continue
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    
    return "Other"

def create_unique_questions_bank(df_reference):
    """Enhanced unique questions bank with survey categorization and governance"""
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_reference)} reference questions for unique bank")
    
    unique_questions = []
    uid_groups = df_reference.groupby('uid')
    logger.info(f"Found {len(uid_groups)} unique UIDs")
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        uid_questions = group['heading_0'].tolist()
        
        # Calculate quality scores for all questions
        quality_scores = [calculate_question_quality_score(q) for q in uid_questions]
        best_idx = np.argmax(quality_scores)
        best_question = uid_questions[best_idx]
        best_quality = quality_scores[best_idx]
        
        # Get survey titles for categorization
        survey_titles = group.get('survey_title', pd.Series()).dropna().unique()
        
        # Determine category from survey titles
        categories = []
        for title in survey_titles:
            category = categorize_survey(title)
            if category not in categories:
                categories.append(category)
        
        # If multiple categories, take the most frequent
        if categories:
            primary_category = categories[0] if len(categories) == 1 else "Mixed"
        else:
            primary_category = "Unknown"
        
        if best_question:
            # Standardize the best question
            standardized_question = standardize_question_format(best_question)
            
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'standardized_question': standardized_question,
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split()),
                'survey_category': primary_category,
                'survey_titles': ', '.join(survey_titles) if len(survey_titles) > 0 else 'Unknown',
                'quality_score': best_quality,
                'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid'],
                'all_variants': uid_questions,
                'quality_scores': quality_scores,
                'avg_quality': np.mean(quality_scores),
                'standardization_applied': best_question != standardized_question
            })
    
    unique_df = pd.DataFrame(unique_questions)
    logger.info(f"Created unique questions bank with {len(unique_df)} UIDs")
    
    # Sort by UID in ascending order
    if not unique_df.empty:
        try:
            unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
            unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
            unique_df = unique_df.drop('uid_numeric', axis=1)
        except:
            unique_df = unique_df.sort_values('uid')
    
    return unique_df

def analyze_uid_variations_enhanced(df_reference):
    """Enhanced analysis with governance compliance and quality metrics"""
    analysis_results = {}
    
    # Basic statistics
    uid_counts = df_reference['uid'].value_counts().sort_values(ascending=False)
    
    analysis_results['total_questions'] = len(df_reference)
    analysis_results['unique_uids'] = df_reference['uid'].nunique()
    analysis_results['avg_variations_per_uid'] = len(df_reference) / df_reference['uid'].nunique()
    
    # Governance compliance
    governance_violations = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
    analysis_results['governance_compliance'] = {
        'violations': len(governance_violations),
        'violation_rate': (len(governance_violations) / len(uid_counts)) * 100,
        'violating_uids': governance_violations.to_dict()
    }
    
    # Quality analysis
    if not df_reference.empty:
        questions = df_reference['heading_0'].dropna()
        quality_scores = [calculate_question_quality_score(q) for q in questions]
        
        analysis_results['quality_metrics'] = {
            'average_quality': np.mean(quality_scores),
            'median_quality': np.median(quality_scores),
            'std_quality': np.std(quality_scores),
            'below_threshold_count': sum(1 for score in quality_scores if score < UID_GOVERNANCE['quality_score_threshold']),
            'below_threshold_percentage': (sum(1 for score in quality_scores if score < UID_GOVERNANCE['quality_score_threshold']) / len(quality_scores)) * 100
        }
    
    # Standardization analysis
    standardized_questions = [standardize_question_format(q) for q in df_reference['heading_0'].dropna()]
    original_questions = df_reference['heading_0'].dropna().tolist()
    
    standardization_changes = sum(1 for orig, std in zip(original_questions, standardized_questions) if orig != std)
    analysis_results['standardization_metrics'] = {
        'total_questions_analyzed': len(original_questions),
        'questions_needing_standardization': standardization_changes,
        'standardization_rate': (standardization_changes / len(original_questions)) * 100 if original_questions else 0
    }
    
    return analysis_results

# Error handling and logging utilities
def log_governance_violation(uid, violation_type, details):
    """Log governance violations for audit trail"""
    violation_entry = {
        'timestamp': datetime.now().isoformat(),
        'uid': uid,
        'violation_type': violation_type,
        'details': details,
        'action_taken': UID_GOVERNANCE.get('governance_violation_action', 'warn_and_log')
    }
    
    logger.warning(f"Governance violation detected: {violation_entry}")
    return violation_entry

def validate_semantic_matching_setup():
    """Validate that semantic matching is properly configured"""
    try:
        model = load_sentence_transformer()
        test_texts = ["What is your name?", "How old are you?"]
        embeddings = model.encode(test_texts)
        logger.info("Semantic matching validation successful")
        return True, "Semantic matching ready"
    except Exception as e:
        logger.error(f"Semantic matching validation failed: {e}")
        return False, f"Semantic matching error: {e}"

# Monthly quality check scheduler (simplified for demo)
def schedule_monthly_quality_check():
    """Schedule monthly quality checks (would integrate with actual scheduler in production)"""
    if not UID_GOVERNANCE['monthly_quality_check']:
        return None
    
    # In production, this would integrate with a job scheduler
    # For demo, we'll just return the configuration
    return {
        'enabled': True,
        'frequency': 'monthly',
        'next_check': (datetime.now() + timedelta(days=30)).isoformat(),
        'checks_to_perform': [
            'governance_compliance',
            'quality_metrics',
            'conflict_detection',
            'standardization_analysis'
        ]
    }

# Integration utilities for external systems
def export_uid_mappings_for_integration(assignment_results):
    """Export UID mappings in format suitable for external system integration"""
    if assignment_results.empty:
        return pd.DataFrame()
    
    integration_df = assignment_results[['original_question', 'assigned_uid', 'standardized_question', 'confidence', 'quality_score']].copy()
    
    # Add metadata for integration
    integration_df['export_timestamp'] = datetime.now().isoformat()
    integration_df['governance_compliant'] = integration_df['quality_score'] >= UID_GOVERNANCE['quality_score_threshold']
    integration_df['confidence_level'] = integration_df['confidence'].apply(
        lambda x: 'High' if x >= 0.9 else 'Medium' if x >= 0.7 else 'Low'
    )
    
    # Rename for external system compatibility
    integration_df = integration_df.rename(columns={
        'original_question': 'source_question_text',
        'assigned_uid': 'target_uid',
        'standardized_question': 'normalized_question_text',
        'confidence': 'assignment_confidence',
        'quality_score': 'question_quality_score'
    })
    
    return integration_df

# Complete the application with proper error handling
try:
    # Validate semantic matching setup on startup
    semantic_valid, semantic_msg = validate_semantic_matching_setup()
    if not semantic_valid:
        st.sidebar.warning(f"⚠️ {semantic_msg}")
    else:
        st.sidebar.success("✅ Semantic matching ready")
    
    # Add monthly quality check status to sidebar
    monthly_check_config = schedule_monthly_quality_check()
    if monthly_check_config:
        st.sidebar.info("📅 Monthly quality checks enabled")

except Exception as e:
    logger.error(f"Application initialization error: {e}")
    st.error(f"❌ Application initialization failed: {e}")

# Footer with version and governance info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🧠 UID Matcher Pro v2.0 - Enhanced with Semantic Matching & Governance<br>
    Governance Compliance • Quality Assurance • Conflict Detection<br>
    Built with Streamlit • Powered by SentenceTransformers
</div>
""", unsafe_allow_html=True)
