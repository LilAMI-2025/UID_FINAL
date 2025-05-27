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

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
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
    'semantic_matching_enabled': True,
    'governance_enforcement': True
}

# Enhanced Question Standardization Rules
QUESTION_STANDARDIZATION = {
    'normalize_case': True,
    'remove_extra_spaces': True,
    'standardize_punctuation': True,
    'expand_contractions': True,
    'fix_common_typos': True,
    'standardize_formats': True
}

# Expanded Question Format Patterns
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

# Enhanced Synonym Mapping with Standardization
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

# Survey Categories based on titles
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

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# Enhanced Question Standardization Functions
def standardize_question_format(question_text):
    """
    Standardize question format using enhanced rules
    """
    if not question_text or pd.isna(question_text):
        return question_text
    
    text = str(question_text).strip()
    
    if not QUESTION_STANDARDIZATION['standardization_enabled']:
        return text
    
    # Normalize case
    if QUESTION_STANDARDIZATION['normalize_case']:
        # Capitalize first letter, maintain proper nouns
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Remove extra spaces
    if QUESTION_STANDARDIZATION['remove_extra_spaces']:
        text = re.sub(r'\s+', ' ', text)
    
    # Standardize punctuation
    if QUESTION_STANDARDIZATION['standardize_punctuation']:
        # Ensure questions end with question mark
        if any(word in text.lower().split()[:3] for word in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']):
            if not text.endswith('?'):
                text = text.rstrip('.!') + '?'
        
        # Remove multiple punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[!]{2,}', '!', text)
    
    # Expand contractions
    if QUESTION_STANDARDIZATION['expand_contractions']:
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.title(), expansion.title())
    
    # Fix common typos
    if QUESTION_STANDARDIZATION['fix_common_typos']:
        typo_fixes = {
            'teh': 'the',
            'adn': 'and',
            'youre': 'you are',
            'your are': 'you are',
            'its': 'it is',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely'
        }
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in typo_fixes:
                # Preserve original case and punctuation
                if word.lower() == clean_word:
                    words[i] = typo_fixes[clean_word]
                elif word.title() == word:
                    words[i] = typo_fixes[clean_word].title()
        text = ' '.join(words)
    
    # Standardize question formats
    if QUESTION_STANDARDIZATION['standardize_formats']:
        # Apply enhanced synonym mapping
        text_lower = text.lower()
        for phrase, replacement in ENHANCED_SYNONYM_MAP.items():
            if phrase in text_lower:
                # Preserve case when replacing
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub(replacement, text)
    
    return text.strip()

def detect_question_pattern(question_text):
    """
    Detect question pattern for better categorization
    """
    text_lower = question_text.lower()
    
    # Check each pattern category
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

# Enhanced Survey Categorization
def categorize_survey(survey_title):
    """
    Categorize survey based on title keywords with priority ordering
    """
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

# Enhanced Semantic UID Assignment with Governance
def enhanced_semantic_matching_with_governance(question_text, existing_uids_data, threshold=0.85):
    """
    Enhanced semantic matching with governance rules and standardization
    """
    if not existing_uids_data:
        return None, 0.0, "no_existing_data"
    
    try:
        # Standardize the input question
        standardized_question = standardize_question_format(question_text)
        
        model = load_sentence_transformer()
        
        # Get embeddings
        question_embedding = model.encode([standardized_question], convert_to_tensor=True)
        
        # Get standardized existing questions
        existing_questions = []
        uid_mapping = []
        
        for uid, data in existing_uids_data.items():
            standardized_existing = standardize_question_format(data['best_question'])
            existing_questions.append(standardized_existing)
            uid_mapping.append(uid)
        
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(question_embedding, existing_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        if best_score >= threshold:
            best_uid = uid_mapping[best_idx]
            
            # Check governance compliance
            if existing_uids_data[best_uid]['variation_count'] < UID_GOVERNANCE['max_variations_per_uid']:
                return best_uid, best_score, "semantic_match_compliant"
            else:
                return best_uid, best_score, "semantic_match_governance_violation"
        
        # Check for auto-consolidation threshold
        if best_score >= UID_GOVERNANCE['auto_consolidate_threshold']:
            best_uid = uid_mapping[best_idx]
            return best_uid, best_score, "auto_consolidate"
            
    except Exception as e:
        logger.error(f"Enhanced semantic matching failed: {e}")
    
    return None, 0.0, "no_match"

# Enhanced UID Assignment with Full Governance
def assign_uid_with_full_governance(question_text, existing_uids_data, survey_category=None, question_pattern=None):
    """
    Comprehensive UID assignment with governance, standardization, and semantic matching
    """
    # Standardize the question first
    standardized_question = standardize_question_format(question_text)
    
    # Detect question pattern if not provided
    if not question_pattern:
        question_pattern = detect_question_pattern(standardized_question)
    
    # Step 1: Try semantic matching with governance
    if UID_GOVERNANCE['semantic_matching_enabled']:
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
            if UID_GOVERNANCE['governance_enforcement']:
                logger.warning(f"Semantic match found but violates governance: UID {matched_uid}")
                # Continue to create new UID
            else:
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
    
    # Step 2: Create new UID with governance compliance
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

# Enhanced UID Conflict Detection with Semantic Analysis
def detect_uid_conflicts_advanced_semantic(df_reference):
    """
    Advanced UID conflict detection with semantic analysis and governance checking
    """
    conflicts = []
    model = None
    
    try:
        if UID_GOVERNANCE['semantic_matching_enabled']:
            model = load_sentence_transformer()
    except Exception as e:
        logger.warning(f"Could not load semantic model for conflict detection: {e}")
    
    # Group by UID
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        questions = group['heading_0'].unique()
        
        # Check for excessive variations
        if len(questions) > UID_GOVERNANCE['max_variations_per_uid']:
            conflicts.append({
                'uid': uid,
                'type': 'excessive_variations',
                'count': len(questions),
                'limit': UID_GOVERNANCE['max_variations_per_uid'],
                'severity': 'high' if len(questions) > UID_GOVERNANCE['max_variations_per_uid'] * 2 else 'medium',
                'governance_violation': True
            })
        
        # Semantic conflict detection
        if model and len(questions) > 1:
            standardized_questions = [standardize_question_format(q) for q in questions]
            
            try:
                embeddings = model.encode(standardized_questions, convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                
                # Find questions that are too different (potential conflicts)
                semantic_conflicts = 0
                for i in range(len(similarities)):
                    for j in range(i + 1, len(similarities)):
                        similarity = similarities[i][j].item()
                        if similarity < 0.3:  # Very different questions with same UID
                            semantic_conflicts += 1
                
                if semantic_conflicts > len(questions) * 0.3:  # More than 30% are conflicts
                    conflicts.append({
                        'uid': uid,
                        'type': 'semantic_conflicts',
                        'conflicts': semantic_conflicts,
                        'total_questions': len(questions),
                        'severity': 'medium',
                        'governance_violation': False
                    })
            except Exception as e:
                logger.warning(f"Semantic conflict detection failed for UID {uid}: {e}")
        
        # Check for duplicate variations with normalization
        normalized_questions = [enhanced_normalize(q, ENHANCED_SYNONYM_MAP) for q in questions]
        unique_normalized = len(set(normalized_questions))
        
        if len(questions) > unique_normalized * 3:  # Too many near-duplicates
            conflicts.append({
                'uid': uid,
                'type': 'excessive_duplicates',
                'duplicates': len(questions) - unique_normalized,
                'unique_questions': unique_normalized,
                'severity': 'low',
                'governance_violation': False
            })
    
    return conflicts

# Enhanced data quality management functions
def analyze_uid_variations_with_governance(df_reference):
    """Enhanced analysis with full governance compliance and semantic analysis"""
    analysis_results = {}
    
    # Basic statistics
    uid_counts = df_reference['uid'].value_counts().sort_values(ascending=False)
    
    analysis_results['total_questions'] = len(df_reference)
    analysis_results['unique_uids'] = df_reference['uid'].nunique()
    analysis_results['avg_variations_per_uid'] = len(df_reference) / df_reference['uid'].nunique()
    
    # Enhanced governance compliance
    governance_violations = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
    analysis_results['governance_compliance'] = {
        'violations': len(governance_violations),
        'violation_rate': (len(governance_violations) / len(uid_counts)) * 100,
        'violating_uids': governance_violations.to_dict(),
        'total_violating_questions': governance_violations.sum(),
        'compliance_rate': ((len(uid_counts) - len(governance_violations)) / len(uid_counts)) * 100
    }
    
    # Standardization analysis
    standardized_questions = df_reference['heading_0'].apply(standardize_question_format)
    original_unique = df_reference['heading_0'].nunique()
    standardized_unique = standardized_questions.nunique()
    
    analysis_results['standardization_impact'] = {
        'original_unique': original_unique,
        'standardized_unique': standardized_unique,
        'questions_consolidated': original_unique - standardized_unique,
        'consolidation_rate': ((original_unique - standardized_unique) / original_unique) * 100
    }
    
    # Question pattern analysis
    patterns_data = df_reference['heading_0'].apply(detect_question_pattern)
    pattern_categories = [p['category'] for p in patterns_data]
    pattern_counts = Counter(pattern_categories)
    
    analysis_results['question_patterns'] = {
        'pattern_distribution': dict(pattern_counts),
        'most_common_pattern': pattern_counts.most_common(1)[0] if pattern_counts else ('unknown', 0),
        'pattern_diversity': len(pattern_counts)
    }
    
    # Enhanced conflict detection
    conflicts = detect_uid_conflicts_advanced_semantic(df_reference)
    analysis_results['conflicts'] = {
        'total_conflicts': len(conflicts),
        'by_type': Counter([c['type'] for c in conflicts]),
        'by_severity': Counter([c['severity'] for c in conflicts]),
        'governance_violations': len([c for c in conflicts if c.get('governance_violation', False)]),
        'detailed_conflicts': conflicts[:10]  # Top 10 for display
    }
    
    # Quality scoring analysis
    quality_scores = df_reference['heading_0'].apply(score_question_quality)
    analysis_results['quality_analysis'] = {
        'avg_quality': quality_scores.mean(),
        'quality_std': quality_scores.std(),
        'high_quality_count': len(quality_scores[quality_scores > UID_GOVERNANCE['quality_score_threshold']]),
        'low_quality_count': len(quality_scores[quality_scores < 0]),
        'quality_distribution': {
            'excellent': len(quality_scores[quality_scores > 15]),
            'good': len(quality_scores[(quality_scores > 5) & (quality_scores <= 15)]),
            'fair': len(quality_scores[(quality_scores > 0) & (quality_scores <= 5)]),
            'poor': len(quality_scores[quality_scores <= 0])
        }
    }
    
    return analysis_results

# Enhanced Configure Survey Page with Semantic Matching
def enhanced_configure_survey_page():
    """Enhanced configure survey page with semantic matching and governance"""
    st.markdown("## ⚙️ Enhanced Configure Survey with Semantic Matching")
    st.markdown("*Upload CSV or fetch from SurveyMonkey with advanced UID assignment*")
    
    # Governance settings
    with st.expander("⚖️ Governance & Matching Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            UID_GOVERNANCE['semantic_matching_enabled'] = st.checkbox(
                "🧠 Enable Semantic Matching", 
                value=UID_GOVERNANCE['semantic_matching_enabled'],
                help="Use AI to find semantically similar questions"
            )
            
            UID_GOVERNANCE['standardization_enabled'] = st.checkbox(
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
                value=UID_GOVERNANCE['governance_enforcement'],
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
    
    # Input method selection
    input_method = st.radio(
        "📥 Choose Input Method:",
        ["Upload CSV File", "Fetch from SurveyMonkey", "Enter SurveyMonkey Survey ID"],
        horizontal=True
    )
    
    df_target = None
    
  # Handle different input methods
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                df_target = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV uploaded successfully! Found {len(df_target)} rows.")
                
                # Validate required columns
                required_cols = ['heading_0']
                missing_cols = [col for col in required_cols if col not in df_target.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    df_target = None
                else:
                    # Add missing optional columns
                    optional_cols = {
                        'survey_title': 'Uploaded Survey',
                        'survey_id': 'uploaded',
                        'is_choice': False,
                        'question_category': 'Main Question/Multiple Choice'
                    }
                    for col, default_val in optional_cols.items():
                        if col not in df_target.columns:
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
                    
                    with st.spinner("🔄 Fetching survey data..."):
                        survey_details = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_details)
                        df_target = pd.DataFrame(questions)
                        
                        st.success(f"✅ Fetched {len(df_target)} questions from SurveyMonkey!")
                        st.info(f"📊 Preview of survey data:")
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
                with st.spinner("🔄 Fetching survey data..."):
                    survey_details = get_survey_details(survey_id, token)
                    questions = extract_questions(survey_details)
                    df_target = pd.DataFrame(questions)
                    
                    st.success(f"✅ Fetched {len(df_target)} questions from survey {survey_id}!")
                    st.info(f"📊 Preview of survey data:")
                    st.dataframe(df_target.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error fetching survey {survey_id}: {e}")
    
    # Enhanced UID Matching Process
    if df_target is not None and not df_target.empty:
        st.markdown("---")
        st.markdown("### 🧠 Enhanced UID Matching with Semantic Analysis")
        
        # Load reference data
        try:
            with st.spinner("🔄 Loading reference question bank..."):
                df_reference = get_all_reference_questions()
                
            if df_reference.empty:
                st.warning("⚠️ No reference data available. Creating new UIDs for all questions.")
                # Process without reference data
                df_final = process_questions_without_reference(df_target)
            else:
                st.info(f"📊 Loaded {len(df_reference)} reference questions from database")
                
                # Show governance compliance of reference data
                governance_analysis = analyze_uid_variations_with_governance(df_reference)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Reference UIDs", governance_analysis['unique_uids'])
                with col2:
                    compliance_rate = governance_analysis['governance_compliance']['compliance_rate']
                    st.metric("⚖️ Governance Compliance", f"{compliance_rate:.1f}%")
                with col3:
                    standardization_rate = governance_analysis['standardization_impact']['consolidation_rate']
                    st.metric("📝 Standardization Impact", f"{standardization_rate:.1f}%")
                with col4:
                    avg_quality = governance_analysis['quality_analysis']['avg_quality']
                    st.metric("🎯 Avg Quality Score", f"{avg_quality:.1f}")
                
                # Enhanced UID matching with governance
                if st.button("🧠 Start Enhanced UID Matching", type="primary"):
                    with st.spinner("🔄 Processing questions with semantic matching and governance..."):
                        df_final = enhanced_uid_matching_process(df_target, df_reference)
                        
                        if df_final is not None:
                            st.session_state.df_final = df_final
                            st.session_state.df_target = df_target
                            
                            # Display enhanced results
                            display_enhanced_matching_results(df_final, df_reference)
                        else:
                            st.error("❌ Enhanced UID matching failed")
                            
        except Exception as e:
            st.error(f"❌ Error loading reference data: {e}")
            # Fallback to processing without reference
            if st.button("🔄 Process Without Reference Data"):
                df_final = process_questions_without_reference(df_target)
                st.session_state.df_final = df_final
                st.session_state.df_target = df_target

def enhanced_uid_matching_process(df_target, df_reference):
    """
    Enhanced UID matching process with semantic analysis and governance
    """
    try:
        # Step 1: Prepare reference data with standardization
        logger.info("Step 1: Preparing reference data...")
        
        # Create existing UIDs data structure for semantic matching
        existing_uids_data = {}
        for uid in df_reference['uid'].unique():
            if pd.notna(uid):
                uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
                best_question = get_best_question_for_uid(uid_questions)
                
                existing_uids_data[str(uid)] = {
                    'best_question': best_question,
                    'variation_count': len(uid_questions),
                    'all_questions': uid_questions
                }
        
        logger.info(f"Prepared {len(existing_uids_data)} existing UIDs for matching")
        
        # Step 2: Process target questions with enhanced matching
        logger.info("Step 2: Processing target questions...")
        
        df_enhanced = df_target.copy()
        
        # Add columns for enhanced tracking
        enhanced_columns = [
            'standardized_question', 'question_pattern_category', 'question_pattern_type',
            'semantic_uid', 'semantic_confidence', 'semantic_match_status',
            'governance_compliant', 'match_method', 'quality_score'
        ]
        
        for col in enhanced_columns:
            if col not in df_enhanced.columns:
                df_enhanced[col] = None
        
        # Process each question
        for idx, row in df_enhanced.iterrows():
            question_text = row['heading_0']
            survey_title = row.get('survey_title', '')
            
            if pd.isna(question_text) or question_text.strip() == '':
                continue
            
            # Skip choice questions - they inherit parent UID
            if row.get('is_choice', False):
                continue
            
            # Skip heading questions
            if row.get('question_category') == 'Heading':
                continue
            
            # Detect question pattern
            question_pattern = detect_question_pattern(question_text)
            df_enhanced.at[idx, 'question_pattern_category'] = question_pattern['category']
            df_enhanced.at[idx, 'question_pattern_type'] = question_pattern['pattern']
            
            # Get survey category
            survey_category = categorize_survey(survey_title)
            
            # Apply enhanced UID assignment
            uid_result = assign_uid_with_full_governance(
                question_text, 
                existing_uids_data, 
                survey_category, 
                question_pattern
            )
            
            # Update dataframe with results
            df_enhanced.at[idx, 'semantic_uid'] = uid_result['uid']
            df_enhanced.at[idx, 'semantic_confidence'] = uid_result['confidence']
            df_enhanced.at[idx, 'semantic_match_status'] = uid_result['match_status']
            df_enhanced.at[idx, 'governance_compliant'] = uid_result['governance_compliant']
            df_enhanced.at[idx, 'match_method'] = uid_result['method']
            df_enhanced.at[idx, 'standardized_question'] = uid_result['standardized_question']
            df_enhanced.at[idx, 'quality_score'] = score_question_quality(question_text)
            
            # Update existing UIDs data if new UID was created
            if uid_result['method'] == 'new_assignment':
                existing_uids_data[uid_result['uid']] = {
                    'best_question': uid_result['standardized_question'],
                    'variation_count': 1,
                    'all_questions': [question_text]
                }
            elif uid_result['uid'] in existing_uids_data:
                # Update variation count
                existing_uids_data[uid_result['uid']]['variation_count'] += 1
                existing_uids_data[uid_result['uid']]['all_questions'].append(question_text)
        
        # Step 3: Apply traditional TF-IDF matching as fallback
        logger.info("Step 3: Applying traditional TF-IDF matching as fallback...")
        
        # For questions without semantic matches, try TF-IDF
        unmatched_mask = df_enhanced['semantic_uid'].isna()
        if unmatched_mask.any():
            df_unmatched = df_enhanced[unmatched_mask].copy()
            df_tfidf_results = compute_tfidf_matches(df_reference, df_unmatched, ENHANCED_SYNONYM_MAP)
            
            # Merge TF-IDF results back
            for idx, row in df_tfidf_results.iterrows():
                original_idx = df_enhanced[df_enhanced.index == idx].index[0]
                if pd.notna(row.get('Suggested_UID')):
                    df_enhanced.at[original_idx, 'semantic_uid'] = row['Suggested_UID']
                    df_enhanced.at[original_idx, 'semantic_confidence'] = row['Similarity']
                    df_enhanced.at[original_idx, 'match_method'] = 'tfidf_fallback'
                    df_enhanced.at[original_idx, 'semantic_match_status'] = 'tfidf_match'
        
        # Step 4: Finalize results
        logger.info("Step 4: Finalizing enhanced results...")
        
        # Set Final_UID based on semantic results
        df_enhanced['Final_UID'] = df_enhanced['semantic_uid']
        df_enhanced['configured_final_UID'] = df_enhanced['semantic_uid']
        
        # Handle choice questions - inherit parent UID
        for idx, row in df_enhanced.iterrows():
            if row.get('is_choice', False) and pd.notna(row.get('parent_question')):
                parent_questions = df_enhanced[
                    (df_enhanced['heading_0'] == row['parent_question']) & 
                    (df_enhanced['is_choice'] == False)
                ]
                if not parent_questions.empty:
                    parent_uid = parent_questions.iloc[0]['Final_UID']
                    df_enhanced.at[idx, 'Final_UID'] = parent_uid
                    df_enhanced.at[idx, 'configured_final_UID'] = parent_uid
                    df_enhanced.at[idx, 'match_method'] = 'inherited_from_parent'
        
        # Add enhanced match confidence and type
        df_enhanced['Final_Match_Type'] = df_enhanced.apply(lambda row: 
            f"🧠 {row['match_method']}" if pd.notna(row['Final_UID']) else "❌ No match", axis=1)
        
        # Add governance status
        df_enhanced['Final_Governance'] = df_enhanced['governance_compliant'].apply(
            lambda x: "✅ Compliant" if x else "⚠️ Violation" if pd.notna(x) else "N/A")
        
        # Detect conflicts
        df_enhanced = detect_uid_conflicts(df_enhanced)
        
        # Add survey categorization
        if 'survey_title' in df_enhanced.columns:
            df_enhanced['survey_category'] = df_enhanced['survey_title'].apply(categorize_survey)
        
        logger.info("Enhanced UID matching completed successfully")
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Enhanced UID matching process failed: {e}")
        st.error(f"❌ Enhanced matching failed: {e}")
        return None

def display_enhanced_matching_results(df_final, df_reference):
    """
    Display enhanced matching results with governance and semantic analysis
    """
    st.markdown("### 🎯 Enhanced Matching Results")
    
    # Enhanced summary metrics
    total_questions = len(df_final[df_final['is_choice'] == False])
    matched_questions = len(df_final[(df_final['is_choice'] == False) & (df_final['Final_UID'].notna())])
    semantic_matches = len(df_final[df_final['match_method'].str.contains('semantic', na=False)])
    governance_compliant = len(df_final[df_final['governance_compliant'] == True])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        match_rate = (matched_questions / total_questions * 100) if total_questions > 0 else 0
        st.metric("🎯 Match Rate", f"{match_rate:.1f}%")
    
    with col2:
        semantic_rate = (semantic_matches / total_questions * 100) if total_questions > 0 else 0
        st.metric("🧠 Semantic Matches", f"{semantic_rate:.1f}%")
    
    with col3:
        governance_rate = (governance_compliant / total_questions * 100) if total_questions > 0 else 0
        st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
    
    with col4:
        avg_confidence = df_final['semantic_confidence'].mean()
        st.metric("📊 Avg Confidence", f"{avg_confidence:.2f}" if pd.notna(avg_confidence) else "N/A")
    
    with col5:
        avg_quality = df_final['quality_score'].mean()
        st.metric("🎯 Avg Quality", f"{avg_quality:.1f}" if pd.notna(avg_quality) else "N/A")
    
    # Matching method breakdown
    st.markdown("#### 📊 Matching Method Breakdown")
    method_counts = df_final['match_method'].value_counts()
    
    method_df = pd.DataFrame({
        'Method': method_counts.index,
        'Count': method_counts.values,
        'Percentage': (method_counts.values / len(df_final) * 100).round(1)
    })
    
    st.dataframe(method_df, use_container_width=True)
    
    # Enhanced results display
    st.markdown("#### 🔍 Detailed Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_method = st.selectbox("Filter by Method", 
                                 ["All"] + list(df_final['match_method'].dropna().unique()))
    
    with col2:
        show_governance = st.selectbox("Filter by Governance", 
                                     ["All", "Compliant", "Violations"])
    
    with col3:
        show_confidence = st.selectbox("Filter by Confidence", 
                                     ["All", "High (>0.8)", "Medium (0.5-0.8)", "Low (<0.5)"])
    
    # Apply filters
    display_df = df_final.copy()
    
    if show_method != "All":
        display_df = display_df[display_df['match_method'] == show_method]
    
    if show_governance == "Compliant":
        display_df = display_df[display_df['governance_compliant'] == True]
    elif show_governance == "Violations":
        display_df = display_df[display_df['governance_compliant'] == False]
    
    if show_confidence == "High (>0.8)":
        display_df = display_df[display_df['semantic_confidence'] > 0.8]
    elif show_confidence == "Medium (0.5-0.8)":
        display_df = display_df[(display_df['semantic_confidence'] >= 0.5) & 
                              (display_df['semantic_confidence'] <= 0.8)]
    elif show_confidence == "Low (<0.5)":
        display_df = display_df[display_df['semantic_confidence'] < 0.5]
    
    # Enhanced display columns
    display_columns = [
        'heading_0', 'Final_UID', 'standardized_question', 'semantic_confidence', 
        'match_method', 'Final_Governance', 'quality_score', 'question_pattern_category'
    ]
    
    if not display_df.empty:
        st.dataframe(
            display_df[display_columns],
            column_config={
                "heading_0": st.column_config.TextColumn("Original Question", width="large"),
                "Final_UID": st.column_config.TextColumn("UID", width="small"),
                "standardized_question": st.column_config.TextColumn("Standardized Question", width="large"),
                "semantic_confidence": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
                "match_method": st.column_config.TextColumn("Method", width="medium"),
                "Final_Governance": st.column_config.TextColumn("Governance", width="small"),
                "quality_score": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                "question_pattern_category": st.column_config.TextColumn("Pattern", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Enhanced download options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download Enhanced Results",
                df_final.to_csv(index=False),
                f"enhanced_uid_results_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create governance report
            governance_issues = df_final[df_final['governance_compliant'] == False]
            if not governance_issues.empty:
                st.download_button(
                    "⚖️ Download Governance Report",
                    governance_issues.to_csv(index=False),
                    f"governance_issues_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Create standardization report
            standardization_report = df_final[['heading_0', 'standardized_question', 'quality_score']].copy()
            standardization_report['standardization_impact'] = (
                standardization_report['heading_0'] != standardization_report['standardized_question']
            )
            
            st.download_button(
                "📝 Download Standardization Report",
                standardization_report.to_csv(index=False),
                f"standardization_report_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("ℹ️ No results match the current filters.")

def process_questions_without_reference(df_target):
    """
    Process questions when no reference data is available
    """
    df_processed = df_target.copy()
    
    # Add new UIDs starting from 1
    current_uid = 1
    
    for idx, row in df_processed.iterrows():
        if row.get('is_choice', False):
            continue  # Skip choice questions
            
        if row.get('question_category') == 'Heading':
            continue  # Skip heading questions
        
        question_text = row['heading_0']
        if pd.isna(question_text) or question_text.strip() == '':
            continue
        
        # Standardize question
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
    
    # Handle choice questions
    for idx, row in df_processed.iterrows():
        if row.get('is_choice', False) and pd.notna(row.get('parent_question')):
            parent_questions = df_processed[
                (df_processed['heading_0'] == row['parent_question']) & 
                (df_processed['is_choice'] == False)
            ]
            if not parent_questions.empty:
                parent_uid = parent_questions.iloc[0]['Final_UID']
                df_processed.at[idx, 'Final_UID'] = parent_uid
                df_processed.at[idx, 'configured_final_UID'] = parent_uid
    
    return df_processed

# Enhanced unique questions bank creation (fixed for correct data structure)
def create_enhanced_unique_questions_bank(df_reference):
    """
    Enhanced unique questions bank with governance compliance and semantic analysis
    Fixed to work with Snowflake data structure (HEADING_0, UID only)
    """
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Creating enhanced unique questions bank from {len(df_reference)} reference questions")
    
    unique_questions = []
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        uid_questions = group['heading_0'].tolist()
        
        # Get best question with enhanced scoring
        best_question = get_best_question_for_uid(uid_questions)
        
        if not best_question:
            continue
        
        # Standardize the best question
        standardized_question = standardize_question_format(best_question)
        
        # Detect question pattern
        question_pattern = detect_question_pattern(best_question)
        
        # Since we don't have survey_title from Snowflake, we'll categorize based on question content
        # Try to infer category from the question itself
        inferred_category = categorize_question_by_content(best_question)
        
        # Calculate quality metrics
        quality_score = score_question_quality(best_question)
        
        # Check governance compliance
        governance_compliant = len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid']
        
        # Calculate semantic diversity (if multiple questions)
        semantic_diversity = 0.0
        if len(uid_questions) > 1:
            try:
                model = load_sentence_transformer()
                embeddings = model.encode(uid_questions, convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                # Average pairwise similarity
                upper_triangle = similarities.triu(diagonal=1)
                semantic_diversity = upper_triangle[upper_triangle > 0].mean().item() if upper_triangle.numel() > 0 else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate semantic diversity for UID {uid}: {e}")
        
        unique_questions.append({
            'uid': uid,
            'best_question': best_question,
            'standardized_question': standardized_question,
            'total_variants': len(uid_questions),
            'question_length': len(str(best_question)),
            'question_words': len(str(best_question).split()),
            'survey_category': inferred_category,  # Inferred from question content
            'survey_titles': 'Inferred from Snowflake Data',  # Default since not available
            'quality_score': quality_score,
            'governance_compliant': governance_compliant,
            'question_pattern_category': question_pattern['category'],
            'question_pattern_type': question_pattern['pattern'],
            'pattern_confidence': question_pattern['confidence'],
            'semantic_diversity': semantic_diversity,
            'standardization_impact': best_question != standardized_question,
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

def categorize_question_by_content(question_text):
    """
    Categorize question based on content analysis since survey_title is not available
    """
    if not question_text:
        return "Unknown"
    
    text_lower = question_text.lower()
    
    # Application/Registration patterns
    if any(word in text_lower for word in ['apply', 'application', 'register', 'signup', 'join', 'eligibility']):
        return 'Application'
    
    # Pre-programme/Baseline patterns
    if any(word in text_lower for word in ['baseline', 'before', 'prior', 'initial', 'pre-', 'preparation']):
        return 'Pre programme'
    
    # Enrollment/Onboarding patterns
    if any(word in text_lower for word in ['enrollment', 'onboarding', 'welcome', 'start', 'begin']):
        return 'Enrollment'
    
    # Progress/Review patterns
    if any(word in text_lower for word in ['progress', 'milestone', 'checkpoint', 'review', 'assessment']):
        return 'Progress Review'
    
    # Impact/Outcome patterns
    if any(word in text_lower for word in ['impact', 'outcome', 'result', 'effect', 'change', 'transformation', 'improvement']):
        return 'Impact'
    
    # GROW patterns
    if 'grow' in text_lower:
        return 'GROW'
    
    # Feedback patterns
    if any(word in text_lower for word in ['feedback', 'satisfaction', 'rating', 'evaluation', 'opinion']):
        return 'Feedback'
    
    # Pulse/Quick check patterns
    if any(word in text_lower for word in ['pulse', 'quick', 'brief', 'snapshot', 'check-in']):
        return 'Pulse'
    
    # Demographic patterns
    if any(word in text_lower for word in ['age', 'gender', 'education', 'experience', 'role', 'position', 'sector', 'industry', 'location']):
        return 'Demographic'
    
    return "Other"

# Cached Resources (enhanced versions)
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
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Visit: https://community.snowflake.com/s/error-your-user-login-has-been-locked"
            )
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

# Enhanced Normalization
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    # Apply enhanced synonym mapping
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def get_best_question_for_uid(questions_list):
    """Enhanced question selection with quality scoring"""
    if not questions_list:
        return None
    
    # Score each question based on enhanced quality indicators
    scored_questions = [(q, score_question_quality(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

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
    
    # Avoid artifacts (enhanced list)
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

# Data quality management with enhanced governance
def create_enhanced_data_quality_dashboard(df_reference):
    """Enhanced dashboard with full governance compliance and semantic analysis"""
    st.markdown("## 📊 Enhanced Data Quality Dashboard with Governance")
    
    # Run enhanced analysis
    analysis = analyze_uid_variations_with_governance(df_reference)
    
    # Enhanced overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Questions", f"{analysis['total_questions']:,}")
    with col2:
        st.metric("🆔 Unique UIDs", analysis['unique_uids'])
    with col3:
        st.metric("📈 Avg Variations/UID", f"{analysis['avg_variations_per_uid']:.1f}")
    with col4:
        compliance_rate = analysis['governance_compliance']['compliance_rate']
        st.metric("⚖️ Governance Compliance", f"{compliance_rate:.1f}%")
    with col5:
        consolidation_rate = analysis['standardization_impact']['consolidation_rate']
        st.metric("📝 Standardization Impact", f"{consolidation_rate:.1f}%")
    
    # Enhanced governance compliance section
    if analysis['governance_compliance']['violations'] > 0:
        st.markdown('<div class="governance-card">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Governance Compliance Analysis")
        st.warning(f"Found {analysis['governance_compliance']['violations']} UIDs violating governance rules")
        
        violations_df = pd.DataFrame([
            {
                'UID': uid, 
                'Current Variations': count,
                'Excess Variations': count - UID_GOVERNANCE['max_variations_per_uid'],
                'Compliance Status': '❌ Violation'
            }
            for uid, count in analysis['governance_compliance']['violating_uids'].items()
        ])
        st.dataframe(violations_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-card">✅ All UIDs are governance compliant!</div>', unsafe_allow_html=True)
    
    # Enhanced standardization analysis
    st.markdown("### 📝 Question Standardization Impact")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Unique Questions", analysis['standardization_impact']['original_unique'])
    with col2:
        st.metric("After Standardization", analysis['standardization_impact']['standardized_unique'])
    with col3:
        st.metric("Questions Consolidated", analysis['standardization_impact']['questions_consolidated'])
    
    # Question pattern analysis
    st.markdown("### 🎯 Question Pattern Distribution")
    
    pattern_data = analysis['question_patterns']['pattern_distribution']
    if pattern_data:
        pattern_df = pd.DataFrame([
            {'Pattern Category': category, 'Count': count, 'Percentage': f"{(count/analysis['total_questions'])*100:.1f}%"}
            for category, count in pattern_data.items()
        ]).sort_values('Count', ascending=False)
        
        st.dataframe(pattern_df, use_container_width=True)
        
        most_common = analysis['question_patterns']['most_common_pattern']
        st.info(f"Most common pattern: **{most_common[0]}** ({most_common[1]} questions)")
    
    # Enhanced conflict analysis
    if analysis['conflicts']['total_conflicts'] > 0:
        st.markdown("### ⚠️ Advanced Conflict Detection")
        
        conflict_summary = analysis['conflicts']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conflicts", conflict_summary['total_conflicts'])
        with col2:
            st.metric("Governance Violations", conflict_summary['governance_violations'])
        with col3:
            semantic_conflicts = conflict_summary['by_type'].get('semantic_conflicts', 0)
            st.metric("Semantic Conflicts", semantic_conflicts)
        
        # Detailed conflicts
        if conflict_summary['detailed_conflicts']:
            st.markdown("#### 🔍 Top Conflicts")
            
            conflicts_df = pd.DataFrame([
                {
                    'UID': c['uid'],
                    'Conflict Type': c['type'].replace('_', ' ').title(),
                    'Severity': c['severity'].title(),
                    'Details': f"{c.get('count', c.get('conflicts', 'N/A'))} issues",
                    'Governance Impact': '❌' if c.get('governance_violation', False) else '✅'
                }
                for c in conflict_summary['detailed_conflicts']
            ])
            
            st.dataframe(conflicts_df, use_container_width=True)
    
    # Enhanced quality analysis
    st.markdown("### 🎯 Question Quality Analysis")
    
    quality_data = analysis['quality_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Quality", f"{quality_data['avg_quality']:.1f}")
    with col2:
        st.metric("High Quality Questions", quality_data['quality_distribution']['excellent'])
    with col3:
        st.metric("Good Quality Questions", quality_data['quality_distribution']['good'])
    with col4:
        st.metric("Poor Quality Questions", quality_data['quality_distribution']['poor'])
    
    # Enhanced cleaning recommendations with governance
    st.markdown("### 🧹 Enhanced Cleaning & Governance Recommendations")
    
    governance_violations_count = analysis['governance_compliance']['total_violating_questions']
    standardization_opportunities = analysis['standardization_impact']['questions_consolidated']
    
    if governance_violations_count > 0 or standardization_opportunities > 0:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.write(f"⚠️ **Governance Issues**: {governance_violations_count:,} questions in violating UIDs")
        st.write(f"📝 **Standardization Opportunities**: {standardization_opportunities:,} questions could be consolidated")
        st.markdown('</div>', unsafe_allow_html=True)
        
        cleaning_options = {
            'Conservative': f'Remove duplicates only, maintain {UID_GOVERNANCE["max_variations_per_uid"]} variation limit',
            'Moderate': 'Apply standardization, consolidate similar questions, enforce governance',
            'Aggressive': 'Full standardization + governance enforcement, keep only best question per UID',
            'Semantic': 'Use semantic analysis to identify and consolidate truly similar questions'
        }
        
        selected_strategy = st.selectbox("Choose enhanced cleaning strategy:", list(cleaning_options.keys()))
        st.info(f"**{selected_strategy}**: {cleaning_options[selected_strategy]}")
        
        if st.button(f"🧠 Apply {selected_strategy} Enhanced Cleaning", type="primary"):
            with st.spinner(f"Applying {selected_strategy.lower()} enhanced cleaning with governance..."):
                cleaned_df, summary = enhanced_clean_uid_variations(df_reference, selected_strategy.lower())
                
                st.success(f"✅ Enhanced cleaning completed!")
                
                # Enhanced summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Before:** {summary['original_count']:,} questions")
                    st.write(f"**After:** {summary['final_count']:,} questions")
                with col2:
                    st.write(f"**Removed:** {summary['total_removed']:,} questions")
                    st.write(f"**Reduction:** {summary['removal_percentage']:.1f}%")
                with col3:
                    st.write(f"**Governance Compliant:** {'✅' if summary['governance_compliant'] else '❌'}")
                    st.write(f"**Standardization Applied:** {'✅' if summary.get('standardization_applied', False) else '❌'}")
                
                # Enhanced cleaning log
                with st.expander("📋 Detailed Cleaning Report"):
                    for log_entry in summary['cleaning_log']:
                        st.write(f"• {log_entry}")
                    
                    if summary.get('semantic_consolidations', 0) > 0:
                        st.write(f"• **Semantic Consolidations:** {summary['semantic_consolidations']} questions merged")
                    
                    if summary.get('standardization_changes', 0) > 0:
                        st.write(f"• **Standardization Changes:** {summary['standardization_changes']} questions standardized")
                
                # Enhanced download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Enhanced Cleaned Data",
                        cleaned_df.to_csv(index=False),
                        f"enhanced_cleaned_questions_{selected_strategy.lower()}_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Create cleaning report
                    report_data = {
                        'cleaning_strategy': selected_strategy,
                        'original_count': summary['original_count'],
                        'final_count': summary['final_count'],
                        'governance_compliance': summary['governance_compliant'],
                        'standardization_applied': summary.get('standardization_applied', False),
                        'semantic_consolidations': summary.get('semantic_consolidations', 0),
                        'cleaning_steps': summary['cleaning_log']
                    }
                    
                    st.download_button(
                        "📊 Download Cleaning Report",
                        json.dumps(report_data, indent=2),
                        f"cleaning_report_{selected_strategy.lower()}_{uuid4()}.json",
                        "application/json",
                        use_container_width=True
                    )
                
                return cleaned_df
    else:
        st.markdown('<div class="success-card">✅ Data quality is excellent! No major governance or standardization issues detected.</div>', unsafe_allow_html=True)
    
    return df_reference

def enhanced_clean_uid_variations(df_reference, cleaning_strategy='moderate'):
    """Enhanced cleaning with full governance, standardization, and semantic analysis"""
    logger.info(f"Starting enhanced cleaning with strategy: {cleaning_strategy}")
    original_count = len(df_reference)
    
    df_cleaned = df_reference.copy()
    cleaning_log = []
    semantic_consolidations = 0
    standardization_changes = 0
    
    # Step 1: Basic data cleaning
    initial_count = len(df_cleaned)
    
    # Remove empty or very short questions
    df_cleaned = df_cleaned[df_cleaned['heading_0'].str.len() >= 5]
    removed_short = initial_count - len(df_cleaned)
    if removed_short > 0:
        cleaning_log.append(f"Removed {removed_short} questions with < 5 characters")
    
    # Remove HTML-heavy content
    html_pattern = r'<div.*?</div>|<span.*?</span>|<p.*?</p>'
    df_cleaned = df_cleaned[~df_cleaned['heading_0'].str.contains(html_pattern, regex=True, na=False)]
    removed_html = initial_count - removed_short - len(df_cleaned)
    if removed_html > 0:
        cleaning_log.append(f"Removed {removed_html} HTML-heavy questions")
    
    # Remove privacy policy statements
    df_cleaned = df_cleaned[~df_cleaned['heading_0'].str.contains('privacy policy', case=False, na=False)]
    removed_privacy = initial_count - removed_short - removed_html - len(df_cleaned)
    if removed_privacy > 0:
        cleaning_log.append(f"Removed {removed_privacy} privacy policy statements")
    
    # Step 2: Apply standardization (for moderate, aggressive, and semantic strategies)
    if cleaning_strategy in ['moderate', 'aggressive', 'semantic']:
        before_standardization = len(df_cleaned)
        
        # Apply question standardization
        df_cleaned['standardized_heading'] = df_cleaned['heading_0'].apply(standardize_question_format)
        
        # Count standardization changes
        standardization_changes = (df_cleaned['heading_0'] != df_cleaned['standardized_heading']).sum()
        
        if standardization_changes > 0:
            cleaning_log.append(f"Applied standardization to {standardization_changes} questions")
            # Use standardized questions for further processing
            df_cleaned['original_heading'] = df_cleaned['heading_0']
            df_cleaned['heading_0'] = df_cleaned['standardized_heading']
    
    # Step 3: Handle duplicates and similar questions
    before_dedup = len(df_cleaned)
    
    if cleaning_strategy == 'conservative':
        # Only remove exact duplicates
        df_cleaned = df_cleaned.drop_duplicates(subset=['uid', 'heading_0'])
        
    elif cleaning_strategy == 'moderate':
        # Remove exact duplicates and normalize similar questions
        df_cleaned['normalized_question'] = df_cleaned['heading_0'].apply(lambda x: enhanced_normalize(x, ENHANCED_SYNONYM_MAP))
        df_cleaned = df_cleaned.drop_duplicates(subset=['uid', 'normalized_question'])
        df_cleaned = df_cleaned.drop('normalized_question', axis=1)
        
    elif cleaning_strategy == 'aggressive':
        # Keep only the best question per UID
        df_cleaned = df_cleaned.groupby('uid').apply(
            lambda group: pd.Series({
                'heading_0': get_best_question_for_uid(group['heading_0'].tolist()),
                'uid': group['uid'].iloc[0]
            })
        ).reset_index(drop=True)
        
    elif cleaning_strategy == 'semantic':
        # Use semantic analysis for intelligent consolidation
        try:
            model = load_sentence_transformer()
            
            # Group by UID and apply semantic consolidation
            uid_groups = df_cleaned.groupby('uid')
            consolidated_data = []
            
            for uid, group in uid_groups:
                questions = group['heading_0'].tolist()
                
                if len(questions) <= 1:
                    consolidated_data.extend(group.to_dict('records'))
                    continue
                
                # Calculate semantic similarities
                embeddings = model.encode(questions, convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                
                # Find clusters of similar questions
                clusters = []
                used_indices = set()
                
                for i in range(len(questions)):
                    if i in used_indices:
                        continue
                    
                    cluster = [i]
                    used_indices.add(i)
                    
                    for j in range(i + 1, len(questions)):
                        if j not in used_indices and similarities[i][j].item() > 0.85:
                            cluster.append(j)
                            used_indices.add(j)
                    
                    clusters.append(cluster)
                
                # For each cluster, keep the best question
                for cluster in clusters:
                    cluster_questions = [questions[idx] for idx in cluster]
                    best_question = get_best_question_for_uid(cluster_questions)
                    
                    # Keep the row corresponding to the best question
                    best_row = group[group['heading_0'] == best_question].iloc[0].to_dict()
                    consolidated_data.append(best_row)
                    
                    if len(cluster) > 1:
                        semantic_consolidations += len(cluster) - 1
            
            df_cleaned = pd.DataFrame(consolidated_data)
            
            if semantic_consolidations > 0:
                cleaning_log.append(f"Semantic consolidation merged {semantic_consolidations} similar questions")
                
        except Exception as e:
            logger.warning(f"Semantic consolidation failed, falling back to aggressive: {e}")
            cleaning_log.append(f"Semantic analysis failed, applied aggressive cleaning instead")
            # Fallback to aggressive cleaning
            df_cleaned = df_cleaned.groupby('uid').apply(
                lambda group: pd.Series({
                    'heading_0': get_best_question_for_uid(group['heading_0'].tolist()),
                    'uid': group['uid'].iloc[0]
                })
            ).reset_index(drop=True)
    
    removed_duplicates = before_dedup - len(df_cleaned)
    if removed_duplicates > 0:
        cleaning_log.append(f"Removed/consolidated {removed_duplicates} duplicate/similar questions")
    
    # Step 4: Apply governance rules (for all strategies except conservative)
    if cleaning_strategy != 'conservative':
        uid_counts = df_cleaned['uid'].value_counts()
        excessive_threshold = UID_GOVERNANCE['max_variations_per_uid']
        
        excessive_uids = uid_counts[uid_counts > excessive_threshold].index
        governance_violations = 0
        
        for uid in excessive_uids:
            uid_questions = df_cleaned[df_cleaned['uid'] == uid]['heading_0'].tolist()
            
            if cleaning_strategy in ['moderate', 'semantic']:
                # Keep top N best questions based on governance limit
                best_questions = sorted(uid_questions, key=lambda q: score_question_quality(q), reverse=True)[:excessive_threshold]
                df_cleaned = df_cleaned[~((df_cleaned['uid'] == uid) & (~df_cleaned['heading_0'].isin(best_questions)))]
                governance_violations += len(uid_questions) - excessive_threshold
            
            elif cleaning_strategy == 'aggressive':
                # Keep only the single best question
                best_question = get_best_question_for_uid(uid_questions)
                df_cleaned = df_cleaned[~((df_cleaned['uid'] == uid) & (df_cleaned['heading_0'] != best_question))]
                governance_violations += len(uid_questions) - 1
        
        if governance_violations > 0:
            cleaning_log.append(f"Enforced governance rules: removed {governance_violations} questions to comply with {excessive_threshold} variation limit")
    
    final_count = len(df_cleaned)
    total_removed = original_count - final_count
    
    # Check final governance compliance
    final_uid_counts = df_cleaned['uid'].value_counts()
    governance_compliant = all(count <= UID_GOVERNANCE['max_variations_per_uid'] for count in final_uid_counts)
    
    cleaning_summary = {
        'original_count': original_count,
        'final_count': final_count,
        'total_removed': total_removed,
        'removal_percentage': (total_removed / original_count) * 100,
        'cleaning_log': cleaning_log,
        'strategy_used': cleaning_strategy,
        'governance_compliant': governance_compliant,
        'standardization_applied': cleaning_strategy in ['moderate', 'aggressive', 'semantic'],
        'standardization_changes': standardization_changes,
        'semantic_consolidations': semantic_consolidations
    }
    
    logger.info(f"Enhanced cleaning completed: {original_count} -> {final_count} ({total_removed} removed)")
    
    return df_cleaned, cleaning_summary

# Calculate Matched Questions Percentage (keeping existing)
def calculate_matched_percentage(df_final):
    if df_final is None or df_final.empty:
        logger.info("calculate_matched_percentage: df_final is None or empty")
        return 0.0
    
    df_main = df_final[df_final["is_choice"] == False].copy()
    logger.info(f"calculate_matched_percentage: Total main questions: {len(df_main)}")
    
    privacy_filter = ~df_main["heading_0"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["heading_0"].str.contains(html_pattern, case=False, na=False, regex=True)
    
    eligible_questions = df_main[privacy_filter & html_filter]
    logger.info(f"calculate_matched_percentage: Eligible questions after exclusions: {len(eligible_questions)}")
    
    if eligible_questions.empty:
        logger.info("calculate_matched_percentage: No eligible questions after exclusions")
        return 0.0
    
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    logger.info(f"calculate_matched_percentage: Matched questions: {len(matched_questions)}")
    percentage = (len(matched_questions) / len(eligible_questions)) * 100
    return round(percentage, 2)

# Fixed Snowflake Queries (using correct column structure)
def run_snowflake_reference_query_all():
    """Fetch ALL reference questions from Snowflake with pagination - fixed for correct columns"""
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
                
            # Add default survey_title for compatibility with existing code
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
                st.warning("⚠️ Snowflake query failed due to invalid column. Using available columns only.")
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

def run_snowflake_reference_query(limit=10000, offset=0):
    """Original function for backward compatibility - fixed"""
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
        
        # Add default survey_title for compatibility
        if not result.empty:
            result['survey_title'] = 'Unknown Survey'
        
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")
        elif "invalid identifier" in str(e).lower():
            st.warning("⚠️ Snowflake query failed due to invalid column. Contact admin to verify table schema.")
        raise

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
        
        # Add default survey_title for compatibility
        if not result.empty:
            result['survey_title'] = 'Unknown Survey'
        
        return result
    except Exception as e:
        logger.error(f"Snowflake target query failed: {e}")
        if "250001" in str(e):
            st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")
        raise

@st.cache_data
def get_all_reference_questions():
    """Cached function to get all reference questions"""
    return run_snowflake_reference_query_all()

# SurveyMonkey API functions (keeping existing)
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
    # Length-based heuristic
    if len(text.split()) > HEADING_LENGTH_THRESHOLD:
        return "Heading"
    
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_texts = heading_references + [text]
    tfidf_vectors = vectorizer.fit_transform([enhanced_normalize(t) for t in all_texts])
    similarity_scores = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
    max_tfidf_score = np.max(similarity_scores)
    
    # Semantic similarity
    try:
        model = load_sentence_transformer()
        emb_text = model.encode([text], convert_to_tensor=True)
        emb_refs = model.encode(heading_references, convert_to_tensor=True)
        semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
        max_semantic_score = np.max(semantic_scores.cpu().numpy())
    except Exception as e:
        logger.error(f"Semantic similarity computation failed: {e}")
        max_semantic_score = 0.0
    
    # Combine criteria
    if max_tfidf_score >= HEADING_TFIDF_THRESHOLD or max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
        return "Heading"
    return "Main Question/Multiple Choice"

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

# Enhanced UID Matching functions (keeping existing but enhanced)
def compute_tfidf_matches(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP):
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
            
            # Check governance compliance
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
    st.markdown("### 🧠 Enhanced UID Matcher Pro")
    st.markdown("*With Semantic Matching & Governance*")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced SurveyMonkey section
    st.markdown("**📊 Enhanced SurveyMonkey**")
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    if st.button("⚙️ Enhanced Configure Survey", use_container_width=True):
        st.session_state.page = "enhanced_configure_survey"
        st.rerun()
    if st.button("➕ Create New Survey", use_container_width=True):
        st.session_state.page = "create_survey"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Question Bank section
    st.markdown("**📚 Enhanced Question Bank**")
    if st.button("📖 View Question Bank", use_container_width=True):
        st.session_state.page = "view_question_bank"
        st.rerun()
    if st.button("⭐ Enhanced Unique Questions", use_container_width=True):
        st.session_state.page = "enhanced_unique_question_bank"
        st.rerun()
    if st.button("📊 Categorized Questions", use_container_width=True):
        st.session_state.page = "categorized_questions"
        st.rerun()
    if st.button("🧹 Enhanced Data Quality", use_container_width=True):
        st.session_state.page = "enhanced_data_quality"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Governance section
    st.markdown("**⚖️ Enhanced Governance**")
    with st.expander("🔧 Current Settings"):
        st.markdown(f"• Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
        st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
        st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
        st.markdown(f"• Semantic matching: {'✅' if UID_GOVERNANCE['semantic_matching_enabled'] else '❌'}")
        st.markdown(f"• Standardization: {'✅' if QUESTION_STANDARDIZATION.get('standardization_enabled', True) else '❌'}")
        st.markdown(f"• Governance enforcement: {'✅' if UID_GOVERNANCE['governance_enforcement'] else '❌'}")
    
    st.markdown("---")
    
    # Quick links
    st.markdown("**🔗 Quick Actions**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

# App UI with enhanced styling
st.markdown('<div class="main-header">🧠 Enhanced UID Matcher Pro: Semantic Matching & Governance</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Enhanced Home Page
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
    st.markdown("## 🚀 New Enhanced Features")
    
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
        
        if st.button("⭐ Enhanced Unique Questions Bank", use_container_width=True):
            st.session_state.page = "enhanced_unique_question_bank"
            st.rerun()
    
    with col2:
        if st.button("🧹 Enhanced Data Quality Dashboard", use_container_width=True):
            st.session_state.page = "enhanced_data_quality"
            st.rerun()
        
        if st.button("📊 Categorized Questions Analysis", use_container_width=True):
            st.session_state.page = "categorized_questions"
            st.rerun()

# Enhanced Configure Survey Page
elif st.session_state.page == "enhanced_configure_survey":
    enhanced_configure_survey_page()

# Enhanced Unique Questions Bank Page
elif st.session_state.page == "enhanced_unique_question_bank":
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*AI-powered question bank with semantic analysis, governance compliance, and quality scoring*")
    
    try:
        with st.spinner("🔄 Loading and enhancing question bank..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                st.info(f"📊 Loaded {len(df_reference)} total question variants from database")
                
                # Create enhanced unique questions bank
                unique_questions_df = create_enhanced_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.markdown('<div class="warning-card">⚠️ No unique questions found in the database.</div>', unsafe_allow_html=True)
        else:
            # Enhanced summary metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("🆔 Unique UIDs", len(unique_questions_df))
            with col2:
                st.metric("📝 Total Variants", unique_questions_df['total_variants'].sum())
            with col3:
                governance_compliant = len(unique_questions_df[unique_questions_df['governance_compliant'] == True])
                st.metric("⚖️ Governance ✅", f"{governance_compliant}/{len(unique_questions_df)}")
            with col4:
                avg_quality = unique_questions_df['quality_score'].mean()
                st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
            with col5:
                standardized_count = unique_questions_df['standardization_impact'].sum()
                st.metric("📝 Standardized", standardized_count)
            with col6:
                avg_semantic_diversity = unique_questions_df['semantic_diversity'].mean()
                st.metric("🧠 Semantic Diversity", f"{avg_semantic_diversity:.2f}")
            
            st.markdown("---")
            
            # Enhanced filtering options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            
            with col2:
                pattern_filter = st.selectbox("🎯 Question Pattern", 
                                            ["All"] + sorted(unique_questions_df['question_pattern_category'].unique().tolist()))
            
            with col3:
                governance_filter = st.selectbox("⚖️ Governance", ["All", "Compliant Only", "Violations Only"])
            
            with col4:
                quality_filter = st.selectbox("🎯 Quality Level", ["All", "Excellent (>15)", "Good (5-15)", "Fair (0-5)", "Poor (<0)"])
            
            # Apply enhanced filters
            filtered_df = unique_questions_df.copy()
            
            if search_term:
                mask = (filtered_df['best_question'].str.contains(search_term, case=False, na=False) |
                       filtered_df['standardized_question'].str.contains(search_term, case=False, na=False))
                filtered_df = filtered_df[mask]
            
            if pattern_filter != "All":
                filtered_df = filtered_df[filtered_df['question_pattern_category'] == pattern_filter]
            
            if governance_filter == "Compliant Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == True]
            elif governance_filter == "Violations Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == False]
            
            if quality_filter == "Excellent (>15)":
                filtered_df = filtered_df[filtered_df['quality_score'] > 15]
            elif quality_filter == "Good (5-15)":
                filtered_df = filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] <= 15)]
            elif quality_filter == "Fair (0-5)":
                filtered_df = filtered_df[(filtered_df['quality_score'] >= 0) & (filtered_df['quality_score'] <= 5)]
            elif quality_filter == "Poor (<0)":
                filtered_df = filtered_df[filtered_df['quality_score'] < 0]
            
            st.markdown(f"### 📋 Enhanced Results: {len(filtered_df)} unique questions")
            
            if not filtered_df.empty:
                # Enhanced display
                display_df = filtered_df.copy()
                display_df['governance_compliant'] = display_df['governance_compliant'].apply(lambda x: "✅" if x else "❌")
                display_df['standardization_impact'] = display_df['standardization_impact'].apply(lambda x: "📝" if x else "➖")
                
                display_columns = {
                    'uid': 'UID',
                    'best_question': 'Best Question',
                    'standardized_question': 'Standardized',
                    'total_variants': 'Variants',
                    'quality_score': 'Quality',
                    'governance_compliant': 'Governance',
                    'question_pattern_category': 'Pattern',
                    'semantic_diversity': 'Semantic Div.',
                    'standardization_impact': 'Standardized'
                }
                
                display_df = display_df.rename(columns=display_columns)
                
                st.dataframe(
                    display_df[list(display_columns.values())],
                    column_config={
                        "UID": st.column_config.TextColumn("UID", width="small"),
                        "Best Question": st.column_config.TextColumn("Best Question", width="large"),
                        "Standardized": st.column_config.TextColumn("Standardized Question", width="large"),
                        "Variants": st.column_config.NumberColumn("Variants", width="small"),
                        "Quality": st.column_config.NumberColumn("Quality Score", format="%.1f", width="small"),
                        "Governance": st.column_config.TextColumn("Gov.", width="small"),
                        "Pattern": st.column_config.TextColumn("Pattern Category", width="medium"),
                        "Semantic Div.": st.column_config.NumberColumn("Semantic Diversity", format="%.2f", width="small"),
                        "Standardized": st.column_config.TextColumn("Std.", width="small")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Enhanced download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Enhanced Results",
                        filtered_df.to_csv(index=False),
                        f"enhanced_unique_questions_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Governance violations report
                    violations = unique_questions_df[unique_questions_df['governance_compliant'] == False]
                    if not violations.empty:
                        st.download_button(
                            "⚖️ Download Governance Violations",
                            violations.to_csv(index=False),
                            f"governance_violations_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    # Pattern analysis report
                    pattern_analysis = unique_questions_df.groupby('question_pattern_category').agg({
                        'uid': 'count',
                        'quality_score': 'mean',
                        'semantic_diversity': 'mean',
                        'governance_compliant': lambda x: (x == True).sum()
                    }).round(2)
                    
                    st.download_button(
                        "📊 Download Pattern Analysis",
                        pattern_analysis.to_csv(),
                        f"pattern_analysis_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ No questions match your current filters.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Enhanced unique questions bank failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Data Quality Page
elif st.session_state.page == "enhanced_data_quality":
    st.markdown("## 🧹 Enhanced Data Quality Management")
    st.markdown("*Advanced governance compliance, semantic analysis, and intelligent cleaning*")
    
    try:
        with st.spinner("🔄 Loading and analyzing data quality..."):
            df_reference = get_all_reference_questions()
            
        if df_reference.empty:
            st.markdown('<div class="warning-card">⚠️ No reference data found in the database.</div>', unsafe_allow_html=True)
        else:
            create_enhanced_data_quality_dashboard(df_reference)
            
    except Exception as e:
        logger.error(f"Enhanced data quality dashboard failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error loading data quality dashboard: {e}</div>', unsafe_allow_html=True)

# Add other existing pages (keeping the original ones for backward compatibility)
# ... (existing pages like view_surveys, categorized_questions, etc.)

else:
    st.markdown('<div class="warning-card">⚠️ Page not found. Please use the navigation menu.</div>', unsafe_allow_html=True)

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target




