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
    page_icon="­ЪДа"
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

# UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True
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

# Synonym Mapping (Enhanced)
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

# Enhanced Semantic UID Assignment
def enhanced_semantic_matching(question_text, existing_uids_data, threshold=0.85):
    """
    Enhanced semantic matching with governance rules
    """
    if not existing_uids_data:
        return None, 0.0
    
    try:
        model = load_sentence_transformer()
        
        # Get embeddings
        question_embedding = model.encode([question_text], convert_to_tensor=True)
        existing_questions = [data['best_question'] for data in existing_uids_data.values()]
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(question_embedding, existing_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        if best_score >= threshold:
            best_uid = list(existing_uids_data.keys())[best_idx]
            return best_uid, best_score
            
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
    
    return None, 0.0

# UID Conflict Detection
def detect_uid_conflicts_advanced(df_reference):
    """
    Advanced UID conflict detection with semantic analysis
    """
    conflicts = []
    
    # Group by UID
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        questions = group['heading_0'].unique()
        
        if len(questions) > UID_GOVERNANCE['max_variations_per_uid']:
            conflicts.append({
                'uid': uid,
                'type': 'excessive_variations',
                'count': len(questions),
                'severity': 'high' if len(questions) > 100 else 'medium'
            })
        
        # Check for semantic conflicts (same questions with different UIDs)
        normalized_questions = [enhanced_normalize(q, ENHANCED_SYNONYM_MAP) for q in questions]
        unique_normalized = len(set(normalized_questions))
        
        if len(questions) > unique_normalized * 2:  # Too many duplicates
            conflicts.append({
                'uid': uid,
                'type': 'duplicate_variations',
                'duplicates': len(questions) - unique_normalized,
                'severity': 'medium'
            })
    
    return conflicts

# Enhanced UID Assignment with Governance
def assign_uid_with_governance(question_text, existing_uids_data, survey_category=None):
    """
    Assign UID with governance rules and semantic matching
    """
    # First try semantic matching
    matched_uid, confidence = enhanced_semantic_matching(question_text, existing_uids_data)
    
    if matched_uid and confidence >= UID_GOVERNANCE['semantic_similarity_threshold']:
        # Check if adding to this UID would violate governance
        if existing_uids_data[matched_uid]['variation_count'] < UID_GOVERNANCE['max_variations_per_uid']:
            return {
                'uid': matched_uid,
                'method': 'semantic_match',
                'confidence': confidence,
                'governance_compliant': True
            }
        else:
            logger.warning(f"UID {matched_uid} exceeds max variations, creating new UID")
    
    # If no semantic match or governance violation, suggest new UID
    # Find next available UID
    if existing_uids_data:
        max_uid = max([int(uid) for uid in existing_uids_data.keys() if uid.isdigit()])
        new_uid = str(max_uid + 1)
    else:
        new_uid = "1"
    
    return {
        'uid': new_uid,
        'method': 'new_assignment',
        'confidence': 1.0,
        'governance_compliant': True
    }

# Enhanced data quality management functions
def analyze_uid_variations(df_reference):
    """Enhanced analysis with governance compliance"""
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
    
    # Identify problematic UIDs
    high_variation_threshold = UID_GOVERNANCE['max_variations_per_uid']
    problematic_uids = uid_counts[uid_counts > high_variation_threshold]
    
    analysis_results['problematic_uids'] = {
        'count': len(problematic_uids),
        'uids': problematic_uids.to_dict(),
        'total_questions_in_problematic': problematic_uids.sum()
    }
    
    # Analyze variation patterns for top problematic UIDs
    variation_analysis = {}
    
    for uid in problematic_uids.head(10).index:
        uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
        
        # Check for duplicates
        duplicates = len(uid_questions) - len(set(uid_questions))
        
        # Check for near-duplicates (similarity analysis)
        unique_questions = list(set(uid_questions))
        
        # Analyze question length distribution
        lengths = [len(str(q)) for q in uid_questions]
        
        # Identify common patterns
        patterns = {
            'empty_or_very_short': sum(1 for q in uid_questions if len(str(q).strip()) < 5),
            'html_tags': sum(1 for q in uid_questions if '<' in str(q) and '>' in str(q)),
            'privacy_policy': sum(1 for q in uid_questions if 'privacy policy' in str(q).lower()),
            'placeholder_text': sum(1 for q in uid_questions if any(placeholder in str(q).lower() 
                                    for placeholder in ['please select', 'click here', 'n/a', '...'])),
            'exact_duplicates': duplicates,
            'unique_variations': len(unique_questions)
        }
        
        variation_analysis[uid] = {
            'total_variations': len(uid_questions),
            'patterns': patterns,
            'avg_length': sum(lengths) / len(lengths),
            'length_range': (min(lengths), max(lengths)),
            'sample_questions': unique_questions[:5],
            'governance_violation': len(uid_questions) > UID_GOVERNANCE['max_variations_per_uid']
        }
    
    analysis_results['variation_patterns'] = variation_analysis
    
    return analysis_results

def clean_uid_variations(df_reference, cleaning_strategy='aggressive'):
    """Enhanced cleaning with governance rules"""
    logger.info(f"Starting cleaning with strategy: {cleaning_strategy}")
    original_count = len(df_reference)
    
    df_cleaned = df_reference.copy()
    cleaning_log = []
    
    # Step 1: Remove obvious junk data
    initial_count = len(df_cleaned)
    
    # Remove empty or very short questions
    df_cleaned = df_cleaned[df_cleaned['heading_0'].str.len() >= 5]
    removed_short = initial_count - len(df_cleaned)
    if removed_short > 0:
        cleaning_log.append(f"Removed {removed_short} questions with < 5 characters")
    
    # Remove HTML-heavy content (likely formatting artifacts)
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
    
    # Step 2: Handle duplicates with enhanced normalization
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
        # Remove duplicates and keep only the best question per UID
        df_cleaned = df_cleaned.groupby('uid').apply(
            lambda group: pd.Series({
                'heading_0': get_best_question_for_uid(group['heading_0'].tolist()),
                'uid': group['uid'].iloc[0]
            })
        ).reset_index(drop=True)
    
    removed_duplicates = before_dedup - len(df_cleaned)
    if removed_duplicates > 0:
        cleaning_log.append(f"Removed {removed_duplicates} duplicate/similar questions")
    
    # Step 3: Apply governance rules
    if cleaning_strategy in ['moderate', 'aggressive']:
        uid_counts = df_cleaned['uid'].value_counts()
        excessive_threshold = UID_GOVERNANCE['max_variations_per_uid']
        
        excessive_uids = uid_counts[uid_counts > excessive_threshold].index
        governance_violations = 0
        
        for uid in excessive_uids:
            uid_questions = df_cleaned[df_cleaned['uid'] == uid]['heading_0'].tolist()
            
            if cleaning_strategy == 'moderate':
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
            cleaning_log.append(f"Removed {governance_violations} questions to comply with governance rules")
    
    final_count = len(df_cleaned)
    total_removed = original_count - final_count
    
    cleaning_summary = {
        'original_count': original_count,
        'final_count': final_count,
        'total_removed': total_removed,
        'removal_percentage': (total_removed / original_count) * 100,
        'cleaning_log': cleaning_log,
        'strategy_used': cleaning_strategy,
        'governance_compliant': True
    }
    
    logger.info(f"Cleaning completed: {original_count} -> {final_count} ({total_removed} removed)")
    
    return df_cleaned, cleaning_summary

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

def create_data_quality_dashboard(df_reference):
    """Enhanced dashboard with governance compliance"""
    st.markdown("## ­ЪЊі Data Quality Dashboard")
    
    # Run analysis
    analysis = analyze_uid_variations(df_reference)
    
    # Overview metrics with governance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("­ЪЊі Total Questions", f"{analysis['total_questions']:,}")
    with col2:
        st.metric("­Ъєћ Unique UIDs", analysis['unique_uids'])
    with col3:
        st.metric("­ЪЊЮ Avg Variations/UID", f"{analysis['avg_variations_per_uid']:.1f}")
    with col4:
        governance_compliance = 100 - analysis['governance_compliance']['violation_rate']
        st.metric("Рџќ№ИЈ Governance Compliance", f"{governance_compliance:.1f}%")
    
    # Governance compliance section
    if analysis['governance_compliance']['violations'] > 0:
        st.markdown("### Рџќ№ИЈ Governance Compliance Issues")
        st.warning(f"Found {analysis['governance_compliance']['violations']} UIDs violating the maximum variations rule ({UID_GOVERNANCE['max_variations_per_uid']} variations per UID)")
        
        violations_df = pd.DataFrame([
            {'UID': uid, 'Violations': count - UID_GOVERNANCE['max_variations_per_uid'], 'Total Variations': count}
            for uid, count in analysis['governance_compliance']['violating_uids'].items()
        ])
        st.dataframe(violations_df, use_container_width=True)
    
    # Rest of the existing dashboard code...
    # Problematic UIDs section
    if analysis['problematic_uids']['count'] > 0:
        st.markdown("### Рџа№ИЈ UIDs with Excessive Variations")
        
        problematic_df = pd.DataFrame([
            {
                'UID': uid, 
                'Variations': count, 
                'Percentage': f"{(count/analysis['total_questions'])*100:.1f}%",
                'Governance Violation': 'РЮї' if count > UID_GOVERNANCE['max_variations_per_uid'] else 'РюЁ'
            }
            for uid, count in analysis['problematic_uids']['uids'].items()
        ])
        
        st.dataframe(problematic_df, use_container_width=True)
        
        # Detailed analysis for top problematic UIDs
        st.markdown("### ­ЪћЇ Detailed Analysis of Top Problematic UIDs")
        
        for uid, details in list(analysis['variation_patterns'].items())[:3]:
            violation_icon = "РЮї" if details['governance_violation'] else "РюЁ"
            with st.expander(f"­Ъєћ UID {uid} - {details['total_variations']} variations {violation_icon}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Patterns Found:**")
                    for pattern, count in details['patterns'].items():
                        if count > 0:
                            st.write(f"Рђб {pattern.replace('_', ' ').title()}: {count}")
                
                with col2:
                    st.markdown("**Statistics:**")
                    st.write(f"Рђб Average length: {details['avg_length']:.0f} characters")
                    st.write(f"Рђб Length range: {details['length_range'][0]} - {details['length_range'][1]}")
                    st.write(f"Рђб Unique variations: {details['patterns']['unique_variations']}")
                    st.write(f"Рђб Governance compliant: {'РЮї' if details['governance_violation'] else 'РюЁ'}")
                
                st.markdown("**Sample Questions:**")
                for i, question in enumerate(details['sample_questions'], 1):
                    st.write(f"{i}. {question[:100]}{'...' if len(question) > 100 else ''}")
    
    # Enhanced cleaning recommendations
    st.markdown("### ­ЪД╣ Enhanced Cleaning Recommendations")
    
    total_junk = sum([
        sum(details['patterns']['empty_or_very_short'] for details in analysis['variation_patterns'].values()),
        sum(details['patterns']['html_tags'] for details in analysis['variation_patterns'].values()),
        sum(details['patterns']['privacy_policy'] for details in analysis['variation_patterns'].values()),
        sum(details['patterns']['exact_duplicates'] for details in analysis['variation_patterns'].values())
    ])
    
    governance_violations_count = sum([
        details['total_variations'] - UID_GOVERNANCE['max_variations_per_uid'] 
        for details in analysis['variation_patterns'].values() 
        if details['governance_violation']
    ])
    
    if total_junk > 0 or governance_violations_count > 0:
        st.warning(f"Рџа№ИЈ Found approximately {total_junk:,} junk questions and {governance_violations_count:,} governance violations that could be cleaned up")
        
        cleaning_options = {
            'Conservative': 'Remove only exact duplicates and obvious junk',
            'Moderate': f'Remove duplicates, normalize similar questions, limit variations per UID to {UID_GOVERNANCE["max_variations_per_uid"]}',
            'Aggressive': 'Keep only the single best question per UID (full governance compliance)'
        }
        
        selected_strategy = st.selectbox("Choose cleaning strategy:", list(cleaning_options.keys()))
        st.info(f"**{selected_strategy}**: {cleaning_options[selected_strategy]}")
        
        if st.button(f"­ЪД╣ Apply {selected_strategy} Cleaning", type="primary"):
            with st.spinner(f"Applying {selected_strategy.lower()} cleaning..."):
                cleaned_df, summary = clean_uid_variations(df_reference, selected_strategy.lower())
                
                st.success(f"РюЁ Cleaning completed!")
                st.write(f"**Before:** {summary['original_count']:,} questions")
                st.write(f"**After:** {summary['final_count']:,} questions")
                st.write(f"**Removed:** {summary['total_removed']:,} questions ({summary['removal_percentage']:.1f}%)")
                st.write(f"**Governance Compliant:** {'РюЁ' if summary['governance_compliant'] else 'РЮї'}")
                
                # Show cleaning log
                with st.expander("­ЪЊІ Cleaning Details"):
                    for log_entry in summary['cleaning_log']:
                        st.write(f"Рђб {log_entry}")
                
                # Offer download
                st.download_button(
                    "­ЪЊЦ Download Cleaned Data",
                    cleaned_df.to_csv(index=False),
                    f"cleaned_question_bank_{selected_strategy.lower()}_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                return cleaned_df
    else:
        st.success("РюЁ Data quality looks good! No major issues detected.")
    
    return df_reference

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
                "­Ъћњ Snowflake connection failed: User account is locked. "
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

def create_unique_questions_bank(df_reference):
    """Enhanced unique questions bank with survey categorization"""
    if df_reference.empty:
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_reference)} reference questions for unique bank")
    
    # Group by UID and get the best question for each
    unique_questions = []
    
    uid_groups = df_reference.groupby('uid')
    logger.info(f"Found {len(uid_groups)} unique UIDs")
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        uid_questions = group['heading_0'].tolist()
        best_question = get_best_question_for_uid(uid_questions)
        
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
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split()),
                'survey_category': primary_category,
                'survey_titles': ', '.join(survey_titles) if len(survey_titles) > 0 else 'Unknown',
                'quality_score': score_question_quality(best_question),
                'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid'],
                'all_variants': uid_questions  # Keep all variants for reference
            })
    
    unique_df = pd.DataFrame(unique_questions)
    logger.info(f"Created unique questions bank with {len(unique_df)} UIDs")
    
    # Sort by UID in ascending order
    if not unique_df.empty:
        # Convert UID to numeric if possible, otherwise sort as string
        try:
            unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
            unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
            unique_df = unique_df.drop('uid_numeric', axis=1)
        except:
            unique_df = unique_df.sort_values('uid')
    
    return unique_df

# Calculate Matched Questions Percentage
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

# Enhanced Snowflake Queries
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
                break  # No more data
                
            all_data.append(result)
            offset += limit
            
            # Log progress
            logger.info(f"Fetched {len(result)} rows, total so far: {sum(len(df) for df in all_data)}")
            
            # Break if we got less than the limit (last batch)
            if len(result) < limit:
                break
                
        except Exception as e:
            logger.error(f"Snowflake reference query failed at offset {offset}: {e}")
            if "250001" in str(e):
                st.warning(
                    "­Ъћњ Cannot fetch Snowflake data: User account is locked. "
                    "UID matching is disabled. Please resolve the lockout and retry."
                )
            elif "invalid identifier" in str(e).lower():
                st.warning(
                    "Рџа№ИЈ Snowflake query failed due to invalid column. "
                    "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                    "Contact your Snowflake admin to verify table schema."
                )
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

def run_snowflake_reference_query(limit=10000, offset=0):
    """Original function for backward compatibility"""
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
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning(
                "­Ъћњ Cannot fetch Snowflake data: User account is locked. "
                "UID matching is disabled. Please resolve the lockout and retry."
            )
        elif "invalid identifier" in str(e).lower():
            st.warning(
                "Рџа№ИЈ Snowflake query failed due to invalid column. "
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Contact your Snowflake admin to verify table schema."
            )
        raise

def run_snowflake_target_query():
    query = """
        SELECT DISTINCT HEADING_0, SURVEY_TITLE
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
        if "250001" in str(e):
            st.warning(
                "­Ъћњ Cannot fetch Snowflake data: User account is locked. "
                "Please resolve the lockout and retry."
            )
        raise

@st.cache_data
def get_all_reference_questions():
    """Cached function to get all reference questions"""
    return run_snowflake_reference_query_all()

# SurveyMonkey API functions
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

def create_survey(token, survey_template):
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": survey_template["title"],
            "nickname": survey_template.get("nickname", survey_template["title"]),
            "language": survey_template.get("language", "en")
        })
        response.raise_for_status()
        survey_id = response.json().get("id")
        return survey_id
    except requests.RequestException as e:
        logger.error(f"Failed to create survey: {e}")
        raise

def create_page(token, survey_id, page_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={
            "title": page_template.get("title", ""),
            "description": page_template.get("description", "")
        })
        response.raise_for_status()
        page_id = response.json().get("id")
        return page_id
    except requests.RequestException as e:
        logger.error(f"Failed to create page for survey {survey_id}: {e}")
        raise

def create_question(token, survey_id, page_id, question_template):
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/pages/{page_id}/questions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        payload = {
            "family": question_template["family"],
            "subtype": question_template["subtype"],
            "headings": [{"heading": question_template["heading"]}],
            "position": question_template["position"],
            "required": question_template.get("is_required", False)
        }
        if "choices" in question_template:
            payload["answers"] = {"choices": question_template["choices"]}
        if question_template["family"] == "matrix":
            payload["answers"] = {
                "rows": question_template.get("rows", []),
                "choices": question_template.get("choices", [])
            }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("id")
    except Exception as e:
        logger.error(f"Failed to create question for page {page_id}: {e}")
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

# Enhanced UID Matching functions
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
            conf = "РюЁ High"
        elif best_score >= TFIDF_LOW_CONFIDENCE:
            conf = "Рџа№ИЈ Low"
        else:
            conf = "РЮї No match"
            best_idx = None
        
        if best_idx is not None:
            matched_uid = df_reference.iloc[best_idx]["uid"]
            matched_question = df_reference.iloc[best_idx]["heading_0"]
            
            # Check governance compliance
            uid_count = len(df_reference[df_reference["uid"] == matched_uid])
            governance_compliant = uid_count <= UID_GOVERNANCE['max_variations_per_uid']
            
            matched_uids.append(matched_uid)
            matched_qs.append(matched_question)
            governance_status.append("РюЁ" if governance_compliant else "Рџа№ИЈ")
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

def compute_semantic_matches(df_reference, df_target):
    try:
        model = load_sentence_transformer()
        emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
        emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb_target, emb_ref)

        sem_matches, sem_scores, sem_governance = [], [], []
        for i in range(len(df_target)):
            best_idx = cosine_scores[i].argmax().item()
            score = cosine_scores[i][best_idx].item()
            
            if score >= SEMANTIC_THRESHOLD:
                matched_uid = df_reference.iloc[best_idx]["uid"]
                # Check governance
                uid_count = len(df_reference[df_reference["uid"] == matched_uid])
                governance_compliant = uid_count <= UID_GOVERNANCE['max_variations_per_uid']
                
                sem_matches.append(matched_uid)
                sem_scores.append(round(score, 4))
                sem_governance.append("РюЁ" if governance_compliant else "Рџа№ИЈ")
            else:
                sem_matches.append(None)
                sem_scores.append(None)
                sem_governance.append("N/A")

        df_target["Semantic_UID"] = sem_matches
        df_target["Semantic_Similarity"] = sem_scores
        df_target["Semantic_Governance"] = sem_governance
        return df_target
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        st.error(f"­Ъџе Semantic matching failed: {e}")
        return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "­ЪДа Semantic" if pd.notnull(row["Semantic_UID"]) else "РЮї No match"

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    df_target["Final_Governance"] = df_target["Governance_Status"].combine_first(df_target["Semantic_Governance"])
    
    # Prevent UID assignment for Heading questions
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
    
    if "survey_id" in df_target.columns and "survey_title" in df_target.columns:
        df_target["survey_id_title"] = df_target.apply(
            lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
            axis=1
        )
        
        # Add survey categorization
        df_target["survey_category"] = df_target["survey_title"].apply(categorize_survey)
    
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "Рџа№ИЈ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=ENHANCED_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes provided.")
        st.error("­Ъџе Input data is empty.")
        return pd.DataFrame()

    if len(df_target) > 10000:
        st.warning("Рџа№ИЈ Large dataset detected. Processing may take time.")

    logger.info(f"Processing {len(df_target)} target questions against {len(df_reference)} reference questions.")
    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"­Ъћё Processing batch {start//batch_size + 1}..."):
            batch_target = compute_tfidf_matches(df_reference, batch_target, synonym_map)
            batch_target = compute_semantic_matches(df_reference, batch_target)
            batch_target = finalize_matches(batch_target, df_reference)
            batch_target = detect_uid_conflicts(batch_target)
        df_results.append(batch_target)
    
    if not df_results:
        logger.warning("No results from batch processing.")
        return pd.DataFrame()
    return pd.concat(df_results, ignore_index=True)

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

# Enhanced View Question Bank Page
if st.session_state.page == "view_question_bank":
    st.markdown("## ­ЪЊќ Enhanced View Question Bank")
    st.markdown("*Complete question repository with governance compliance and quality insights*")
    
    try:
        with st.spinner("­Ъћё Fetching ALL Snowflake question bank data..."):
            df_reference = get_all_reference_questions()
        
        if df_reference.empty:
            st.markdown('<div class="warning-card">Рџа№ИЈ No data retrieved from Snowflake.</div>', unsafe_allow_html=True)
        else:
            # Enhanced metrics with governance
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("­ЪЊі Total Questions", len(df_reference))
            
            with col2:
                unique_uids = df_reference['uid'].nunique()
                st.metric("­Ъєћ Unique UIDs", unique_uids)
            
            with col3:
                avg_variants = len(df_reference) / unique_uids if unique_uids > 0 else 0
                st.metric("­ЪЊЮ Avg Variants/UID", f"{avg_variants:.1f}")
            
            with col4:
                # Governance compliance check
                uid_counts = df_reference['uid'].value_counts()
                compliant_uids = len(uid_counts[uid_counts <= UID_GOVERNANCE['max_variations_per_uid']])
                compliance_rate = (compliant_uids / unique_uids) * 100 if unique_uids > 0 else 0
                st.metric("Рџќ№ИЈ Governance Rate", f"{compliance_rate:.1f}%")
            
            with col5:
                # Data completeness
                completeness = 100  # All fetched data is complete
                st.metric("РюЁ Data Loaded", f"{completeness:.0f}%")
            
            # Quick governance alert
            violations = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
            if not violations.empty:
                st.warning(f"Рџќ№ИЈ Governance Alert: {len(violations)} UIDs exceed the maximum variation limit ({UID_GOVERNANCE['max_variations_per_uid']} per UID)")
            
            st.markdown("---")
            
            # Enhanced search and filter
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_query = st.text_input("­ЪћЇ Search questions", placeholder="Type to filter questions...")
            
            with col2:
                uid_filter = st.text_input("­Ъєћ Filter by UID", placeholder="Enter UID...")
            
            with col3:
                sort_by = st.selectbox("­Ъћё Sort by", ["UID (ascending)", "UID (descending)", "Question length", "Survey title"])
            
            # Additional filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                governance_filter = st.selectbox("Рџќ№ИЈ Governance Filter", ["All", "Compliant Only", "Violations Only"])
            
            with col2:
                if 'survey_title' in df_reference.columns:
                    category_filter = st.selectbox("­ЪЊі Survey Category", ["All"] + sorted([categorize_survey(title) for title in df_reference['survey_title'].unique() if pd.notna(title)]))
                else:
                    category_filter = "All"
            
            with col3:
                variation_filter = st.selectbox("­ЪЊЮ Variation Count", ["All", "Single (1)", "Few (2-5)", "Many (6-20)", "Excessive (>20)"])
            
            # Apply filters
            filtered_df = df_reference.copy()
            
            if search_query:
                filtered_df = filtered_df[filtered_df['heading_0'].str.contains(search_query, case=False, na=False)]
            
            if uid_filter:
                filtered_df = filtered_df[filtered_df['uid'].astype(str).str.contains(uid_filter, case=False, na=False)]
            
            # Governance filter
            if governance_filter != "All":
                uid_variation_counts = filtered_df['uid'].value_counts()
                if governance_filter == "Compliant Only":
                    compliant_uids = uid_variation_counts[uid_variation_counts <= UID_GOVERNANCE['max_variations_per_uid']].index
                    filtered_df = filtered_df[filtered_df['uid'].isin(compliant_uids)]
                elif governance_filter == "Violations Only":
                    violating_uids = uid_variation_counts[uid_variation_counts > UID_GOVERNANCE['max_variations_per_uid']].index
                    filtered_df = filtered_df[filtered_df['uid'].isin(violating_uids)]
            
            # Category filter
            if category_filter != "All" and 'survey_title' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['survey_title'].apply(lambda x: categorize_survey(x) == category_filter if pd.notna(x) else False)]
            
            # Variation count filter
            if variation_filter != "All":
                uid_counts_filtered = filtered_df['uid'].value_counts()
                if variation_filter == "Single (1)":
                    target_uids = uid_counts_filtered[uid_counts_filtered == 1].index
                elif variation_filter == "Few (2-5)":
                    target_uids = uid_counts_filtered[(uid_counts_filtered >= 2) & (uid_counts_filtered <= 5)].index
                elif variation_filter == "Many (6-20)":
                    target_uids = uid_counts_filtered[(uid_counts_filtered >= 6) & (uid_counts_filtered <= 20)].index
                elif variation_filter == "Excessive (>20)":
                    target_uids = uid_counts_filtered[uid_counts_filtered > 20].index
                
                filtered_df = filtered_df[filtered_df['uid'].isin(target_uids)]
            
            # Apply sorting
            if sort_by == "UID (ascending)":
                try:
                    filtered_df['uid_numeric'] = pd.to_numeric(filtered_df['uid'], errors='coerce')
                    filtered_df = filtered_df.sort_values(['uid_numeric', 'uid'], na_position='last')
                    filtered_df = filtered_df.drop('uid_numeric', axis=1)
                except:
                    filtered_df = filtered_df.sort_values('uid')
            elif sort_by == "UID (descending)":
                try:
                    filtered_df['uid_numeric'] = pd.to_numeric(filtered_df['uid'], errors='coerce')
                    filtered_df = filtered_df.sort_values(['uid_numeric', 'uid'], ascending=False, na_position='last')
                    filtered_df = filtered_df.drop('uid_numeric', axis=1)
                except:
                    filtered_df = filtered_df.sort_values('uid', ascending=False)
            elif sort_by == "Question length":
                filtered_df['question_length'] = filtered_df['heading_0'].str.len()
                filtered_df = filtered_df.sort_values('question_length', ascending=False)
                filtered_df = filtered_df.drop('question_length', axis=1)
            elif sort_by == "Survey title" and 'survey_title' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('survey_title', na_position='last')
            
            st.markdown(f"### ­ЪЊІ Enhanced Question Bank ({len(filtered_df)} questions showing all variations)")
            
            # Add enhanced grouping information
            if not filtered_df.empty:
                uid_counts_display = filtered_df['uid'].value_counts().sort_index()
                top_uids = uid_counts_display.head(5)
                
                # Enhanced info with governance status
                governance_violations_display = len(uid_counts_display[uid_counts_display > UID_GOVERNANCE['max_variations_per_uid']])
                
                info_text = f"­ЪњА Showing variations across {len(uid_counts_display)} UIDs. "
                info_text += f"UIDs with most variations: {', '.join([f'{uid}({count})' for uid, count in top_uids.items()])}. "
                if governance_violations_display > 0:
                    info_text += f"Рџќ№ИЈ {governance_violations_display} UIDs violate governance rules."
                
                st.info(info_text)
            
            # Enhanced display with additional columns
            display_columns = ['uid', 'heading_0']
            if 'survey_title' in filtered_df.columns:
                display_columns.append('survey_title')
                # Add survey category
                filtered_df['survey_category'] = filtered_df['survey_title'].apply(lambda x: categorize_survey(x) if pd.notna(x) else 'Unknown')
                display_columns.append('survey_category')
            
            st.dataframe(
                filtered_df[display_columns],
                column_config={
                    "uid": st.column_config.TextColumn("UID", width="small"),
                    "heading_0": st.column_config.TextColumn("Question Variation", width="large"),
                    "survey_title": st.column_config.TextColumn("Survey Title", width="medium") if 'survey_title' in display_columns else None,
                    "survey_category": st.column_config.TextColumn("Category", width="small") if 'survey_category' in display_columns else None
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Enhanced download options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "­ЪЊЦ Download Filtered Question Bank",
                    filtered_df.to_csv(index=False),
                    f"enhanced_question_bank_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Generate governance report
                governance_violations_df = df_reference.groupby('uid').size().reset_index(name='variation_count')
                governance_violations_df = governance_violations_df[governance_violations_df['variation_count'] > UID_GOVERNANCE['max_variations_per_uid']]
                
                if not governance_violations_df.empty:
                    st.download_button(
                        "Рџќ№ИЈ Download Governance Violations",
                        governance_violations_df.to_csv(index=False),
                        f"governance_violations_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.success("РюЁ No governance violations")
            
            with col3:
                # Generate quality insights report
                quality_insights = []
                for uid in filtered_df['uid'].unique():
                    uid_questions = filtered_df[filtered_df['uid'] == uid]['heading_0'].tolist()
                    if uid_questions:
                        best_question = get_best_question_for_uid(uid_questions)
                        quality_score = score_question_quality(best_question)
                        
                        quality_insights.append({
                            'uid': uid,
                            'best_question': best_question,
                            'quality_score': quality_score,
                            'total_variations': len(uid_questions),
                            'governance_compliant': len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid']
                        })
                
                if quality_insights:
                    quality_df = pd.DataFrame(quality_insights)
                    st.download_button(
                        "­Ъј» Download Quality Analysis",
                        quality_df.to_csv(index=False),
                        f"quality_analysis_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
    except Exception as e:
        logger.error(f"Enhanced Snowflake processing failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">­Ъћњ Snowflake connection failed: User account is locked. Contact your Snowflake admin or wait 15РђЊ30 minutes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Enhanced View Surveys Page
elif st.session_state.page == "view_surveys":
    st.markdown("## ­ЪЉЂ№ИЈ Enhanced View Surveys")
    st.markdown("*Browse and analyze your SurveyMonkey surveys with categorization*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey token is missing in secrets configuration.</div>', unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("­Ъћё Fetching surveys from SurveyMonkey..."):
            surveys = get_surveys(token)
            
        if not surveys:
            st.markdown('<div class="warning-card">Рџа№ИЈ No surveys found or invalid API response.</div>', unsafe_allow_html=True)
        else:
            # Enhanced survey metrics with categorization
            survey_categories = [categorize_survey(s.get('title', '')) for s in surveys]
            category_counts = pd.Series(survey_categories).value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("­ЪЊі Total Surveys", len(surveys))
            with col2:
                recent_surveys = [s for s in surveys if s.get('date_created', '').startswith('2024') or s.get('date_created', '').startswith('2025')]
                st.metric("­ЪєЋ Recent (2024-2025)", len(recent_surveys))
            with col3:
                st.metric("­ЪЊі Categories", len(category_counts))
            with col4:
                most_common_category = category_counts.index[0] if not category_counts.empty else "None"
                st.metric("­ЪЈє Top Category", most_common_category)
            
            # Category breakdown
            if not category_counts.empty:
                st.markdown("### ­ЪЊі Survey Categories")
                category_cols = st.columns(min(4, len(category_counts)))
                for i, (category, count) in enumerate(category_counts.head(4).items()):
                    with category_cols[i]:
                        st.metric(f"­ЪЊІ {category}", count)
            
            st.markdown("---")
            
            # Process selected surveys
            choices = {s["title"]: s["id"] for s in surveys}
            selected_survey = st.selectbox("­Ъј» Choose Survey", [""] + list(choices.keys()), index=0)
            selected_survey_ids = st.multiselect(
                "­ЪЊІ SurveyID/Title",
                [f"{s['id']} - {s['title']}" for s in surveys],
                default=[],
                help="Select one or more surveys by ID and title"
            )
            category_filter = st.selectbox("­ЪЊі Filter by Category", ["All"] + sorted(category_counts.index.tolist()))
            
            selected_survey_ids_from_title = []
            if selected_survey:
                selected_survey_ids_from_title.append(choices[selected_survey])
            
            all_selected_survey_ids = list(set(selected_survey_ids_from_title + [
                s.split(" - ")[0] for s in selected_survey_ids
            ]))
            
            # Apply category filter
            if category_filter != "All":
                filtered_surveys = [s for s in surveys if categorize_survey(s.get('title', '')) == category_filter]
                if not all_selected_survey_ids:  # If no specific surveys selected, show all in category
                    all_selected_survey_ids = [s['id'] for s in filtered_surveys]
            
            if all_selected_survey_ids:
                combined_questions = []
                progress_bar = st.progress(0)
                
                for i, survey_id in enumerate(all_selected_survey_ids):
                    with st.spinner(f"­Ъћё Fetching survey questions for ID {survey_id}..."):
                        survey_json = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                    progress_bar.progress((i + 1) / len(all_selected_survey_ids))
                
                st.session_state.df_target = pd.DataFrame(combined_questions)
                
                if st.session_state.df_target.empty:
                    st.markdown('<div class="warning-card">Рџа№ИЈ No questions found in the selected survey(s).</div>', unsafe_allow_html=True)
                else:
                    # Enhanced analysis metrics
                    st.markdown("### ­ЪЊі Enhanced Survey Analysis")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        total_questions = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                        st.metric("РЮЊ Questions", total_questions)
                    with col2:
                        total_choices = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == True])
                        st.metric("­ЪЊЮ Choices", total_choices)
                    with col3:
                        headings_count = len(st.session_state.df_target[st.session_state.df_target["question_category"] == "Heading"])
                        st.metric("­ЪЊІ Headings", headings_count)
                    with col4:
                        unique_surveys = st.session_state.df_target["survey_id"].nunique()
                        st.metric("­ЪЊі Surveys", unique_surveys)
                    with col5:
                        # Add survey categories
                        st.session_state.df_target["survey_category"] = st.session_state.df_target["survey_title"].apply(categorize_survey)
                        unique_categories = st.session_state.df_target["survey_category"].nunique()
                        st.metric("­ЪЈи№ИЈ Categories", unique_categories)
                    
                    st.markdown("---")
                    
                    # Enhanced display options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_main_only = st.checkbox("­ЪЊІ Show only main questions", value=False)
                    with col2:
                        question_filter = st.selectbox("­ЪћЇ Filter by Q category", 
                                                     ["All", "Main Question/Multiple Choice", "Heading"])
                    with col3:
                        survey_category_filter = st.selectbox("­ЪЊі Filter by S category",
                                                            ["All"] + sorted(st.session_state.df_target["survey_category"].unique().tolist()))
                    
                    # Filter and display data
                    display_df = st.session_state.df_target.copy()
                    
                    if show_main_only:
                        display_df = display_df[display_df["is_choice"] == False]
                    
                    if question_filter != "All":
                        display_df = display_df[display_df["question_category"] == question_filter]
                    
                    if survey_category_filter != "All":
                        display_df = display_df[display_df["survey_category"] == survey_category_filter]
                    
                    display_df["survey_id_title"] = display_df.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                    
                    st.markdown(f"### ­ЪЊІ Enhanced Survey Questions ({len(display_df)} items)")
                    
                    st.dataframe(
                        display_df[["survey_id_title", "heading_0", "position", "is_choice", "parent_question", "schema_type", "question_category", "survey_category"]],
                        column_config={
                            "survey_id_title": st.column_config.TextColumn("Survey ID/Title", width="medium"),
                            "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                            "position": st.column_config.NumberColumn("Position", width="small"),
                            "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                            "parent_question": st.column_config.TextColumn("Parent Question", width="medium"),
                            "schema_type": st.column_config.TextColumn("Schema Type", width="small"),
                            "question_category": st.column_config.TextColumn("Q Category", width="small"),
                            "survey_category": st.column_config.TextColumn("S Category", width="small")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Enhanced download option
                    st.download_button(
                        "­ЪЊЦ Download Enhanced Survey Data",
                        display_df.to_csv(index=False),
                        f"enhanced_survey_data_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">Рё╣№ИЈ Select a survey or category to view questions and analysis.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Enhanced SurveyMonkey processing failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Update Question Bank Page
elif st.session_state.page == "update_question_bank":
    st.markdown("## ­Ъћё Enhanced Update Question Bank")
    st.markdown("*Match new questions with existing UIDs using enhanced algorithms and governance rules*")
    
    try:
        with st.spinner("­Ъћё Fetching Snowflake data..."):
            df_reference = get_all_reference_questions()
            df_target = run_snowflake_target_query()
        
        if df_reference.empty or df_target.empty:
            st.markdown('<div class="warning-card">Рџа№ИЈ No data retrieved from Snowflake for matching.</div>', unsafe_allow_html=True)
        else:
            # Enhanced initial metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("­ЪЊі Total Questions", len(df_reference))
            with col2:
                unique_uids = df_reference['uid'].nunique()
                st.metric("­Ъєћ Unique UIDs", unique_uids)
            with col3:
                avg_variants = len(df_reference) / unique_uids if unique_uids > 0 else 0
                st.metric("­ЪЊЮ Avg Variants/UID", f"{avg_variants:.1f}")
            with col4:
                unmatched_questions = len(df_target)
                st.metric("­ЪЊІ Unmatched