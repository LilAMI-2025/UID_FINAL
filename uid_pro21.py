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

# Custom CSS for better UI (consolidated)
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
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-card { 
        background: #fff3cd; 
        border-left: 4px solid #ffc107; 
    }
    
    .success-card { 
        background: #d4edda; 
        border-left: 4px solid #28a745; 
    }
    
    .info-card { 
        background: #d1ecf1; 
        border-left: 4px solid #17a2b8; 
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (consolidated)
MATCHING_THRESHOLDS = {
    'tfidf_high': 0.60,
    'tfidf_low': 0.50,
    'semantic': 0.60,
    'heading_tfidf': 0.55,
    'heading_semantic': 0.65,
    'heading_length': 50
}

UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True
}

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# Survey Categories (consolidated)
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

# Enhanced Synonym Mapping (consolidated)
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

# Cached Resources (optimized)
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
    """Cached function to get all reference questions"""
    return run_snowflake_reference_query_all()

# Core Functions (consolidated and optimized)
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced normalization with synonym mapping"""
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    
    # Apply synonym mapping
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def categorize_survey(survey_title):
    """Categorize survey based on title keywords"""
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

def score_question_quality(question):
    """Enhanced scoring function for question quality"""
    score = 0
    text = str(question).lower().strip()
    length = len(text)
    
    # Length scoring (sweet spot is 10-100 characters)
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
    
    return score

def get_best_question_for_uid(questions_list):
    """Enhanced question selection with quality scoring"""
    if not questions_list:
        return None
    
    scored_questions = [(q, score_question_quality(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

def enhanced_semantic_matching(question_text, existing_uids_data, threshold=0.85):
    """Enhanced semantic matching with governance rules"""
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

def create_unique_questions_bank(df_reference):
    """Enhanced unique questions bank with survey categorization"""
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
        best_question = get_best_question_for_uid(uid_questions)
        
        # Get survey titles for categorization
        survey_titles = group.get('survey_title', pd.Series()).dropna().unique()
        
        # Determine category from survey titles
        categories = []
        for title in survey_titles:
            category = categorize_survey(title)
            if category not in categories:
                categories.append(category)
        
        primary_category = categories[0] if len(categories) == 1 else "Mixed" if categories else "Unknown"
        
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
                'all_variants': uid_questions
            })
    
    unique_df = pd.DataFrame(unique_questions)
    logger.info(f"Created unique questions bank with {len(unique_df)} UIDs")
    
    # Sort by UID
    if not unique_df.empty:
        try:
            unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
            unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
            unique_df = unique_df.drop('uid_numeric', axis=1)
        except:
            unique_df = unique_df.sort_values('uid')
    
    return unique_df

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
        
        duplicates = len(uid_questions) - len(set(uid_questions))
        unique_questions = list(set(uid_questions))
        lengths = [len(str(q)) for q in uid_questions]
        
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
    
    # Step 2: Handle duplicates based on strategy
    before_dedup = len(df_cleaned)
    
    if cleaning_strategy == 'conservative':
        df_cleaned = df_cleaned.drop_duplicates(subset=['uid', 'heading_0'])
        
    elif cleaning_strategy == 'moderate':
        df_cleaned['normalized_question'] = df_cleaned['heading_0'].apply(lambda x: enhanced_normalize(x, ENHANCED_SYNONYM_MAP))
        df_cleaned = df_cleaned.drop_duplicates(subset=['uid', 'normalized_question'])
        df_cleaned = df_cleaned.drop('normalized_question', axis=1)
        
    elif cleaning_strategy == 'aggressive':
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
                best_questions = sorted(uid_questions, key=lambda q: score_question_quality(q), reverse=True)[:excessive_threshold]
                df_cleaned = df_cleaned[~((df_cleaned['uid'] == uid) & (~df_cleaned['heading_0'].isin(best_questions)))]
                governance_violations += len(uid_questions) - excessive_threshold
            
            elif cleaning_strategy == 'aggressive':
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

def create_data_quality_dashboard(df_reference):
    """Enhanced dashboard with governance compliance"""
    st.markdown("## 📊 Data Quality Dashboard")
    
    # Run analysis
    analysis = analyze_uid_variations(df_reference)
    
    # Overview metrics with governance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Questions", f"{analysis['total_questions']:,}")
    with col2:
        st.metric("🆔 Unique UIDs", analysis['unique_uids'])
    with col3:
        st.metric("📈 Avg Variations/UID", f"{analysis['avg_variations_per_uid']:.1f}")
    with col4:
        governance_compliance = 100 - analysis['governance_compliance']['violation_rate']
        st.metric("⚖️ Governance Compliance", f"{governance_compliance:.1f}%")
    
    # Governance compliance section
    if analysis['governance_compliance']['violations'] > 0:
        st.markdown("### ⚖️ Governance Compliance Issues")
        st.warning(f"Found {analysis['governance_compliance']['violations']} UIDs violating the maximum variations rule ({UID_GOVERNANCE['max_variations_per_uid']} variations per UID)")
        
        violations_df = pd.DataFrame([
            {'UID': uid, 'Violations': count - UID_GOVERNANCE['max_variations_per_uid'], 'Total Variations': count}
            for uid, count in analysis['governance_compliance']['violating_uids'].items()
        ])
        st.dataframe(violations_df, use_container_width=True)
    
    # Problematic UIDs section
    if analysis['problematic_uids']['count'] > 0:
        st.markdown("### ⚠️ UIDs with Excessive Variations")
        
        problematic_df = pd.DataFrame([
            {
                'UID': uid, 
                'Variations': count, 
                'Percentage': f"{(count/analysis['total_questions'])*100:.1f}%",
                'Governance Violation': '❌' if count > UID_GOVERNANCE['max_variations_per_uid'] else '✅'
            }
            for uid, count in analysis['problematic_uids']['uids'].items()
        ])
        
        st.dataframe(problematic_df, use_container_width=True)
        
        # Detailed analysis for top problematic UIDs
        st.markdown("### 🔍 Detailed Analysis of Top Problematic UIDs")
        
        for uid, details in list(analysis['variation_patterns'].items())[:3]:
            violation_icon = "❌" if details['governance_violation'] else "✅"
            with st.expander(f"🆔 UID {uid} - {details['total_variations']} variations {violation_icon}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Patterns Found:**")
                    for pattern, count in details['patterns'].items():
                        if count > 0:
                            st.write(f"• {pattern.replace('_', ' ').title()}: {count}")
                
                with col2:
                    st.markdown("**Statistics:**")
                    st.write(f"• Average length: {details['avg_length']:.0f} characters")
                    st.write(f"• Length range: {details['length_range'][0]} - {details['length_range'][1]}")
                    st.write(f"• Unique variations: {details['patterns']['unique_variations']}")
                    st.write(f"• Governance compliant: {'❌' if details['governance_violation'] else '✅'}")
                
                st.markdown("**Sample Questions:**")
                for i, question in enumerate(details['sample_questions'], 1):
                    st.write(f"{i}. {question[:100]}{'...' if len(question) > 100 else ''}")
    
    # Enhanced cleaning recommendations
    st.markdown("### 🧹 Enhanced Cleaning Recommendations")
    
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
        st.warning(f"⚠️ Found approximately {total_junk:,} junk questions and {governance_violations_count:,} governance violations that could be cleaned up")
        
        cleaning_options = {
            'Conservative': 'Remove only exact duplicates and obvious junk',
            'Moderate': f'Remove duplicates, normalize similar questions, limit variations per UID to {UID_GOVERNANCE["max_variations_per_uid"]}',
            'Aggressive': 'Keep only the single best question per UID (full governance compliance)'
        }
        
        selected_strategy = st.selectbox("Choose cleaning strategy:", list(cleaning_options.keys()))
        st.info(f"**{selected_strategy}**: {cleaning_options[selected_strategy]}")
        
        if st.button(f"🧹 Apply {selected_strategy} Cleaning", type="primary"):
            with st.spinner(f"Applying {selected_strategy.lower()} cleaning..."):
                cleaned_df, summary = clean_uid_variations(df_reference, selected_strategy.lower())
                
                st.success(f"✅ Cleaning completed!")
                st.write(f"**Before:** {summary['original_count']:,} questions")
                st.write(f"**After:** {summary['final_count']:,} questions")
                st.write(f"**Removed:** {summary['total_removed']:,} questions ({summary['removal_percentage']:.1f}%)")
                st.write(f"**Governance Compliant:** {'✅' if summary['governance_compliant'] else '❌'}")
                
                # Show cleaning log
                with st.expander("📋 Cleaning Details"):
                    for log_entry in summary['cleaning_log']:
                        st.write(f"• {log_entry}")
                
                # Offer download
                st.download_button(
                    "📥 Download Cleaned Data",
                    cleaned_df.to_csv(index=False),
                    f"cleaned_question_bank_{selected_strategy.lower()}_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                return cleaned_df
    else:
        st.success("✅ Data quality looks good! No major issues detected.")
    
    return df_reference

# Snowflake Database Functions (optimized)
def run_snowflake_reference_query_all():
    """Fetch ALL reference questions from Snowflake with pagination"""
    all_data = []
    limit = 10000
    offset = 0
    
    query = """
        SELECT HEADING_0, UID, SURVEY_TITLE
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL
        ORDER BY CAST(UID AS INTEGER) ASC
        LIMIT :limit OFFSET :offset
    """
    
    while True:
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
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

def run_snowflake_target_query():
    """Fetch target questions that need UID assignment"""
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
            st.warning("🔒 Cannot fetch Snowflake data: User account is locked.")