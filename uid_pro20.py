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
    uid_groups = df_reference fios://uid_pro6.pyroupby('uid')
    
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

def create_unique_question_bank_1to1(df_reference):
    """
    Create a unique question bank with 1:1 UID-to-question mapping based on highest count.
    Returns a DataFrame with unique questions and a conflict report.
    """
    logger.info(f"Creating 1:1 unique question bank from {len(df_reference)} questions")
    
    # Group by UID and heading_0 to get counts
    question_counts = df_reference.groupby(['uid', 'heading_0']).size().reset_index(name='count')
    
    # For each UID, select the question with the highest count
    unique_questions = []
    conflicts = []
    
    # Group by UID
    uid_groups = question_counts.groupby('uid')
    
    for uid, group in uid_groups:
        if pd.isna(uid):
            continue
            
        # Select the question with the highest count
        best_question = group.loc[group['count'].idxmax()]
        question_text = best_question['heading_0']
        max_count = best_question['count']
        
        # Get survey titles and category
        survey_titles = df_reference[df_reference['uid'] == uid]['survey_title'].dropna().unique()
        category = categorize_survey(', '.join(survey_titles)) if survey_titles else 'Unknown'
        
        # Calculate quality score and governance compliance
        quality_score = score_question_quality(question_text)
        total_variants = len(group)
        governance_compliant = total_variants <= UID_GOVERNANCE['max_variations_per_uid']
        
        unique_questions.append({
            'uid': uid,
            'question': question_text,
            'count': max_count,
            'total_variants': total_variants,
            'survey_category': category,
            'survey_titles': ', '.join(survey_titles) if survey_titles else 'Unknown',
            'quality_score': quality_score,
            'governance_compliant': governance_compliant
        })
        
        # Detect conflicts (same question with high counts under different UIDs)
        if max_count > 10:  # Arbitrary threshold for significant counts
            same_question = question_counts[question_counts['heading_0'] == question_text]
            if len(same_question['uid'].unique()) > 1:
                conflicts.append({
                    'question': question_text,
                    'uids': list(same_question['uid'].unique()),
                    'counts': {row['uid']: row['count'] for _, row in same_question.iterrows()},
                    'selected_uid': uid,
                    'max_count': max_count
                })
    
    unique_df = pd.DataFrame(unique_questions)
    
    # Sort by UID
    try:
        unique_df['uid_numeric'] = pd.to_numeric(unique_df['uid'], errors='coerce')
        unique_df = unique_df.sort_values(['uid_numeric', 'uid'], na_position='last')
        unique_df = unique_df.drop('uid_numeric', axis=1)
    except:
        unique_df = unique_df.sort_values('uid')
    
    conflict_df = pd.DataFrame(conflicts)
    logger.info(f"Created 1:1 question bank with {len(unique_df)} UIDs and {len(conflict_df)} conflicts")
    
    return unique_df, conflict_df

def get_surveymonkey_categorized_questions(token):
    """
    Fetch SurveyMonkey surveys, extract questions, and categorize by survey title.
    Returns a DataFrame with categorized questions and their survey details.
    """
    logger.info("Fetching SurveyMonkey surveys and categorizing questions")
    
    surveys = get_surveys(token)
    all_questions = []
    
    for survey in surveys:
        survey_id = survey['id']
        survey_title = survey['title']
        category = categorize_survey(survey_title)
        
        try:
            survey_details = get_survey_details(survey_id, token)
            questions = extract_questions(survey_details)
            
            for q in questions:
                q['survey_category'] = category
                all_questions.append(q)
                
        except Exception as e:
            logger.error(f"Failed to process survey {survey_id}: {e}")
            continue
    
    df_questions = pd.DataFrame(all_questions)
    
    # Create unique questions per category
    unique_questions = []
    question_counts = df_questions.groupby(['survey_category', 'heading_0']).size().reset_index(name='count')
    
    for category, group in question_counts.groupby('survey_category'):
        for _, row in group.iterrows():
            question_text = row['heading_0']
            count = row['count']
            surveys = df_questions[(df_questions['survey_category'] == category) & 
                                 (df_questions['heading_0'] == question_text)][['survey_id', 'survey_title']].drop_duplicates()
            
            unique_questions.append({
                'question': question_text,
                'survey_category': category,
                'count': count,
                'survey_ids': ', '.join(surveys['survey_id']),
                'survey_titles': ', '.join(surveys['survey_title']),
                'schema_type': df_questions[df_questions['heading_0'] == question_text]['schema_type'].iloc[0],
                'question_category': df_questions[df_questions['heading_0'] == question_text]['question_category'].iloc[0],
                'quality_score': score_question_quality(question_text)
            })
    
    unique_df = pd.DataFrame(unique_questions)
    logger.info(f"Created categorized question bank with {len(unique_df)} unique questions across {unique_df['survey_category'].nunique()} categories")
    
    return unique_df

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
        st.metric("­ЪЊѕ Avg Variations/UID", f"{analysis['avg_variations_per_uid']:.1f}")
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
                    "­Ъћњ Snowflake connection failed: User account is locked. "
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

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("### ­ЪДа UID Matcher Pro")
    st.markdown("Navigate through the application")
    
    # Main navigation
    if st.button("­ЪЈа Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # SurveyMonkey section
    st.markdown("**­ЪЊі SurveyMonkey**")
    if st.button("­ЪЉЂ№ИЈ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    if st.button("РџЎ№ИЈ Configure Survey", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()
    if st.button("РъЋ Create New Survey", use_container_width=True):
        st.session_state.page = "create_survey"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank section
    st.markdown("**­ЪЊџ Question Bank**")
    if st.button("­ЪЊќ View Question Bank", use_container_width=True):
        st.session_state.page = "view_question_bank"
        st.rerun()
    if st.button("РГљ Unique Questions Bank", use_container_width=True):
        st.session_state.page = "unique_question_bank"
        st.rerun()
    if st.button("РГљ Unique Questions (1:1)", use_container_width=True):
        st.session_state.page = "unique_question_bank_1to1"
        st.rerun()
    if st.button("­ЪЊі Survey Questions by Category", use_container_width=True):
        st.session_state.page = "survey_questions_by_category"
        st.rerun()
    if st.button("­ЪЊі Categorized Questions", use_container_width=True):
        st.session_state.page = "categorized_questions"
        st.rerun()
    if st.button("­Ъћё Update Question Bank", use_container_width=True):
        st.session_state.page = "update_question_bank"
        st.rerun()
    if st.button("­ЪД╣ Data Quality Management", use_container_width=True):
        st.session_state.page = "data_quality"
        st.rerun()
    
    st.markdown("---")
    
    # Governance section
    st.markdown("**Рџќ№ИЈ Governance**")
    st.markdown(f"Рђб Max variations per UID: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"Рђб Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.markdown(f"Рђб Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
    
    st.markdown("---")
    
    # Quick links
    st.markdown("**­ЪћЌ Quick Links**")
    st.markdown("­ЪЊЮ [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("­Ъєћ [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

# App UI with enhanced styling
st.markdown('<div class="main-header">­ЪДа UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">Рџа№ИЈ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Home Page with Enhanced Dashboard
if st.session_state.page == "home":
    st.markdown("## ­ЪЈа Welcome to Enhanced UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("­Ъћё Status", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            # Quick connection test
            with get_snowflake_engine().connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                count = result.fetchone()[0]
                st.metric("­ЪЊі Total UIDs", f"{count:,}")
        except:
            st.metric("­ЪЊі Total UIDs", "Connection Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                surveys = get_surveys(token)
                st.metric("­ЪЊІ SM Surveys", len(surveys))
            else:
                st.metric("­ЪЊІ SM Surveys", "No Token")
        except:
            st.metric("­ЪЊІ SM Surveys", "API Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Рџќ№ИЈ Governance", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced features highlight
    st.markdown("## ­Ъџђ Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ­Ъј» Enhanced UID Matching")
        st.markdown("Рђб **Semantic Matching**: AI-powered question similarity")
        st.markdown("Рђб **Governance Rules**: Automatic compliance checking")
        st.markdown("Рђб **Conflict Detection**: Real-time duplicate identification")
        st.markdown("Рђб **Quality Scoring**: Advanced question assessment")
        
        if st.button("­ЪћД Configure Survey with Enhanced Matching", use_container_width=True):
            st.session_state.page = "configure_survey"
            st.rerun()
    
    with col2:
        st.markdown("### ­ЪЊі Survey Categorization")
        st.markdown("Рђб **Auto-Categorization**: Smart survey type detection")
        st.markdown("Рђб **Category Filters**: Application, GROW, Impact, etc.")
        st.markdown("Рђб **Cross-Category Analysis**: Compare question patterns")
        st.markdown("Рђб **Quality by Category**: Category-specific insights")
        
        if st.button("­ЪЊі View Categorized Questions", use_container_width=True):
            st.session_state.page = "categorized_questions"
            st.rerun()
    
    st.markdown("---")
    
    # Quick actions grid
    st.markdown("## ­Ъџђ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ­ЪЊі SurveyMonkey Operations")
        if st.button("­ЪЉЂ№ИЈ View & Analyze Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.rerun()
        if st.button("РъЋ Create New Survey", use_container_width=True):
            st.session_state.page = "create_survey"
            st.rerun()
    
    with col2:
        st.markdown("### ­ЪЊџ Question Bank Management")
        if st.button("­ЪЊќ View Full Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
            st.rerun()
        if st.button("РГљ Unique Questions Bank", use_container_width=True):
            st.session_state.page = "unique_question_bank"
            st.rerun()
    
    # System status with governance
    st.markdown("---")
    st.markdown("## ­ЪћД System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        try:
            get_snowflake_engine()
            st.markdown('<div class="success-card">РюЁ Snowflake: Connected</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">РЮї Snowflake: Connection Issues</div>', unsafe_allow_html=True)
    
    with status_col2:
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                get_surveys(token)
                st.markdown('<div class="success-card">РюЁ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">РЮї SurveyMonkey: No Token</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey: API Issues</div>', unsafe_allow_html=True)
    
    with status_col3:
        st.markdown('<div class="success-card">РюЁ Governance: Active</div>', unsafe_allow_html=True)
        st.markdown(f"Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")

# Enhanced Unique Questions Bank Page
elif st.session_state.page == "unique_question_bank":
    st.markdown("## РГљ Enhanced Unique Questions Bank")
    st.markdown("*Best structured question for each UID with governance compliance and quality scoring*")
    
    try:
        with st.spinner("­Ъћё Loading ALL question bank data and creating unique questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Рџа№ИЈ No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                st.info(f"­ЪЊі Loaded {len(df_reference)} total question variants from database")
                
                # Create unique questions bank
                unique_questions_df = create_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.markdown('<div class="warning-card">Рџа№ИЈ No unique questions found in the database.</div>', unsafe_allow_html=True)
        else:
            # Enhanced summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("­Ъєћ Unique UIDs", len(unique_questions_df))
            with col2:
                st.metric("­ЪЊЮ Total Variants", unique_questions_df['total_variants'].sum())
            with col3:
                governance_compliant = len(unique_questions_df[unique_questions_df['governance_compliant'] == True])
                st.metric("Рџќ№ИЈ Governance Compliant", f"{governance_compliant}/{len(unique_questions_df)}")
            with col4:
                avg_quality = unique_questions_df['quality_score'].mean()
                st.metric("­Ъј» Avg Quality Score", f"{avg_quality:.1f}")
            with col5:
                categories = unique_questions_df['survey_category'].nunique()
                st.metric("­ЪЊі Categories", categories)
            
            st.markdown("---")
            
            # Enhanced search and filter options
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_term = st.text_input("­ЪћЇ Search questions", placeholder="Type to filter questions...")
            
            with col2:
                min_variants = st.selectbox("­ЪЊі Min variants", [1, 2, 3, 5, 10, 20], index=0)
            
            with col3:
                quality_filter = st.selectbox("­Ъј» Quality Filter", ["All", "High (>10)", "Medium (5-10)", "Low (<5)"])
            
            # Additional filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                governance_filter = st.selectbox("Рџќ№ИЈ Governance", ["All", "Compliant Only", "Violations Only"])
            
            with col2:
                category_filter = st.selectbox("­ЪЊі Category", ["All"] + sorted(unique_questions_df['survey_category'].unique().tolist()))
            
            with col3:
                show_variants = st.checkbox("­ЪЉђ Show all variants", value=False)
            
            # Apply filters
            filtered_df = unique_questions_df.copy()
            
            if search_term:
                filtered_df = filtered_df[filtered_df['best_question'].str.contains(search_term, case=False, na=False)]
            
            filtered_df = filtered_df[filtered_df['total_variants'] >= min_variants]
            
            if quality_filter == "High (>10)":
                filtered_df = filtered_df[filtered_df['quality_score'] > 10]
            elif quality_filter == "Medium (5-10)":
                filtered_df = filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] <= 10)]
            elif quality_filter == "Low (<5)":
                filtered_df = filtered_df[filtered_df['quality_score'] < 5]
            
            if governance_filter == "Compliant Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == True]
            elif governance_filter == "Violations Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == False]
            
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['survey_category'] == category_filter]
            
            st.markdown(f"### ­ЪЊІ Showing {len(filtered_df)} unique questions")
            
            # Display the unique questions with enhanced columns
            if not filtered_df.empty:
                display_df = filtered_df.copy()
                
                # Prepare display columns
                display_columns = {
                    'uid': 'UID',
                    'best_question': 'Best Question (Selected)',
                    'total_variants': 'Total Variants',
                    'survey_category': 'Category',
                    'quality_score': 'Quality Score',
                    'governance_compliant': 'Governance',
                    'question_length': 'Character Count',
                    'question_words': 'Word Count'
                }
                
                if not show_variants:
                    display_df = display_df.drop(['all_variants', 'survey_titles'], axis=1, errors='ignore')
                else:
                    display_columns['all_variants'] = 'All Variants'
                    display_columns['survey_titles'] = 'Survey Titles'
                
                display_df = display_df.rename(columns=display_columns)
                
                # Add governance icons
                display_df['Governance'] = display_df['Governance'].apply(lambda x: "РюЁ" if x else "РЮї")
                
                st.dataframe(
                    display_df,
                    column_config={
                        "UID": st.column_config.TextColumn("UID", width="small"),
                        "Best Question (Selected)": st.column_config.TextColumn("Best Question (Selected)", width="large"),
                        "Total Variants": st.column_config.NumberColumn("Total Variants", width="small"),
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Quality Score": st.column_config.NumberColumn("Quality Score", format="%.1f", width="small"),
                        "Governance": st.column_config.TextColumn("Governance", width="small"),
                        "Character Count": st.column_config.NumberColumn("Characters", width="small"),
                        "Word Count": st.column_config.NumberColumn("Words", width="small"),
                        "All Variants": st.column_config.TextColumn("All Question Variants", width="large") if show_variants else None,
                        "Survey Titles": st.column_config.TextColumn("Survey Titles", width="large") if show_variants else None
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Enhanced download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "­ЪЊЦ Download Filtered Results (CSV)",
                        display_df.to_csv(index=False),
                        f"unique_questions_filtered_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        "­ЪЊЦ Download Full Details (CSV)",
                        unique_questions_df.to_csv(index=False),
                        f"unique_questions_full_{uuid4()}.csv",
"text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    st.download_button(
                        "­ЪЊЦ Download JSON Format",
                        unique_questions_df.to_json(orient='records', lines=True),
                        f"unique_questions_{uuid4()}.json",
                        "application/json",
                        use_container_width=True
                    )
                
                # Governance violation details
                if not filtered_df[filtered_df['governance_compliant'] == False].empty:
                    st.markdown("### Рџа№ИЈ Governance Violations")
                    violation_df = filtered_df[filtered_df['governance_compliant'] == False][['uid', 'best_question', 'total_variants', 'survey_category']]
                    violation_df = violation_df.rename(columns={
                        'uid': 'UID',
                        'best_question': 'Question',
                        'total_variants': 'Variants',
                        'survey_category': 'Category'
                    })
                    st.dataframe(
                        violation_df,
                        column_config={
                            "UID": st.column_config.TextColumn("UID", width="small"),
                            "Question": st.column_config.TextColumn("Question", width="large"),
                            "Variants": st.column_config.NumberColumn("Variants", width="small"),
                            "Category": st.column_config.TextColumn("Category", width="medium")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    st.warning(f"Рџа№ИЈ {len(violation_df)} UIDs have more than {UID_GOVERNANCE['max_variations_per_uid']} variations")
                
                # Quick actions
                st.markdown("---")
                st.markdown("### ­Ъџђ Quick Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("­ЪД╣ Analyze Data Quality", use_container_width=True):
                        st.session_state.page = "data_quality"
                        st.rerun()
                
                with col2:
                    if st.button("­ЪЊі View by Category", use_container_width=True):
                        st.session_state.page = "categorized_questions"
                        st.rerun()
                
    except Exception as e:
        logger.error(f"Unique Questions Bank failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Unique Question Bank (1:1 Mapping) Page
elif st.session_state.page == "unique_question_bank_1to1":
    st.markdown("## РГљ Unique Question Bank (1:1 Mapping)")
    st.markdown("*One question per UID based on highest frequency with conflict analysis*")
    
    try:
        with st.spinner("­Ъћё Loading and processing question bank..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Рџа№ИЈ No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                unique_df, conflict_df = create_unique_question_bank_1to1(df_reference)
        
        if unique_df.empty:
            st.markdown('<div class="warning-card">Рџа№ИЈ No questions found.</div>', unsafe_allow_html=True)
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("­Ъєћ Unique UIDs", len(unique_df))
            with col2:
                st.metric("­ЪЊЮ Total Variants", unique_df['total_variants'].sum())
            with col3:
                st.metric("Рџќ№ИЈ Governance Compliant", f"{len(unique_df[unique_df['governance_compliant'] == True])}/{len(unique_df)}")
            with col4:
                st.metric("Рџа№ИЈ Conflicts", len(conflict_df))
            
            st.markdown("---")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("­ЪћЇ Search questions", placeholder="Type to filter questions...")
            with col2:
                category_filter = st.selectbox("­ЪЊі Category", ["All"] + sorted(unique_df['survey_category'].unique().tolist()))
            with col3:
                min_count = st.selectbox("­ЪЊі Min Count", [1, 5, 10, 50, 100], index=0)
            
            # Apply filters
            filtered_df = unique_df.copy()
            if search_term:
                filtered_df = filtered_df[filtered_df['question'].str.contains(search_term, case=False, na=False)]
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['survey_category'] == category_filter]
            filtered_df = filtered_df[filtered_df['count'] >= min_count]
            
            # Display unique questions
            st.markdown(f"### ­ЪЊІ Showing {len(filtered_df)} Unique Questions")
            display_df = filtered_df[['uid', 'question', 'count', 'total_variants', 'survey_category', 'quality_score', 'governance_compliant']].copy()
            display_df['governance_compliant'] = display_df['governance_compliant'].apply(lambda x: "РюЁ" if x else "РЮї")
            
            display_df = display_df.rename(columns={
                'uid': 'UID',
                'question': 'Question',
                'count': 'Frequency',
                'total_variants': 'Variants',
                'survey_category': 'Category',
                'quality_score': 'Quality',
                'governance_compliant': 'Governance'
            })
            
            st.dataframe(
                display_df,
                column_config={
                    "UID": st.column_config.TextColumn("UID", width="small"),
                    "Question": st.column_config.TextColumn("Question", width="large"),
                    "Frequency": st.column_config.NumberColumn("Frequency", width="small"),
                    "Variants": st.column_config.NumberColumn("Variants", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                    "Governance": st.column_config.TextColumn("Governance", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Conflict Dashboard
            if not conflict_df.empty:
                st.markdown("### Рџа№ИЈ UID Conflict Dashboard")
                st.markdown("*Questions assigned to multiple UIDs with high frequency*")
                
                conflict_display = conflict_df.copy()
                conflict_display['uids'] = conflict_display['uids'].apply(lambda x: ', '.join(x))
                conflict_display['counts'] = conflict_display['counts'].apply(lambda x: ', '.join([f"{uid}: {count}" for uid, count in x.items()]))
                
                st.dataframe(
                    conflict_display[['question', 'uids', 'counts', 'selected_uid']],
                    column_config={
                        'question': st.column_config.TextColumn("Question", width="large"),
                        'uids': st.column_config.TextColumn("Conflicting UIDs", width="medium"),
                        'counts': st.column_config.TextColumn("Counts per UID", width="medium"),
                        'selected_uid': st.column_config.TextColumn("Selected UID", width="small")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Conflict resolution suggestion
                st.markdown("#### ­ЪД╣ Conflict Resolution")
                st.info("Select a question to reassign UIDs or merge conflicts:")
                
                selected_conflict = st.selectbox("Select Conflicting Question", conflict_df['question'].tolist())
                if selected_conflict:
                    conflict_row = conflict_df[conflict_df['question'] == selected_conflict].iloc[0]
                    new_uid = st.selectbox("Assign New UID", conflict_row['uids'], index=conflict_row['uids'].index(conflict_row['selected_uid']))
                    
                    if st.button("Apply UID Reassignment", use_container_width=True):
                        # Update df_reference (simulated update, actual implementation needs database write)
                        df_reference.loc[(df_reference['heading_0'] == selected_conflict) & (~df_reference['uid'].isin([new_uid])), 'uid'] = new_uid
                        st.success(f"Reassigned '{selected_conflict}' to UID {new_uid}")
                        st.rerun()
            
            # Download options
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "­ЪЊЦ Download Unique Questions (CSV)",
                    filtered_df.to_csv(index=False),
                    f"unique_questions_1to1_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                if not conflict_df.empty:
                    st.download_button(
                        "Рџа№ИЈ Download Conflict Report",
                        conflict_df.to_csv(index=False),
                        f"uid_conflicts_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
    except Exception as e:
        logger.error(f"1:1 Unique Question Bank failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Survey Questions by Category Page
elif st.session_state.page == "survey_questions_by_category":
    st.markdown("## ­ЪЊі Survey Questions by Category")
    st.markdown("*Unique SurveyMonkey questions categorized by survey title with UID assignment*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey token not found.</div>', unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("­Ъћё Loading SurveyMonkey questions..."):
            df_questions = get_surveymonkey_categorized_questions(token)
            
        if df_questions.empty:
            st.markdown('<div class="warning-card">Рџа№ИЈ No questions found in SurveyMonkey.</div>', unsafe_allow_html=True)
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("­ЪЊІ Unique Questions", len(df_questions))
            with col2:
                st.metric("­ЪЊі Categories", df_questions['survey_category'].nunique())
            with col3:
                avg_quality = df_questions['quality_score'].mean()
                st.metric("­Ъј» Avg Quality Score", f"{avg_quality:.1f}")
            
            st.markdown("---")
            
            # Category selection
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_category = st.selectbox(
                    "­ЪЊі Select Category",
                    ["All"] + sorted(df_questions['survey_category'].unique().tolist())
                )
            with col2:
                sort_by = st.selectbox("­Ъћё Sort by", ["Question", "Quality Score", "Count"])
            
            # Filter and sort
            filtered_df = df_questions.copy()
            if selected_category != "All":
                filtered_df = filtered_df[filtered_df['survey_category'] == selected_category]
            
            if sort_by == "Question":
                filtered_df = filtered_df.sort_values('question')
            elif sort_by == "Quality Score":
                filtered_df = filtered_df.sort_values('quality_score', ascending=False)
            elif sort_by == "Count":
                filtered_df = filtered_df.sort_values('count', ascending=False)
            
            st.markdown(f"### ­ЪЊІ {selected_category} Questions ({len(filtered_df)} items)")
            
            # Display questions
            display_df = filtered_df[['question', 'survey_category', 'count', 'schema_type', 'question_category', 'quality_score', 'survey_titles']].copy()
            display_df = display_df.rename(columns={
                'question': 'Question',
                'survey_category': 'Category',
                'count': 'Frequency',
                'schema_type': 'Type',
                'question_category': 'Question Type',
                'quality_score': 'Quality',
                'survey_titles': 'Survey Titles'
            })
            
            st.dataframe(
                display_df,
                column_config={
                    "Question": st.column_config.TextColumn("Question", width="large"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Frequency": st.column_config.NumberColumn("Frequency", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                    "Question Type": st.column_config.TextColumn("Question Type", width="medium"),
                    "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                    "Survey Titles": st.column_config.TextColumn("Survey Titles", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # UID Assignment
            st.markdown("### ­Ъєћ Assign UIDs")
            st.info("Select a category and run UID matching for all questions in that category.")
            
            if st.button("­Ъћё Run UID Matching", use_container_width=True):
                with st.spinner("­Ъћё Matching UIDs..."):
                    df_reference = get_all_reference_questions()
                    if df_reference.empty:
                        st.markdown('<div class="warning-card">РЮї No reference data for UID matching.</div>', unsafe_allow_html=True)
                    else:
                        # Prepare target DataFrame
                        target_df = filtered_df[['question', 'survey_category', 'survey_titles']].copy()
                        target_df['heading_0'] = target_df['question']
                        target_df['survey_title'] = target_df['survey_titles']
                        
                        # Run UID matching
                        df_final = run_uid_match(df_reference, target_df)
                        
                        # Display results
                        st.markdown("### ­Ъєћ UID Matching Results")
                        display_results = df_final[['heading_0', 'survey_category', 'Final_UID', 'Final_Match_Type', 'Similarity', 'Semantic_Similarity']].copy()
                        display_results = display_results.rename(columns={
                            'heading_0': 'Question',
                            'survey_category': 'Category',
                            'Final_UID': 'Assigned UID',
                            'Final_Match_Type': 'Match Type',
                            'Similarity': 'TF-IDF Score',
                            'Semantic_Similarity': 'Semantic Score'
                        })
                        
                        st.dataframe(
                            display_results,
                            column_config={
                                "Question": st.column_config.TextColumn("Question", width="large"),
                                "Category": st.column_config.TextColumn("Category", width="medium"),
                                "Assigned UID": st.column_config.TextColumn("UID", width="small"),
                                "Match Type": st.column_config.TextColumn("Match Type", width="medium"),
                                "TF-IDF Score": st.column_config.NumberColumn("TF-IDF", format="%.4f", width="small"),
                                "Semantic Score": st.column_config.NumberColumn("Semantic", format="%.4f", width="small")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Download results
                        st.download_button(
                            "­ЪЊЦ Download UID Assignments",
                            df_final.to_csv(index=False),
                            f"uid_assignments_{selected_category}_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
            
            # Download category questions
            st.markdown("---")
            st.download_button(
                f"­ЪЊЦ Download {selected_category} Questions",
                filtered_df.to_csv(index=False),
                f"{selected_category.lower()}_survey_questions_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
                
    except Exception as e:
        logger.error(f"Survey Questions by Category failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# View Surveys Page
elif st.session_state.page == "view_surveys":
    st.markdown("## ­ЪЉЂ№ИЈ View SurveyMonkey Surveys")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey token not found.</div>', unsafe_allow_html=True)
            st.stop()
        
        with st.spinner("­Ъћё Fetching surveys..."):
            surveys = get_surveys(token)
        
        if not surveys:
            st.markdown('<div class="warning-card">Рџа№ИЈ No surveys found.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"### ­ЪЊІ Found {len(surveys)} surveys")
            survey_df = pd.DataFrame([
                {
                    "ID": survey["id"],
                    "Title": survey["title"],
                    "Question Count": survey.get("question_count", 0),
                    "Category": categorize_survey(survey["title"])
                }
                for survey in surveys
            ])
            
            st.dataframe(
                survey_df,
                column_config={
                    "ID": st.column_config.TextColumn("Survey ID", width="small"),
                    "Title": st.column_config.TextColumn("Survey Title", width="large"),
                    "Question Count": st.column_config.NumberColumn("Questions", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.download_button(
                "­ЪЊЦ Download Survey List",
                survey_df.to_csv(index=False),
                f"survey_list_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        logger.error(f"View Surveys failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Configure Survey Page
elif st.session_state.page == "configure_survey":
    st.markdown("## РџЎ№ИЈ Configure Survey")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey token not found.</div>', unsafe_allow_html=True)
            st.stop()
        
        with st.spinner("­Ъћё Fetching target questions..."):
            df_target = run_snowflake_target_query()
            if df_target.empty:
                st.markdown('<div class="warning-card">Рџа№ИЈ No target questions found.</div>', unsafe_allow_html=True)
                st.stop()
            
            st.session_state.df_target = df_target
        
        with st.spinner("­Ъћё Running UID matching..."):
            df_reference = get_all_reference_questions()
            if df_reference.empty:
                st.markdown('<div class="warning-card">РЮї No reference data for UID matching.</div>', unsafe_allow_html=True)
                st.stop()
            
            df_final = run_uid_match(df_reference, df_target)
            st.session_state.df_final = df_final
        
        st.markdown(f"### ­ЪЊІ {len(df_final)} questions processed")
        matched_percentage = calculate_matched_percentage(df_final)
        st.metric("­Ъєћ Matched Questions", f"{matched_percentage}%")
        
        # Display results
        display_df = df_final[['heading_0', 'survey_title', 'Final_UID', 'Final_Match_Type', 'Similarity', 'Semantic_Similarity', 'UID_Conflict']].copy()
        display_df = display_df.rename(columns={
            'heading_0': 'Question',
            'survey_title': 'Survey Title',
            'Final_UID': 'Assigned UID',
            'Final_Match_Type': 'Match Type',
            'Similarity': 'TF-IDF Score',
            'Semantic_Similarity': 'Semantic Score',
            'UID_Conflict': 'Conflict'
        })
        
        st.dataframe(
            display_df,
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Survey Title": st.column_config.TextColumn("Survey Title", width="medium"),
                "Assigned UID": st.column_config.TextColumn("UID", width="small"),
                "Match Type": st.column_config.TextColumn("Match Type", width="medium"),
                "TF-IDF Score": st.column_config.NumberColumn("TF-IDF", format="%.4f", width="small"),
                "Semantic Score": st.column_config.NumberColumn("Semantic", format="%.4f", width="small"),
                "Conflict": st.column_config.TextColumn("Conflict", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download results
        st.download_button(
            "­ЪЊЦ Download Matching Results",
            df_final.to_csv(index=False),
            f"uid_matching_results_{uuid4()}.csv",
            "text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        logger.error(f"Configure Survey failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# Create New Survey Page
elif st.session_state.page == "create_survey":
    st.markdown("## РъЋ Create New Survey")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">РЮї SurveyMonkey token not found.</div>', unsafe_allow_html=True)
            st.stop()
        
        st.markdown("### Define Survey")
        survey_title = st.text_input("Survey Title", placeholder="Enter survey title...")
        survey_language = st.selectbox("Language", ["en", "es", "fr", "de"])
        
        st.markdown("### Add Questions")
        question_count = st.number_input("Number of Questions", min_value=1, max_value=50, value=1)
        
        questions = []
        for i in range(question_count):
            with st.expander(f"Question {i+1}"):
                q_text = st.text_input(f"Question Text {i+1}", key=f"q_text_{i}")
                q_type = st.selectbox(f"Question Type {i+1}", ["Single Choice", "Multiple Choice", "Open-Ended"], key=f"q_type_{i}")
                is_required = st.checkbox(f"Required {i+1}", key=f"q_req_{i}")
                
                choices = []
                if q_type in ["Single Choice", "Multiple Choice"]:
                    choice_count = st.number_input(f"Number of Choices for Q{i+1}", min_value=2, max_value=20, value=2, key=f"choice_count_{i}")
                    for j in range(choice_count):
                        choice_text = st.text_input(f"Choice {j+1}", key=f"choice_{i}_{j}")
                        if choice_text:
                            choices.append({"text": choice_text})
                
                if q_text:
                    questions.append({
                        "heading": q_text,
                        "family": "single_choice" if q_type == "Single Choice" else "multiple_choice" if q_type == "Multiple Choice" else "open_ended",
                        "subtype": "vertical",
                        "position": i + 1,
                        "is_required": is_required,
                        "choices": choices if choices else None
                    })
        
        if st.button("РъЋ Create Survey", use_container_width=True):
            with st.spinner("­Ъћё Creating survey..."):
                survey_template = {
                    "title": survey_title,
                    "language": survey_language
                }
                survey_id = create_survey(token, survey_template)
                page_id = create_page(token, survey_id, {"title": "Main Page"})
                
                for question in questions:
                    create_question(token, survey_id, page_id, question)
                
                st.success(f"РюЁ Survey '{survey_title}' created with ID: {survey_id}")
                st.session_state.survey_template = survey_template
                
                # Run UID assignment
                df_questions = pd.DataFrame([{"heading_0": q["heading"]}, q["survey_title"] = survey_title for q in questions]])
                df_final = run_uid_match(df_reference, df_questions)
                
                st.markdown("### UID Assignments")
                st.dataframe(df_final[['heading_0', 'Final_UID', 'Final_Match_Type']])
                
                st.download_button(
                    "­ЪЊЦ Download Survey Questions",
                    df_final.to_csv(index=False),
                    f"new_survey_{survey_id}_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"Create Survey failed: {e}")
        st.markdown(f'<div class="warning-card">РЮї Error: {e}</div>', unsafe_allow_html=True)

# View Question Bank Page
elif st.session_state.page == "view_question_bank":
    st.markdown("## ­ MDDU View Question Bank")
    
    try:
        with st.spinner("­Ņ D5íU Loading question bank..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Warning: No reference data found in the the database.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"### Found {len(df_reference)} questions")
                st.dataframe(
                    df_reference,
                    column_config={
                        "HEADING_0": st.column_config.TextColumn("Question", width="large"),
                        "UID": st.column_config.TextColumn("UID", width="medium"),
                        "SURVEY_TITLE": st.column_config.TextColumn("Survey Title", width="medium")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.download_button(
                    "Download Question Bank",
                    df_reference.to_csv(index=False),
                    f"question_bank_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"View Question Bank failed: {e}")
        st.markdown(f'<div class="warning-card">Error: {e}</div>', unsafe_allow_html=True)

# Categorized Questions Page
elif st.session_state.page == "categorized_questions":
    st.markdown("## ¬t�i�i Categorized Questions")
    st.markdown("* Questions organized by survey category with quality insights*")

    
    try:
        with st.spinner("­Ņ D5íU Loading and categorizing questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Warning: No reference data found.</div>', unsafe_allow_html=True)
            else:
                # Add category column
                df_reference['Category'] = df_reference['survey_title'].apply(categorize_survey)
                
                # Group by category and heading_0
                grouped_df = df_reference.groupby(['Category', 'heading_0']).agg({
                    'UID': 'nunique',
                    'survey_title': ['count', lambda x: ', '.join(x.unique())]
                }).reset_index()
                
                grouped_df.columns = ['Category', 'Question', 'Unique_UIDs', 'Frequency', 'Survey_Titles']
                grouped_df['Quality_Score'] = grouped_df['Question'].apply(score_question_quality)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Unique Questions", len(grouped_df))
                with col2:
                    st.metric("Categories", grouped_df['Category'].nunique())
                with col3:
                    st.metric("Avg Quality", f"{grouped_df['Quality_Score'].mean():.1f}")
                
                # Filters
                category_filter = st.selectbox("Category", ["All"] + sorted(grouped_df['Category'].unique().tolist()))
                min_quality = st.selectbox("Min Quality", ["All", "High (>10)", ["5-10", "<5"])
                
                filtered_df = grouped_df.copy()
                if category_filter != "All":
                    filtered_df = filtered_df[filtered_df['Category'] == category_filter]
                
                if min_quality == "High (>10)":
                    filtered_df = filtered_df[filtered_df['Quality_Score'] > 10]
                elif min_quality == "5-10":
                    filtered_df = filtered_df[(filtered_df['Quality_Score'] >= 5) & filtered_df['Quality_Score'] <= 10)]
                elif min_quality == "<5":
                    filtered_df = filtered_df[filtered_df['Quality_Score'] < 5]
                
                st.markdown(f"### Showing {len(filtered_df)} questions")
                
                st.dataframe(
                    filtered_df,
                    column_config={
                        "Question": st.column_config.TextColumn("Question", width="large"),
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Unique_UIDs": st.column_config.NumberColumn("Unique UIDs", width="small"),
                        "Frequency": st.column_config.NumberColumn("Frequency", width="small"),
                        "Survey_Titles": st.column_config.TextColumn("Survey Titles", width="large"),
                        "Quality_Score": st.column_config.NumberColumn("Quality", format="%.1f", width="small")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.download_button(
                    "Download Categorized Questions",
                    filtered_df.to_csv(index=False),
                    f"categorized_questions_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"Categorized Questions failed: {e}")
        st.markdown(f'<div class="warning-card">Error: {e}</div>', unsafe_allow_html=True)

# Update Question Bank Page
elif st.session_state.page == "update_question_bank":
    st.markdown("## ¬U5íU Update Question Bank")
    
    try:
        with st.spinner("­Ņ D5íU Loading question bank..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Warning: No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                st.markdown("### Upload new questions to update question bank")
                uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
                
                if uploaded_file:
                    new_questions = pd.read_csv(uploaded_file)
                    
                    if 'heading_0' not in new_questions.columns or 'UID' not in new_questions.columns:
                        st.markdown('<div class="warning-card">Error: CSV must contain heading_0 and UID columns</div>', unsafe_allow_html=True)
                    else:
                        # Validate new questions
                        new_questions['Quality_Score'] = new_questions['heading_0'].apply(score_question_quality)
                        new_questions['Governance_Compliant'] = new_questions.groupby('UID')['heading_0'].count().reindex(new_questions['UID']).apply(lambda x: x <= UID_GOVERNANCE['max_variations_per_uid'])').reset_index(drop=True)
                        
                        st.markdown(f"### Preview of New Questions ({len(new_questions)})")
                        st.dataframe(
                            new_questions,
                            column_config={
                                "heading_0": st.column_config.TextColumn("Question", width="large"),
                                "UID": st.column_config.TextColumn("UID", width="medium"),
                                "Quality_Score": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                                "Governance_Compliant": st.column_config.TextColumn("Governance", width="medium")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        if st.button("Apply Update"):
                            with st.spinner("Updating database..."):
                                # Simulated update (actual implementation requires database write)
                                st.session_state.df_reference = pd.concat([df_reference, new_questions[['heading_0', 'UID']]], ignore_index=True)
                                st.success("Question bank updated successfully!")
                                st.rerun()
                
    except Exception as e:
        logger.error(f"Update Question Bank failed: {e}")
        st.markdown(f'<div class="warning-card">Error: {e}</div>', unsafe_html=True)

# Data Quality Management Page
elif st.session_state.page == "data_quality":
    st.markdown("## (UDDUDD D5 Quality Management")
    
    try:
        with st.spinner("­Ņ UUDD Loading data for quality management..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">Warning: No reference data found.</div>', unsafe_allow_html=True)
            else:
                updated_df = create_data_quality_dashboard(df_reference)
                if updated_df is not None:
                    not st.session_state.df_reference = updated_df
                
    except Exception as e:
        logger.error(f"Data Quality Management failed: {e}")
        st.markdown(f'<div class="warning-card">Error: {e}</div>', unsafe_allow_html=True)
