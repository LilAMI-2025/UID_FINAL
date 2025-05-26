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
    
    .governance-violation {
        background: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #dc3545;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.75  # Increased for better semantic matching
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# Enhanced UID Governance Rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 25,  # Reduced for better governance
    'semantic_similarity_threshold': 0.80,  # Higher threshold for auto-assignment
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'monthly_quality_check': True,
    'standardized_format_required': True,
    'semantic_matching_enabled': True
}

# Enhanced Survey Categories with more keywords
SURVEY_CATEGORIES = {
    'Application': ['application', 'apply', 'registration', 'signup', 'join', 'register', 'enroll form'],
    'Pre programme': ['pre-programme', 'pre programme', 'preparation', 'readiness', 'baseline', 'pre-program', 'pre program', 'before'],
    'Enrollment': ['enrollment', 'enrolment', 'onboarding', 'welcome', 'start', 'begin', 'commence'],
    'Progress Review': ['progress', 'review', 'milestone', 'checkpoint', 'assessment', 'evaluation', 'mid-term', 'interim'],
    'Impact': ['impact', 'outcome', 'result', 'effect', 'change', 'transformation', 'post', 'after', 'completion'],
    'GROW': ['GROW'],  # Exact match for CAPS - this takes priority
    'Feedback': ['feedback', 'evaluation', 'rating', 'satisfaction', 'opinion', 'comment', 'review'],
    'Pulse': ['pulse', 'quick', 'brief', 'snapshot', 'check-in', 'temperature', 'quick check']
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
    "could you please": "what is",
    "would you please": "what is",
    "kindly select": "what is",
    "choose one": "select",
    "pick one": "select"
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

# Enhanced Survey Categorization with Priority
def categorize_survey(survey_title):
    """
    Enhanced categorization with priority ordering and better keyword matching
    """
    if not survey_title or pd.isna(survey_title):
        return "Unknown"
    
    title_lower = survey_title.lower().strip()
    
    # Priority 1: Check GROW first (exact match for CAPS)
    if 'GROW' in survey_title and survey_title.count('GROW') > 0:
        return 'GROW'
    
    # Priority 2: Check for specific patterns
    category_scores = {}
    
    for category, keywords in SURVEY_CATEGORIES.items():
        if category == 'GROW':  # Already checked
            continue
            
        score = 0
        for keyword in keywords:
            # Exact word match gets higher score
            if f" {keyword.lower()} " in f" {title_lower} ":
                score += 10
            # Partial match gets lower score
            elif keyword.lower() in title_lower:
                score += 5
        
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score
    if category_scores:
        return max(category_scores, key=category_scores.get)
    
    return "Other"

# Enhanced Semantic UID Assignment with Governance
def enhanced_semantic_matching(question_text, existing_uids_data, threshold=None):
    """
    Enhanced semantic matching with governance rules and conflict detection
    """
    if not existing_uids_data:
        return None, 0.0, "new_uid_needed"
    
    if threshold is None:
        threshold = UID_GOVERNANCE['semantic_similarity_threshold']
    
    try:
        model = load_sentence_transformer()
        
        # Normalize and clean the question
        cleaned_question = standardize_question_format(question_text)
        
        # Get embeddings
        question_embedding = model.encode([cleaned_question], convert_to_tensor=True)
        existing_questions = [data['best_question'] for data in existing_uids_data.values()]
        existing_embeddings = model.encode(existing_questions, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(question_embedding, existing_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        logger.info(f"Semantic matching: best score {best_score:.3f} (threshold: {threshold:.3f})")
        
        if best_score >= threshold:
            best_uid = list(existing_uids_data.keys())[best_idx]
            
            # Check governance compliance before assignment
            current_variations = existing_uids_data[best_uid].get('variation_count', 0)
            if current_variations >= UID_GOVERNANCE['max_variations_per_uid']:
                logger.warning(f"UID {best_uid} exceeds max variations ({current_variations}/{UID_GOVERNANCE['max_variations_per_uid']})")
                return best_uid, best_score, "governance_violation"
            
            return best_uid, best_score, "semantic_match"
            
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
    
    return None, 0.0, "no_match"

# Standardized Question Format
def standardize_question_format(question_text):
    """
    Standardize question format before UID assignment
    """
    if not question_text or pd.isna(question_text):
        return ""
    
    text = str(question_text).strip()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep question marks
    text = re.sub(r'[^\w\s\?\.\,\-\(\)]', '', text)
    
    # Ensure proper capitalization
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Ensure question ends with proper punctuation
    if text and not text.endswith(('?', '.', ':')):
        if any(qword in text.lower().split()[:3] for qword in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']):
            text += '?'
        else:
            text += '.'
    
    return text

# Enhanced UID Assignment with Full Governance
def assign_uid_with_enhanced_governance(question_text, existing_uids_data, survey_category=None, force_new=False):
    """
    Enhanced UID assignment with semantic matching and full governance compliance
    """
    standardized_question = standardize_question_format(question_text)
    
    if not force_new and UID_GOVERNANCE['semantic_matching_enabled']:
        # Try semantic matching first
        matched_uid, confidence, status = enhanced_semantic_matching(standardized_question, existing_uids_data)
        
        if status == "semantic_match":
            # Update variation count
            existing_uids_data[matched_uid]['variation_count'] = existing_uids_data[matched_uid].get('variation_count', 0) + 1
            
            return {
                'uid': matched_uid,
                'method': 'semantic_match',
                'confidence': confidence,
                'governance_compliant': True,
                'status': 'assigned',
                'standardized_question': standardized_question
            }
        
        elif status == "governance_violation":
            logger.warning(f"Governance violation for UID {matched_uid}, creating new UID")
            # Continue to create new UID
    
    # Create new UID
    if existing_uids_data:
        # Find next available UID
        existing_numeric_uids = [int(uid) for uid in existing_uids_data.keys() if uid.isdigit()]
        if existing_numeric_uids:
            new_uid = str(max(existing_numeric_uids) + 1)
        else:
            new_uid = "1"
    else:
        new_uid = "1"
    
    # Initialize new UID data
    existing_uids_data[new_uid] = {
        'best_question': standardized_question,
        'variation_count': 1,
        'survey_category': survey_category or 'Unknown',
        'created_date': datetime.now().isoformat(),
        'quality_score': score_question_quality(standardized_question)
    }
    
    return {
        'uid': new_uid,
        'method': 'new_assignment',
        'confidence': 1.0,
        'governance_compliant': True,
        'status': 'new_uid_created',
        'standardized_question': standardized_question
    }

# Monthly Quality Check Function
def run_monthly_quality_check(df_reference):
    """
    Run comprehensive monthly quality check with governance compliance
    """
    logger.info("Starting monthly quality check...")
    
    quality_report = {
        'check_date': datetime.now().isoformat(),
        'total_questions': len(df_reference),
        'unique_uids': df_reference['uid'].nunique(),
        'governance_violations': [],
        'quality_issues': [],
        'recommendations': []
    }
    
    # Check governance violations
    uid_counts = df_reference['uid'].value_counts()
    violations = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
    
    for uid, count in violations.items():
        quality_report['governance_violations'].append({
            'uid': uid,
            'variation_count': count,
            'excess': count - UID_GOVERNANCE['max_variations_per_uid'],
            'severity': 'high' if count > UID_GOVERNANCE['max_variations_per_uid'] * 2 else 'medium'
        })
    
    # Check quality issues
    for uid in df_reference['uid'].unique():
        uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
        
        # Check for low quality questions
        low_quality = [q for q in uid_questions if score_question_quality(q) < UID_GOVERNANCE['quality_score_threshold']]
        if low_quality:
            quality_report['quality_issues'].append({
                'uid': uid,
                'low_quality_count': len(low_quality),
                'total_variations': len(uid_questions),
                'sample_low_quality': low_quality[:3]
            })
    
    # Generate recommendations
    if quality_report['governance_violations']:
        quality_report['recommendations'].append("Implement UID consolidation for governance violations")
    
    if quality_report['quality_issues']:
        quality_report['recommendations'].append("Review and improve low-quality question variations")
    
    quality_report['recommendations'].append("Run semantic deduplication to reduce variations")
    
    return quality_report

# Enhanced Conflict Detection
def detect_advanced_uid_conflicts(df_reference):
    """
    Advanced conflict detection with semantic analysis and governance checks
    """
    conflicts = []
    
    # Group by UID
    uid_groups = df_reference.groupby('uid')
    
    for uid, group in uid_groups:
        questions = group['heading_0'].unique()
        
        # Governance violation check
        if len(questions) > UID_GOVERNANCE['max_variations_per_uid']:
            conflicts.append({
                'uid': uid,
                'type': 'governance_violation',
                'count': len(questions),
                'max_allowed': UID_GOVERNANCE['max_variations_per_uid'],
                'severity': 'high' if len(questions) > UID_GOVERNANCE['max_variations_per_uid'] * 2 else 'medium',
                'auto_fix_available': True
            })
        
        # Semantic conflict detection
        if len(questions) > 1 and UID_GOVERNANCE['semantic_matching_enabled']:
            try:
                model = load_sentence_transformer()
                embeddings = model.encode(list(questions), convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                
                # Find questions that are too different (potential conflicts)
                avg_similarity = similarities.mean().item()
                if avg_similarity < 0.7:  # Low average similarity indicates potential conflicts
                    conflicts.append({
                        'uid': uid,
                        'type': 'semantic_conflict',
                        'avg_similarity': avg_similarity,
                        'question_count': len(questions),
                        'severity': 'medium',
                        'sample_questions': list(questions)[:3]
                    })
                    
            except Exception as e:
                logger.error(f"Semantic conflict detection failed for UID {uid}: {e}")
        
        # Quality inconsistency check
        quality_scores = [score_question_quality(q) for q in questions]
        quality_std = np.std(quality_scores)
        
        if quality_std > 5.0:  # High variation in quality scores
            conflicts.append({
                'uid': uid,
                'type': 'quality_inconsistency',
                'quality_std': quality_std,
                'quality_range': (min(quality_scores), max(quality_scores)),
                'severity': 'low',
                'recommendations': ['Review question variations for quality consistency']
            })
    
    # Cross-UID semantic conflicts (questions that should have the same UID)
    if UID_GOVERNANCE['semantic_matching_enabled']:
        try:
            model = load_sentence_transformer()
            unique_questions_df = create_unique_questions_bank(df_reference)
            
            if len(unique_questions_df) > 1:
                best_questions = unique_questions_df['best_question'].tolist()
                uids = unique_questions_df['uid'].tolist()
                
                embeddings = model.encode(best_questions, convert_to_tensor=True)
                similarities = util.cos_sim(embeddings, embeddings)
                
                # Find pairs with high similarity but different UIDs
                for i in range(len(uids)):
                    for j in range(i + 1, len(uids)):
                        similarity = similarities[i][j].item()
                        if similarity > UID_GOVERNANCE['auto_consolidate_threshold']:
                            conflicts.append({
                                'type': 'cross_uid_semantic_match',
                                'uid1': uids[i],
                                'uid2': uids[j],
                                'similarity': similarity,
                                'question1': best_questions[i],
                                'question2': best_questions[j],
                                'severity': 'high',
                                'auto_consolidate_recommended': True
                            })
                            
        except Exception as e:
            logger.error(f"Cross-UID conflict detection failed: {e}")
    
    return conflicts

# Enhanced unique questions bank with categories
def create_enhanced_unique_questions_bank(df_reference):
    """
    Enhanced unique questions bank with survey categorization and governance compliance
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
        best_question = get_best_question_for_uid(uid_questions)
        
        # Enhanced survey categorization
        survey_titles = group.get('survey_title', pd.Series()).dropna().unique()
        
        # Determine primary category with confidence scoring
        category_votes = {}
        for title in survey_titles:
            category = categorize_survey(title)
            category_votes[category] = category_votes.get(category, 0) + 1
        
        if category_votes:
            primary_category = max(category_votes, key=category_votes.get)
            category_confidence = category_votes[primary_category] / len(survey_titles)
        else:
            primary_category = "Unknown"
            category_confidence = 0.0
        
        # Calculate governance compliance
        governance_compliant = len(uid_questions) <= UID_GOVERNANCE['max_variations_per_uid']
        
        # Calculate quality metrics
        quality_scores = [score_question_quality(q) for q in uid_questions]
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        # Detect potential issues
        issues = []
        if not governance_compliant:
            issues.append("governance_violation")
        if avg_quality < UID_GOVERNANCE['quality_score_threshold']:
            issues.append("low_quality")
        if quality_std > 5.0:
            issues.append("quality_inconsistency")
        
        if best_question:
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'standardized_question': standardize_question_format(best_question),
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split()),
                'survey_category': primary_category,
                'category_confidence': round(category_confidence, 2),
                'survey_titles': ', '.join(survey_titles) if len(survey_titles) > 0 else 'Unknown',
                'quality_score': round(avg_quality, 1),
                'quality_std': round(quality_std, 1),
                'governance_compliant': governance_compliant,
                'issues': ', '.join(issues) if issues else 'none',
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
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

# Enhanced Configure Survey Page
def enhanced_configure_survey_page():
    """
    Enhanced configure survey page with semantic matching and governance
    """
    st.markdown("## ⚙️ Enhanced Configure Survey with Semantic Matching")
    st.markdown("*Advanced UID assignment with semantic matching, governance rules, and standardized formatting*")
    
    # Display governance settings
    with st.expander("⚖️ Governance & Matching Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**UID Governance Rules:**")
            st.write(f"• Max variations per UID: {UID_GOVERNANCE['max_variations_per_uid']}")
            st.write(f"• Semantic similarity threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
            st.write(f"• Quality score threshold: {UID_GOVERNANCE['quality_score_threshold']}")
            st.write(f"• Conflict detection: {'✅' if UID_GOVERNANCE['conflict_detection_enabled'] else '❌'}")
        
        with col2:
            st.markdown("**Matching Features:**")
            st.write(f"• Semantic matching: {'✅' if UID_GOVERNANCE['semantic_matching_enabled'] else '❌'}")
            st.write(f"• Standardized formatting: {'✅' if UID_GOVERNANCE['standardized_format_required'] else '❌'}")
            st.write(f"• Monthly quality checks: {'✅' if UID_GOVERNANCE['monthly_quality_check'] else '❌'}")
    
    # Token input
    surveymonkey_token = st.text_input("🔑 SurveyMonkey API Token", type="password", 
                                      value=st.secrets.get("surveymonkey", {}).get("token", ""))
    
    if not surveymonkey_token:
        st.warning("⚠️ Please enter your SurveyMonkey API token to continue.")
        return
    
    try:
        # Load reference data with governance check
        with st.spinner("🔄 Loading reference data and checking governance..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.error("❌ No reference questions found. Cannot proceed with UID matching.")
                return
            
            # Run governance check
            conflicts = detect_advanced_uid_conflicts(df_reference)
            governance_violations = [c for c in conflicts if c['type'] == 'governance_violation']
            
            if governance_violations:
                st.warning(f"⚠️ Found {len(governance_violations)} governance violations in reference data")
                with st.expander("View Governance Issues"):
                    for violation in governance_violations:
                        st.error(f"UID {violation['uid']}: {violation['count']} variations (max: {violation['max_allowed']})")
        
        # Survey selection
        surveys = get_surveys(surveymonkey_token)
        if not surveys:
            st.error("❌ No surveys found or API error.")
            return
        
        survey_options = {f"{s['id']} - {s['title']}": s['id'] for s in surveys}
        selected_survey = st.selectbox("📋 Select Survey to Configure", list(survey_options.keys()))
        
        if selected_survey:
            survey_id = survey_options[selected_survey]
            
            # Matching settings
            st.markdown("### 🔧 Enhanced Matching Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_semantic = st.checkbox("🧠 Use Semantic Matching", value=True, 
                                         help="Enable AI-powered semantic similarity matching")
                semantic_threshold = st.slider("Semantic Threshold", 0.5, 0.95, 
                                             float(UID_GOVERNANCE['semantic_similarity_threshold']), 0.05)
            
            with col2:
                force_standardization = st.checkbox("📝 Force Question Standardization", value=True,
                                                   help="Standardize question format before UID assignment")
                auto_categorize = st.checkbox("📊 Auto-categorize by Survey Title", value=True,
                                            help="Automatically categorize questions based on survey title")
            
            with col3:
                governance_mode = st.selectbox("⚖️ Governance Mode", 
                                             ["Strict", "Moderate", "Permissive"],
                                             help="Strict: Enforce all rules, Moderate: Warn on violations, Permissive: Log only")
                
                create_new_uids = st.checkbox("➕ Allow New UID Creation", value=True,
                                            help="Create new UIDs when no good match is found")
            
            # Process survey button
            if st.button("🚀 Process Survey with Enhanced Matching", type="primary"):
                with st.spinner("🔄 Processing survey with enhanced semantic matching..."):
                    
                    # Get survey details
                    survey_details = get_survey_details(survey_id, surveymonkey_token)
                    questions_data = extract_questions(survey_details)
                    df_target = pd.DataFrame(questions_data)
                    
                    if df_target.empty:
                        st.error("❌ No questions found in the selected survey.")
                        return
                    
                    # Auto-categorize survey if enabled
                    survey_category = None
                    if auto_categorize:
                        survey_title = survey_details.get('title', '')
                        survey_category = categorize_survey(survey_title)
                        st.info(f"📊 Auto-detected survey category: **{survey_category}**")
                    
                    # Create enhanced UID assignments
                    enhanced_results = []
                    existing_uids_data = {}
                    
                    # Build existing UIDs data structure
                    unique_bank = create_enhanced_unique_questions_bank(df_reference)
                    for _, row in unique_bank.iterrows():
                        existing_uids_data[row['uid']] = {
                            'best_question': row['best_question'],
                            'variation_count': row['total_variants'],
                            'survey_category': row['survey_category'],
                            'quality_score': row['quality_score']
                        }
                    
                    # Process each question
                    progress_bar = st.progress(0)
                    for i, (_, question_row) in enumerate(df_target.iterrows()):
                        
                        if question_row.get('question_category') == 'Heading':
                            # Skip headings for UID assignment
                            enhanced_results.append({
                                **question_row.to_dict(),
                                'Enhanced_UID': None,
                                'Enhancement_Method': 'heading_skipped',
                                'Enhancement_Confidence': 0.0,
                                'Governance_Status': 'N/A',
                                'Standardized_Question': question_row['heading_0']
                            })
                        else:
                            # Apply enhanced UID assignment
                            assignment_result = assign_uid_with_enhanced_governance(
                                question_row['heading_0'],
                                existing_uids_data,
                                survey_category,
                                force_new=not use_semantic
                            )
                            
                            # Apply governance checks
                            governance_status = "✅"
                            if governance_mode == "Strict" and not assignment_result['governance_compliant']:
                                governance_status = "❌"
                            elif governance_mode == "Moderate" and not assignment_result['governance_compliant']:
                                governance_status = "⚠️"
                            
                            enhanced_results.append({
                                **question_row.to_dict(),
                                'Enhanced_UID': assignment_result['uid'],
                                'Enhancement_Method': assignment_result['method'],
                                'Enhancement_Confidence': round(assignment_result['confidence'], 3),
                                'Governance_Status': governance_status,
                                'Standardized_Question': assignment_result['standardized_question']
                            })
                        
                        progress_bar.progress((i + 1) / len(df_target))
                    
                    # Create results dataframe
                    df_enhanced = pd.DataFrame(enhanced_results)
                    
                    # Apply choice inheritance (choices get parent question's UID)
                    for i, row in df_enhanced.iterrows():
                        if row.get('is_choice') and row.get('parent_question'):
                            parent_uid = df_enhanced[
                                (df_enhanced['heading_0'] == row['parent_question']) & 
                                (df_enhanced['is_choice'] == False)
                            ]['Enhanced_UID'].iloc[0] if len(df_enhanced[
                                (df_enhanced['heading_0'] == row['parent_question']) & 
                                (df_enhanced['is_choice'] == False)
                            ]) > 0 else None
                            
                            if parent_uid:
                                df_enhanced.at[i, 'Enhanced_UID'] = parent_uid
                                df_enhanced.at[i, 'Enhancement_Method'] = 'choice_inheritance'
                    
                    # Store results in session state
                    st.session_state.df_enhanced = df_enhanced
                    st.session_state.survey_category = survey_category
                    
                    # Display results summary
                    st.success("✅ Enhanced processing completed!")
                    
                    # Enhanced summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_questions = len(df_enhanced[df_enhanced['is_choice'] == False])
                        st.metric("📊 Total Questions", total_questions)
                    
                    with col2:
                        matched_questions = len(df_enhanced[
                            (df_enhanced['Enhanced_UID'].notna()) & 
                            (df_enhanced['is_choice'] == False) &
                            (df_enhanced['Enhancement_Method'] == 'semantic_match')
                        ])
                        st.metric("🧠 Semantic Matches", matched_questions)
                    
                    with col3:
                        new_uids = len(df_enhanced[
                            (df_enhanced['Enhanced_UID'].notna()) & 
                            (df_enhanced['is_choice'] == False) &
                            (df_enhanced['Enhancement_Method'] == 'new_assignment')
                        ])
                        st.metric("➕ New UIDs", new_uids)
                    
                    with col4:
                        governance_compliant = len(df_enhanced[df_enhanced['Governance_Status'] == '✅'])
                        total_with_uid = len(df_enhanced[df_enhanced['Enhanced_UID'].notna()])
                        compliance_rate = (governance_compliant / total_with_uid * 100) if total_with_uid > 0 else 0
                        st.metric("⚖️ Governance Rate", f"{compliance_rate:.1f}%")
                    
                    # Display method breakdown
                    st.markdown("### 📊 Enhancement Method Breakdown")
                    method_counts = df_enhanced[df_enhanced['is_choice'] == False]['Enhancement_Method'].value_counts()
                    
                    method_cols = st.columns(len(method_counts))
                    for i, (method, count) in enumerate(method_counts.items()):
                        with method_cols[i]:
                            method_name = method.replace('_', ' ').title()
                            st.metric(f"🔧 {method_name}", count)
                    
                    # Show sample results
                    st.markdown("### 📋 Sample Enhanced Results")
                    sample_df = df_enhanced[df_enhanced['is_choice'] == False].head(10)[
                        ['heading_0', 'Enhanced_UID', 'Enhancement_Method', 'Enhancement_Confidence', 'Governance_Status', 'Standardized_Question']
                    ].copy()
                    
                    sample_df = sample_df.rename(columns={
                        'heading_0': 'Original Question',
                        'Enhanced_UID': 'Assigned UID',
                        'Enhancement_Method': 'Method',
                        'Enhancement_Confidence': 'Confidence',
                        'Governance_Status': 'Governance',
                        'Standardized_Question': 'Standardized Format'
                    })
                    
                    st.dataframe(sample_df, use_container_width=True)
                    
                    # Download enhanced results
                    st.markdown("### 📥 Download Enhanced Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📥 Download All Enhanced Results",
                            df_enhanced.to_csv(index=False),
                            f"enhanced_survey_results_{survey_id}_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create governance report
                        governance_report = df_enhanced[
                            df_enhanced['Governance_Status'].isin(['❌', '⚠️'])
                        ][['heading_0', 'Enhanced_UID', 'Enhancement_Method', 'Governance_Status']].copy()
                        
                        if not governance_report.empty:
                            st.download_button(
                                "⚖️ Download Governance Issues",
                                governance_report.to_csv(index=False),
                                f"governance_issues_{survey_id}_{uuid4()}.csv",
                                "text/csv",
                                use_container_width=True
                            )
    
    except Exception as e:
        logger.error(f"Enhanced configure survey failed: {e}")
        st.error(f"❌ Error: {e}")

# Rest of the existing functions remain the same...

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

# Snowflake query functions
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
            raise
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total reference questions fetched: {len(final_df)}")
        return final_df
    else:
        logger.warning("No reference data fetched")
        return pd.DataFrame()

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

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "df_enhanced" not in st.session_state:
    st.session_state.df_enhanced = None
if "survey_category" not in st.session_state:
    st.session_state.survey_category = None

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 UID Matcher Pro Enhanced")
    st.markdown("Advanced semantic matching & governance")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced SurveyMonkey section
    st.markdown("**📊 Enhanced SurveyMonkey**")
    if st.button("⚙️ Enhanced Configure Survey", use_container_width=True):
        st.session_state.page = "enhanced_configure_survey"
        st.rerun()
    if st.button("👁️ View Surveys", use_container_width=True):
        st.session_state.page = "view_surveys"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Question Bank section
    st.markdown("**📚 Enhanced Question Bank**")
    if st.button("⭐ Enhanced Unique Bank", use_container_width=True):
        st.session_state.page = "enhanced_unique_bank"
        st.rerun()
    if st.button("📊 Category Analysis", use_container_width=True):
        st.session_state.page = "category_analysis"
        st.rerun()
    if st.button("🔍 Conflict Detection", use_container_width=True):
        st.session_state.page = "conflict_detection"
        st.rerun()
    if st.button("📈 Quality Dashboard", use_container_width=True):
        st.session_state.page = "quality_dashboard"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Governance section
    st.markdown("**⚖️ Enhanced Governance**")
    st.markdown(f"• Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
    st.markdown(f"• Semantic matching: {'✅' if UID_GOVERNANCE['semantic_matching_enabled'] else '❌'}")

# App UI with enhanced styling
st.markdown('<div class="main-header">🧠 Enhanced UID Matcher Pro: Semantic AI + Governance</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Enhanced Home Page
if st.session_state.page == "home":
    st.markdown("## 🏠 Enhanced UID Matcher Pro Dashboard")
    st.markdown("*Advanced semantic matching with AI governance and categorization*")
    
    # Enhanced dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🧠 AI Status", "Semantic Ready")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            with get_snowflake_engine().connect() as conn:
                result = conn.execute(text("SELECT COUNT(DISTINCT UID) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                count = result.fetchone()[0]
                st.metric("🆔 Unique UIDs", f"{count:,}")
        except:
            st.metric("🆔 Unique UIDs", "Connection Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Governance", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Categories", len(SURVEY_CATEGORIES))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced features showcase
    st.markdown("## 🚀 Enhanced AI Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Advanced Semantic Matching")
        st.markdown("• **AI-Powered Similarity**: Using sentence transformers for deep understanding")
        st.markdown("• **Standardized Formatting**: Auto-format questions before matching")
        st.markdown("• **Confidence Scoring**: Transparent matching confidence levels")
        st.markdown("• **Governance Integration**: Real-time compliance checking")
        
        if st.button("⚙️ Try Enhanced Configure Survey", use_container_width=True):
            st.session_state.page = "enhanced_configure_survey"
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Smart Categorization")
        st.markdown("• **Auto-Detection**: Smart survey category identification")
        st.markdown("• **8 Categories**: Application, Pre programme, Enrollment, Progress Review, Impact, GROW, Feedback, Pulse")
        st.markdown("• **Category Analytics**: Cross-category quality analysis")
        st.markdown("• **Governance by Category**: Category-specific compliance tracking")
        
        if st.button("📊 Explore Category Analysis", use_container_width=True):
            st.session_state.page = "category_analysis"
            st.rerun()
    
    # Quick actions with enhanced features
    st.markdown("---")
    st.markdown("## 🎯 Enhanced Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Run Conflict Detection", use_container_width=True):
            st.session_state.page = "conflict_detection"
            st.rerun()
    
    with col2:
        if st.button("📈 Quality Dashboard", use_container_width=True):
            st.session_state.page = "quality_dashboard"
            st.rerun()
    
    with col3:
        if st.button("⭐ Enhanced Question Bank", use_container_width=True):
            st.session_state.page = "enhanced_unique_bank"
            st.rerun()

# Enhanced Configure Survey Page
elif st.session_state.page == "enhanced_configure_survey":
    enhanced_configure_survey_page()

# Enhanced Unique Question Bank Page
elif st.session_state.page == "enhanced_unique_bank":
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*AI-powered question bank with semantic matching and governance compliance*")
    
    try:
        with st.spinner("🧠 Loading enhanced question bank with AI analysis..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found.</div>', unsafe_allow_html=True)
            else:
                enhanced_unique_df = create_enhanced_unique_questions_bank(df_reference)
        
        if enhanced_unique_df.empty:
            st.markdown('<div class="warning-card">⚠️ No enhanced questions found.</div>', unsafe_allow_html=True)
        else:
            # Enhanced summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🆔 Unique UIDs", len(enhanced_unique_df))
            with col2:
                total_variants = enhanced_unique_df['total_variants'].sum()
                st.metric("📝 Total Variants", f"{total_variants:,}")
            with col3:
                governance_compliant = len(enhanced_unique_df[enhanced_unique_df['governance_compliant'] == True])
                st.metric("⚖️ Compliant", f"{governance_compliant}/{len(enhanced_unique_df)}")
            with col4:
                avg_quality = enhanced_unique_df['quality_score'].mean()
                st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
            with col5:
                categories = enhanced_unique_df['survey_category'].nunique()
                st.metric("📊 Categories", categories)
            
            st.markdown("---")
            
            # Enhanced filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter...")
            
            with col2:
                category_filter = st.selectbox("📊 Category Filter", 
                    ["All"] + sorted(enhanced_unique_df['survey_category'].unique()))
            
            with col3:
                governance_filter = st.selectbox("⚖️ Governance Filter", 
                    ["All", "Compliant Only", "Violations Only"])
            
            with col4:
                quality_filter = st.selectbox("🎯 Quality Filter", 
                    ["All", "High (>10)", "Medium (5-10)", "Low (<5)"])
            
            # Apply enhanced filters
            filtered_df = enhanced_unique_df.copy()
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['best_question'].str.contains(search_term, case=False, na=False) |
                    filtered_df['standardized_question'].str.contains(search_term, case=False, na=False)
                ]
            
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['survey_category'] == category_filter]
            
            if governance_filter == "Compliant Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == True]
            elif governance_filter == "Violations Only":
                filtered_df = filtered_df[filtered_df['governance_compliant'] == False]
            
            if quality_filter == "High (>10)":
                filtered_df = filtered_df[filtered_df['quality_score'] > 10]
            elif quality_filter == "Medium (5-10)":
                filtered_df = filtered_df[(filtered_df['quality_score'] >= 5) & (filtered_df['quality_score'] <= 10)]
            elif quality_filter == "Low (<5)":
                filtered_df = filtered_df[filtered_df['quality_score'] < 5]
            
            st.markdown(f"### 📋 Enhanced Results ({len(filtered_df)} questions)")
            
            if not filtered_df.empty:
                # Display enhanced results
                display_df = filtered_df[
                    ['uid', 'best_question', 'standardized_question', 'survey_category', 
                     'total_variants', 'quality_score', 'governance_compliant', 'issues']
                ].copy()
                
                display_df['governance_compliant'] = display_df['governance_compliant'].apply(lambda x: "✅" if x else "❌")
                
                display_df = display_df.rename(columns={
                    'uid': 'UID',
                    'best_question': 'Original Question',
                    'standardized_question': 'Standardized Format',
                    'survey_category': 'Category',
                    'total_variants': 'Variants',
                    'quality_score': 'Quality',
                    'governance_compliant': 'Governance',
                    'issues': 'Issues'
                })
                
                st.dataframe(
                    display_df,
                    column_config={
                        "UID": st.column_config.TextColumn("UID", width="small"),
                        "Original Question": st.column_config.TextColumn("Original Question", width="large"),
                        "Standardized Format": st.column_config.TextColumn("Standardized Format", width="large"),
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Variants": st.column_config.NumberColumn("Variants", width="small"),
                        "Quality": st.column_config.NumberColumn("Quality", format="%.1f", width="small"),
                        "Governance": st.column_config.TextColumn("Governance", width="small"),
                        "Issues": st.column_config.TextColumn("Issues", width="medium")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Enhanced downloads
                st.markdown("### 📥 Enhanced Downloads")
                
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
                    governance_issues = filtered_df[filtered_df['governance_compliant'] == False]
                    if not governance_issues.empty:
                        st.download_button(
                            "⚖️ Download Governance Issues",
                            governance_issues.to_csv(index=False),
                            f"governance_violations_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    # Category breakdown
                    category_summary = enhanced_unique_df.groupby('survey_category').agg({
                        'uid': 'count',
                        'quality_score': 'mean',
                        'governance_compliant': lambda x: (x == True).sum(),
                        'total_variants': 'sum'
                    }).round(2)
                    
                    st.download_button(
                        "📊 Download Category Summary",
                        category_summary.to_csv(),
                        f"category_summary_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ No questions match your filters.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Enhanced unique bank failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Category Analysis Page
elif st.session_state.page == "category_analysis":
    st.markdown("## 📊 Enhanced Category Analysis")
    st.markdown("*Comprehensive analysis of questions by survey categories with AI insights*")
    
    try:
        with st.spinner("🔄 Loading category analysis..."):
            df_reference = get_all_reference_questions()
            enhanced_unique_df = create_enhanced_unique_questions_bank(df_reference)
        
        if enhanced_unique_df.empty:
            st.markdown('<div class="warning-card">⚠️ No data available for analysis.</div>', unsafe_allow_html=True)
        else:
            # Category overview with enhanced metrics
            st.markdown("### 📊 Category Overview")
            
            category_stats = enhanced_unique_df.groupby('survey_category').agg({
                'uid': 'count',
                'total_variants': ['sum', 'mean'],
                'quality_score': ['mean', 'std'],
                'governance_compliant': lambda x: (x == True).sum(),
                'category_confidence': 'mean'
            }).round(2)
            
            # Flatten column names
            category_stats.columns = ['Questions', 'Total_Variants', 'Avg_Variants', 'Avg_Quality', 'Quality_Std', 'Governance_Compliant', 'Avg_Confidence']
            category_stats['Governance_Rate'] = (category_stats['Governance_Compliant'] / category_stats['Questions'] * 100).round(1)
            category_stats = category_stats.sort_values('Questions', ascending=False)
            
            # Display category cards
            categories = list(category_stats.index)
            cols = st.columns(min(4, len(categories)))
            
            for i, category in enumerate(categories):
                with cols[i % 4]:
                    stats = category_stats.loc[category]
                    
                    # Determine card color based on governance rate
                    if stats['Governance_Rate'] >= 90:
                        card_class = "success-card"
                    elif stats['Governance_Rate'] >= 70:
                        card_class = "warning-card"
                    else:
                        card_class = "governance-violation"
                    
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    st.markdown(f"**📋 {category}**")
                    st.markdown(f"Questions: {stats['Questions']}")
                    st.markdown(f"Quality: {stats['Avg_Quality']:.1f}")
                    st.markdown(f"Governance: {stats['Governance_Rate']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed category statistics table
            st.markdown("### 📈 Detailed Category Statistics")
            
            display_stats = category_stats.copy()
            display_stats = display_stats.rename(columns={
                'Questions': 'Total Questions',
                'Total_Variants': 'All Variants',
                'Avg_Variants': 'Avg Variants/UID',
                'Avg_Quality': 'Avg Quality Score',
                'Quality_Std': 'Quality Std Dev',
                'Governance_Compliant': 'Governance Compliant',
                'Avg_Confidence': 'Avg Category Confidence',
                'Governance_Rate': 'Governance Rate (%)'
            })
            
            st.dataframe(
                display_stats,
                column_config={
                    "Total Questions": st.column_config.NumberColumn("Total Questions", width="small"),
                    "All Variants": st.column_config.NumberColumn("All Variants", width="small"),
                    "Avg Variants/UID": st.column_config.NumberColumn("Avg Variants/UID", format="%.1f", width="small"),
                    "Avg Quality Score": st.column_config.NumberColumn("Avg Quality Score", format="%.1f", width="small"),
                    "Quality Std Dev": st.column_config.NumberColumn("Quality Std Dev", format="%.1f", width="small"),
                    "Governance Compliant": st.column_config.NumberColumn("Governance Compliant", width="small"),
                    "Avg Category Confidence": st.column_config.NumberColumn("Avg Category Confidence", format="%.2f", width="small"),
                    "Governance Rate (%)": st.column_config.NumberColumn("Governance Rate (%)", format="%.1f", width="small")
                },
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Category-specific analysis
            st.markdown("### 🔍 Category-Specific Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_category = st.selectbox(
                    "📊 Select Category for Deep Analysis",
                    ["All"] + sorted(enhanced_unique_df['survey_category'].unique())
                )
            
            with col2:
                analysis_type = st.selectbox(
                    "🔬 Analysis Type",
                    ["Overview", "Quality Analysis", "Governance Issues", "Question Samples"]
                )
            
            # Filter data by category
            if selected_category == "All":
                analysis_df = enhanced_unique_df.copy()
            else:
                analysis_df = enhanced_unique_df[enhanced_unique_df['survey_category'] == selected_category].copy()
            
            if analysis_type == "Overview":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Questions", len(analysis_df))
                with col2:
                    avg_quality = analysis_df['quality_score'].mean()
                    st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
                with col3:
                    governance_rate = (analysis_df['governance_compliant'] == True).sum() / len(analysis_df) * 100
                    st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
                with col4:
                    total_variants = analysis_df['total_variants'].sum()
                    st.metric("📝 Total Variants", total_variants)
                
                # Show distribution charts would go here if we had plotting libraries
                st.markdown("**Quality Score Distribution:**")
                quality_bins = pd.cut(analysis_df['quality_score'], bins=5)
                quality_dist = quality_bins.value_counts().sort_index()
                for interval, count in quality_dist.items():
                    st.write(f"• {interval}: {count} questions")
            
            elif analysis_type == "Quality Analysis":
                st.markdown(f"### 🎯 Quality Analysis - {selected_category}")
                
                # Quality metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_quality = len(analysis_df[analysis_df['quality_score'] > 10])
                    st.metric("🌟 High Quality (>10)", high_quality)
                
                with col2:
                    medium_quality = len(analysis_df[(analysis_df['quality_score'] >= 5) & (analysis_df['quality_score'] <= 10)])
                    st.metric("📊 Medium Quality (5-10)", medium_quality)
                
                with col3:
                    low_quality = len(analysis_df[analysis_df['quality_score'] < 5])
                    st.metric("⚠️ Low Quality (<5)", low_quality)
                
                # Show quality issues
                quality_issues = analysis_df[analysis_df['issues'] != 'none']
                if not quality_issues.empty:
                    st.markdown("**Questions with Quality Issues:**")
                    issue_display = quality_issues[['uid', 'best_question', 'quality_score', 'issues']].copy()
                    st.dataframe(issue_display, use_container_width=True)
            
            elif analysis_type == "Governance Issues":
                st.markdown(f"### ⚖️ Governance Issues - {selected_category}")
                
                governance_violations = analysis_df[analysis_df['governance_compliant'] == False]
                
                if governance_violations.empty:
                    st.success("✅ No governance violations found in this category!")
                else:
                    st.error(f"❌ Found {len(governance_violations)} governance violations")
                    
                    violation_display = governance_violations[
                        ['uid', 'best_question', 'total_variants', 'issues']
                    ].copy()
                    
                    violation_display = violation_display.rename(columns={
                        'uid': 'UID',
                        'best_question': 'Question',
                        'total_variants': 'Variants Count',
                        'issues': 'Issues'
                    })
                    
                    st.dataframe(violation_display, use_container_width=True)
                    
                    # Governance recommendations
                    st.markdown("**🔧 Recommended Actions:**")
                    total_excess = (governance_violations['total_variants'] - UID_GOVERNANCE['max_variations_per_uid']).sum()
                    st.write(f"• Consolidate {total_excess} excess variations")
                    st.write(f"• Review {len(governance_violations)} UIDs for semantic duplicates")
                    st.write("• Implement standardized question formatting")
            
            elif analysis_type == "Question Samples":
                st.markdown(f"### 📋 Question Samples - {selected_category}")
                
                # Show best questions (highest quality)
                best_questions = analysis_df.nlargest(5, 'quality_score')[
                    ['uid', 'best_question', 'quality_score', 'total_variants']
                ]
                
                st.markdown("**🌟 Highest Quality Questions:**")
                for _, row in best_questions.iterrows():
                    st.markdown(f"**UID {row['uid']}** (Quality: {row['quality_score']:.1f}, Variants: {row['total_variants']})")
                    st.write(f"_{row['best_question']}_")
                    st.markdown("---")
                
                # Show problematic questions
                if not analysis_df[analysis_df['governance_compliant'] == False].empty:
                    problematic = analysis_df[analysis_df['governance_compliant'] == False].head(3)
                    
                    st.markdown("**⚠️ Questions Needing Attention:**")
                    for _, row in problematic.iterrows():
                        st.markdown(f"**UID {row['uid']}** (Issues: {row['issues']})")
                        st.write(f"_{row['best_question']}_")
                        st.markdown("---")
            
            # Download category-specific data
            st.markdown("### 📥 Download Category Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    f"📥 Download {selected_category} Data",
                    analysis_df.to_csv(index=False),
                    f"{selected_category.lower()}_analysis_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📊 Download Category Statistics",
                    category_stats.to_csv(),
                    f"category_statistics_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                if selected_category != "All":
                    governance_issues = analysis_df[analysis_df['governance_compliant'] == False]
                    if not governance_issues.empty:
                        st.download_button(
                            "⚖️ Download Governance Issues",
                            governance_issues.to_csv(index=False),
                            f"governance_issues_{selected_category.lower()}_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
    except Exception as e:
        logger.error(f"Category analysis failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Conflict Detection Page
elif st.session_state.page == "conflict_detection":
    st.markdown("## 🔍 Advanced Conflict Detection")
    st.markdown("*AI-powered conflict detection with semantic analysis and governance compliance*")
    
    try:
        with st.spinner("🔄 Running advanced conflict detection..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found.</div>', unsafe_allow_html=True)
            else:
                conflicts = detect_advanced_uid_conflicts(df_reference)
        
        if not conflicts:
            st.success("✅ No conflicts detected! Your UID data is clean.")
        else:
            st.warning(f"⚠️ Found {len(conflicts)} potential conflicts")
            
            # Categorize conflicts by type
            conflict_types = {}
            for conflict in conflicts:
                conflict_type = conflict['type']
                if conflict_type not in conflict_types:
                    conflict_types[conflict_type] = []
                conflict_types[conflict_type].append(conflict)
            
            # Display conflicts by type
            for conflict_type, type_conflicts in conflict_types.items():
                st.markdown(f"### 🚨 {conflict_type.replace('_', ' ').title()} ({len(type_conflicts)} issues)")
                
                if conflict_type == "governance_violation":
                    st.markdown("*UIDs exceeding the maximum allowed variations*")
                    
                    for conflict in type_conflicts:
                        severity_icon = "🔴" if conflict['severity'] == 'high' else "🟡"
                        
                        with st.expander(f"{severity_icon} UID {conflict['uid']} - {conflict['count']} variations"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Current Count:** {conflict['count']}")
                                st.write(f"**Max Allowed:** {conflict['max_allowed']}")
                            
                            with col2:
                                st.write(f"**Excess:** {conflict['count'] - conflict['max_allowed']}")
                                st.write(f"**Severity:** {conflict['severity']}")
                            
                            with col3:
                                if conflict.get('auto_fix_available'):
                                    st.success("🔧 Auto-fix available")
                                else:
                                    st.info("👋 Manual review needed")
                
                elif conflict_type == "semantic_conflict":
                    st.markdown("*UIDs with semantically different questions*")
                    
                    for conflict in type_conflicts:
                        with st.expander(f"🧠 UID {conflict['uid']} - Low semantic similarity ({conflict['avg_similarity']:.2f})"):
                            st.write(f"**Average Similarity:** {conflict['avg_similarity']:.3f}")
                            st.write(f"**Question Count:** {conflict['question_count']}")
                            
                            st.markdown("**Sample Questions:**")
                            for i, question in enumerate(conflict['sample_questions'], 1):
                                st.write(f"{i}. {question[:100]}{'...' if len(question) > 100 else ''}")
                
                elif conflict_type == "cross_uid_semantic_match":
                    st.markdown("*Different UIDs with semantically similar questions*")
                    
                    for conflict in type_conflicts:
                        similarity_pct = conflict['similarity'] * 100
                        
                        with st.expander(f"🔄 UID {conflict['uid1']} ↔ UID {conflict['uid2']} - {similarity_pct:.1f}% similar"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**UID {conflict['uid1']}:**")
                                st.write(f"_{conflict['question1']}_")
                            
                            with col2:
                                st.markdown(f"**UID {conflict['uid2']}:**")
                                st.write(f"_{conflict['question2']}_")
                            
                            st.write(f"**Semantic Similarity:** {similarity_pct:.1f}%")
                            
                            if conflict.get('auto_consolidate_recommended'):
                                st.warning("🔄 Auto-consolidation recommended")
                
                elif conflict_type == "quality_inconsistency":
                    st.markdown("*UIDs with inconsistent question quality*")
                    
                    for conflict in type_conflicts:
                        with st.expander(f"📊 UID {conflict['uid']} - Quality inconsistency (std: {conflict['quality_std']:.1f})"):
                            st.write(f"**Quality Std Dev:** {conflict['quality_std']:.2f}")
                            st.write(f"**Quality Range:** {conflict['quality_range'][0]:.1f} - {conflict['quality_range'][1]:.1f}")
                            
                            if 'recommendations' in conflict:
                                st.markdown("**Recommendations:**")
                                for rec in conflict['recommendations']:
                                    st.write(f"• {rec}")
        
        # Action buttons for conflict resolution
        if conflicts:
            st.markdown("---")
            st.markdown("### 🔧 Conflict Resolution Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Generate Consolidation Plan", use_container_width=True):
                    consolidation_plan = []
                    
                    for conflict in conflicts:
                        if conflict['type'] == 'governance_violation':
                            consolidation_plan.append({
                                'action': 'reduce_variations',
                                'uid': conflict['uid'],
                                'current_count': conflict['count'],
                                'target_count': conflict['max_allowed'],
                                'method': 'keep_best_quality'
                            })
                        elif conflict['type'] == 'cross_uid_semantic_match':
                            consolidation_plan.append({
                                'action': 'merge_uids',
                                'source_uid': conflict['uid2'],
                                'target_uid': conflict['uid1'],
                                'similarity': conflict['similarity'],
                                'method': 'semantic_merge'
                            })
                    
                    if consolidation_plan:
                        plan_df = pd.DataFrame(consolidation_plan)
                        st.dataframe(plan_df, use_container_width=True)
                        
                        st.download_button(
                            "📥 Download Consolidation Plan",
                            plan_df.to_csv(index=False),
                            f"consolidation_plan_{uuid4()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
            
            with col2:
                monthly_report = run_monthly_quality_check(df_reference)
                
                st.download_button(
                    "📊 Download Quality Report",
                    json.dumps(monthly_report, indent=2),
                    f"monthly_quality_report_{uuid4()}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export all conflicts
                conflicts_df = pd.DataFrame(conflicts)
                
                st.download_button(
                    "🔍 Download All Conflicts",
                    conflicts_df.to_csv(index=False),
                    f"detected_conflicts_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"Conflict detection failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Quality Dashboard Page
elif st.session_state.page == "quality_dashboard":
    st.markdown("## 📈 Enhanced Quality Dashboard")
    st.markdown("*Comprehensive quality metrics with AI insights and governance tracking*")
    
    try:
        with st.spinner("📊 Generating quality dashboard..."):
            df_reference = get_all_reference_questions()
            enhanced_unique_df = create_enhanced_unique_questions_bank(df_reference)
            monthly_report = run_monthly_quality_check(df_reference)
        
        if enhanced_unique_df.empty:
            st.markdown('<div class="warning-card">⚠️ No data available for quality analysis.</div>', unsafe_allow_html=True)
        else:
            # Overall quality metrics
            st.markdown("### 📊 Overall Quality Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_questions = len(enhanced_unique_df)
                st.metric("📊 Total Questions", total_questions)
            
            with col2:
                avg_quality = enhanced_unique_df['quality_score'].mean()
                quality_trend = "📈" if avg_quality > UID_GOVERNANCE['quality_score_threshold'] else "📉"
                st.metric("🎯 Avg Quality Score", f"{avg_quality:.1f} {quality_trend}")
            
            with col3:
                governance_rate = (enhanced_unique_df['governance_compliant'] == True).sum() / len(enhanced_unique_df) * 100
                governance_icon = "✅" if governance_rate > 90 else "⚠️" if governance_rate > 70 else "❌"
                st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}% {governance_icon}")
            
            with col4:
                quality_violations = len(monthly_report['quality_issues'])
                st.metric("🚨 Quality Issues", quality_violations)
            
            with col5:
                governance_violations = len(monthly_report['governance_violations'])
                st.metric("⚠️ Governance Violations", governance_violations)
            
            st.markdown("---")
            
            # Quality distribution
            st.markdown("### 📊 Quality Score Distribution")
            
            quality_ranges = [
                ("🌟 Excellent (>15)", enhanced_unique_df['quality_score'] > 15),
                ("🎯 Good (10-15)", (enhanced_unique_df['quality_score'] >= 10) & (enhanced_unique_df['quality_score'] <= 15)),
                ("📊 Average (5-10)", (enhanced_unique_df['quality_score'] >= 5) & (enhanced_unique_df['quality_score'] < 10)),
                ("⚠️ Below Average (0-5)", (enhanced_unique_df['quality_score'] >= 0) & (enhanced_unique_df['quality_score'] < 5)),
                ("🚨 Poor (<0)", enhanced_unique_df['quality_score'] < 0)
            ]
            
            quality_cols = st.columns(len(quality_ranges))
            
            for i, (label, condition) in enumerate(quality_ranges):
                count = condition.sum()
                percentage = (count / len(enhanced_unique_df) * 100) if len(enhanced_unique_df) > 0 else 0
                
                with quality_cols[i]:
                    st.metric(label, f"{count} ({percentage:.1f}%)")
            
            st.markdown("---")
            
            # Category-wise quality analysis
            st.markdown("### 📊 Quality by Category")
            
            category_quality = enhanced_unique_df.groupby('survey_category').agg({
                'quality_score': ['count', 'mean', 'std', 'min', 'max'],
                'governance_compliant': lambda x: (x == True).sum()
            }).round(2)
            
            category_quality.columns = ['Count', 'Mean_Quality', 'Std_Quality', 'Min_Quality', 'Max_Quality', 'Governance_Compliant']
            category_quality['Governance_Rate'] = (category_quality['Governance_Compliant'] / category_quality['Count'] * 100).round(1)
            
            # Display category quality cards
            for category, stats in category_quality.iterrows():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Determine quality status
                    if stats['Mean_Quality'] > 10:
                        status_icon = "🌟"
                        card_class = "success-card"
                    elif stats['Mean_Quality'] > 5:
                        status_icon = "📊"
                        card_class = "info-card"
                    else:
                        status_icon = "⚠️"
                        card_class = "warning-card"
                    
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    st.markdown(f"**{status_icon} {category}**")
                    st.markdown(f"Questions: {stats['Count']}")
                    st.markdown(f"Avg Quality: {stats['Mean_Quality']:.1f}")
                    st.markdown(f"Governance: {stats['Governance_Rate']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Monthly quality report
            st.markdown("### 📅 Monthly Quality Report")
            
            report_date = datetime.fromisoformat(monthly_report['check_date']).strftime("%B %Y")
            st.markdown(f"**Report Date:** {report_date}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Summary:**")
                st.write(f"• Total Questions: {monthly_report['total_questions']:,}")
                st.write(f"• Unique UIDs: {monthly_report['unique_uids']:,}")
                st.write(f"• Governance Violations: {len(monthly_report['governance_violations'])}")
                st.write(f"• Quality Issues: {len(monthly_report['quality_issues'])}")
            
            with col2:
                st.markdown("**🔧 Recommendations:**")
                for rec in monthly_report['recommendations']:
                    st.write(f"• {rec}")
            
            # Detailed issues
            if monthly_report['governance_violations']:
                st.markdown("**⚖️ Governance Violations:**")
                
                violations_df = pd.DataFrame(monthly_report['governance_violations'])
                violations_display = violations_df[['uid', 'variation_count', 'excess', 'severity']].copy()
                violations_display = violations_display.rename(columns={
                    'uid': 'UID',
                    'variation_count': 'Current Count',
                    'excess': 'Excess Variations',
                    'severity': 'Severity'
                })
                
                st.dataframe(violations_display, use_container_width=True)
            
            if monthly_report['quality_issues']:
                st.markdown("**🎯 Quality Issues:**")
                
                quality_issues_df = pd.DataFrame(monthly_report['quality_issues'])
                quality_display = quality_issues_df[['uid', 'low_quality_count', 'total_variations']].copy()
                quality_display = quality_display.rename(columns={
                    'uid': 'UID',
                    'low_quality_count': 'Low Quality Questions',
                    'total_variations': 'Total Variations'
                })
                
                st.dataframe(quality_display, use_container_width=True)
            
            # Download quality reports
            st.markdown("### 📥 Download Quality Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📊 Download Quality Dashboard",
                    enhanced_unique_df.to_csv(index=False),
                    f"quality_dashboard_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📅 Download Monthly Report",
                    json.dumps(monthly_report, indent=2),
                    f"monthly_quality_report_{uuid4()}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                st.download_button(
                    "📊 Download Category Quality Analysis",
                    category_quality.to_csv(),
                    f"category_quality_analysis_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"Quality dashboard failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Add any other existing pages here...
else:
    st.markdown("### 🚧 Page Under Development")
    st.markdown("This page is being enhanced with new AI-powered features.")
    st.markdown("Please use the sidebar to navigate to available enhanced features.")
    
    # Quick navigation for development
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Go to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        if st.button("⚙️ Enhanced Configure", use_container_width=True):
            st.session_state.page = "enhanced_configure_survey"
            st.rerun()
    
    with col3:
        if st.button("⭐ Enhanced Question Bank", use_container_width=True):
            st.session_state.page = "enhanced_unique_bank"
            st.rerun()
