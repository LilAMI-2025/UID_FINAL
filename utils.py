"""
Utility functions for UID Matcher Reflex App
Contains data processing, matching algorithms, and helper functions
"""

import re
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ============= CONSTANTS =============

# Matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"

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

# AMI Structure Categories
SURVEY_STAGES = {
    "Recruitment Survey": ["application", "apply", "applying", "candidate", "candidacy"],
    "Pre-Programme Survey": ["pre programme", "pre-programme", "pre program", "pre-program"],
    "LL Feedback Survey": ["ll feedback", "learning lab", "in-person", "multilingual"],
    "Pulse Check Survey": ["pulse", "check-in", "checkin", "pulse check"],
    "Progress Review Survey": ["progress", "review", "assessment", "evaluation"],
    "Growth Goal Reflection": ["growth goal", "post-ll", "reflection"],
    "AP Survey": ["ap survey", "accountability partner", "ap post"],
    "Longitudinal Survey": ["longitudinal", "impact", "annual impact"],
    "CEO/Client Lead Survey": ["ceo", "client lead", "clientlead"],
    "Other": ["drop-out", "attrition", "finance link", "mentorship application"]
}

RESPONDENT_TYPES = {
    "Participant": ["participant", "learner", "student", "individual", "person"],
    "Business": ["business", "enterprise", "company", "entrepreneur", "owner"],
    "Team member": ["team member", "staff", "employee", "worker"],
    "Accountability Partner": ["accountability partner", "ap", "manager", "supervisor"],
    "Client Lead": ["client lead", "ceo", "executive", "leadership"],
    "Managers": ["managers", "management", "supervisor"]
}

PROGRAMMES = {
    "Grow Your Business (GYB)": ["gyb", "grow your business", "grow business"],
    "Micro Enterprise Accelerator (MEA)": ["mea", "micro enterprise", "accelerator"],
    "Start your Business (SYB)": ["syb", "start your business", "start business"],
    "Leadership Development Programme (LDP)": ["ldp", "leadership development", "leadership"],
    "Management Development Programme (MDP)": ["mdp", "management development", "management"],
    "Thrive at Work (T@W)": ["taw", "thrive at work", "thrive", "t@w"],
    "Bootcamp": ["bootcamp", "boot camp", "survival bootcamp", "work readiness"],
    "Academy": ["academy", "care academy"],
    "Finance Link": ["finance link"],
    "Custom": ["winning behaviours", "custom", "learning needs"],
    "ALL": ["all programmes", "template", "multilingual"]
}

# ============= TEXT PROCESSING FUNCTIONS =============

def enhanced_normalize(text: str, synonym_map: Dict[str, str] = ENHANCED_SYNONYM_MAP) -> str:
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

def score_question_quality(question: str) -> int:
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

def clean_question_text(text: str) -> str:
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

# ============= AMI CATEGORIZATION FUNCTIONS =============

def categorize_survey_by_ami_structure(title: str) -> Dict[str, str]:
    """Categorize survey based on AMI structure: Survey Stage, Respondent Type, Programme"""
    if not isinstance(title, str):
        return {"Survey Stage": "Other", "Respondent Type": "Participant", "Programme": "Custom"}
    
    title_lower = title.lower().strip()
    
    # Determine Survey Stage
    survey_stage = "Other"
    for stage, keywords in SURVEY_STAGES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            survey_stage = stage
            break
    
    # Determine Respondent Type
    respondent_type = "Participant"
    for resp_type, keywords in RESPONDENT_TYPES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            respondent_type = resp_type
            break
    
    # Determine Programme
    programme = "Custom"
    for prog, keywords in PROGRAMMES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            programme = prog
            break
    
    return {
        "Survey Stage": survey_stage,
        "Respondent Type": respondent_type,
        "Programme": programme
    }

# ============= IDENTITY DETECTION FUNCTIONS =============

def contains_identity_info(text: str) -> bool:
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

def determine_identity_type(text: str) -> str:
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
    else:
        return 'Other'

# ============= UID MATCHING FUNCTIONS =============

def get_tfidf_vectors(df_reference: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    """Create TF-IDF vectors for reference questions"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

def load_sentence_transformer():
    """Load the sentence transformer model"""
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

def run_uid_match(question_bank: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
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
        
        # Process questions
        batch_vectors = vectorizer.transform(df_target_norm["norm_text"])
        
        # Calculate TF-IDF similarities
        tfidf_similarities = cosine_similarity(batch_vectors, reference_vectors)
        
        # Process each question
        for i, (idx, row) in enumerate(df_target_norm.iterrows()):
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
                    max_semantic_score = max(semantic_scores).item()
                    
                    if max_semantic_score >= SEMANTIC_THRESHOLD:
                        max_semantic_idx = semantic_scores.argmax().item()
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

# ============= EXPORT FUNCTIONS =============

def prepare_export_data(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def calculate_matched_percentage(df_final: pd.DataFrame) -> float:
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

# ============= SURVEY PROCESSING FUNCTIONS =============

def extract_questions(survey_json: Dict) -> List[Dict]:
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
                    "required": False
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
                            "required": False
                        })
    return questions 