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

# Synonym Mapping
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

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

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def get_best_question_for_uid(questions_list):
    """
    Select the best structured question from a list of questions with the same UID.
    Prioritizes English format and better structure.
    """
    if not questions_list:
        return None
    
    # Score each question based on quality indicators
    def score_question(question):
        score = 0
        text = str(question).lower()
        
        # Prefer questions that are complete sentences
        if text.endswith('?'):
            score += 10
        
        # Prefer questions with proper capitalization (indicates better formatting)
        if any(c.isupper() for c in question):
            score += 5
        
        # Prefer longer, more descriptive questions
        word_count = len(text.split())
        if 5 <= word_count <= 20:
            score += 8
        elif word_count > 20:
            score += 3
        
        # Avoid very short or incomplete questions
        if word_count < 3:
            score -= 10
        
        # Prefer questions without HTML tags or special formatting
        if '<' not in text and '>' not in text:
            score += 5
        
        # Prefer questions that don't contain common formatting artifacts
        artifacts = ['click here', 'please select', '...', 'n/a', 'other']
        if not any(artifact in text for artifact in artifacts):
            score += 3
        
        # Prefer questions that look like proper English
        english_indicators = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were']
        if any(indicator in text for indicator in english_indicators):
            score += 7
        
        return score
    
    # Score all questions and return the best one
    scored_questions = [(q, score_question(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

def create_unique_questions_bank(df_reference):
    """
    Create a unique questions bank with the best question for each UID.
    """
    if df_reference.empty:
        return pd.DataFrame()
    
    # Group by UID and get the best question for each
    unique_questions = []
    
    for uid in df_reference['uid'].unique():
        if pd.isna(uid):
            continue
            
        uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
        best_question = get_best_question_for_uid(uid_questions)
        
        if best_question:
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split())
            })
    
    unique_df = pd.DataFrame(unique_questions)
    
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

# Snowflake Queries
def run_snowflake_reference_query(limit=10000, offset=0):
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
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning(
                "🔒 Cannot fetch Snowflake data: User account is locked. "
                "UID matching is disabled. Please resolve the lockout and retry."
            )
        elif "invalid identifier" in str(e).lower():
            st.warning(
                "⚠️ Snowflake query failed due to invalid column. "
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Contact your Snowflake admin to verify table schema."
            )
        raise

def run_snowflake_target_query():
    query = """
        SELECT DISTINCT HEADING_0
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
                "🔒 Cannot fetch Snowflake data: User account is locked. "
                "Please resolve the lockout and retry."
            )
        raise

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

# UID Matching functions
def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
    df_reference["norm_text"] = df_reference["heading_0"].apply(enhanced_normalize)
    df_target["norm_text"] = df_target["heading_0"].apply(enhanced_normalize)

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs = [], [], [], []
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
        matched_uids.append(df_reference.iloc[best_idx]["uid"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["heading_0"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    try:
        model = load_sentence_transformer()
        emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
        emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb_target, emb_ref)

        sem_matches, sem_scores = [], []
        for i in range(len(df_target)):
            best_idx = cosine_scores[i].argmax().item()
            score = cosine_scores[i][best_idx].item()
            sem_matches.append(df_reference.iloc[best_idx]["uid"] if score >= SEMANTIC_THRESHOLD else None)
            sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

        df_target["Semantic_UID"] = sem_matches
        df_target["Semantic_Similarity"] = sem_scores
        return df_target
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        st.error(f"🚨 Semantic matching failed: {e}")
        return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    
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
    
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes provided.")
        st.error("🚨 Input data is empty.")
        return pd.DataFrame()

    if len(df_target) > 10000:
        st.warning("⚠️ Large dataset detected. Processing may take time.")

    logger.info(f"Processing {len(df_target)} target questions against {len(df_reference)} reference questions.")
    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"🔄 Processing batch {start//batch_size + 1}..."):
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
    st.markdown("### 🧠 UID Matcher Pro")
    st.markdown("Navigate through the application")
    
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
    if st.button("⚙️ Configure Survey", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()
    if st.button("➕ Create New Survey", use_container_width=True):
        st.session_state.page = "create_survey"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank section
    st.markdown("**📚 Question Bank**")
    if st.button("📖 View Question Bank", use_container_width=True):
        st.session_state.page = "view_question_bank"
        st.rerun()
    if st.button("⭐ Unique Questions Bank", use_container_width=True):
        st.session_state.page = "unique_question_bank"
        st.rerun()
    if st.button("🔄 Update Question Bank", use_container_width=True):
        st.session_state.page = "update_question_bank"
        st.rerun()
    
    st.markdown("---")
    
   # Quick links
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")


# App UI with enhanced styling
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Snowflake + SurveyMonkey</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Home Page with Enhanced Dashboard
if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔄 Status", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            # Quick connection test
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
    
    st.markdown("---")
    
    # Quick actions grid
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 SurveyMonkey Operations")
        if st.button("👁️ View & Analyze Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
            st.rerun()
        if st.button("⚙️ Configure Survey with UIDs", use_container_width=True):
            st.session_state.page = "configure_survey"
            st.rerun()
        if st.button("➕ Create New Survey", use_container_width=True):
            st.session_state.page = "create_survey"
            st.rerun()
    
    with col2:
        st.markdown("### 📚 Question Bank Management")
        if st.button("📖 View Full Question Bank", use_container_width=True):
            st.session_state.page = "view_question_bank"
            st.rerun()
        if st.button("⭐ Unique Questions Bank", use_container_width=True):
            st.session_state.page = "unique_question_bank"
            st.rerun()
        if st.button("🔄 Update & Match Questions", use_container_width=True):
            st.session_state.page = "update_question_bank"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
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

# Unique Questions Bank Page
elif st.session_state.page == "unique_question_bank":
    st.markdown("## ⭐ Unique Questions Bank")
    st.markdown("*The best structured question for each UID, organized in ascending order*")
    
    try:
        with st.spinner("🔄 Loading question bank and creating unique questions..."):
            df_reference = run_snowflake_reference_query()
            unique_questions_df = create_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.markdown('<div class="warning-card">⚠️ No unique questions found in the database.</div>', unsafe_allow_html=True)
        else:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Unique UIDs", len(unique_questions_df))
            with col2:
                st.metric("📝 Total Variants", unique_questions_df['total_variants'].sum())
            with col3:
                avg_length = unique_questions_df['question_length'].mean()
                st.metric("📏 Avg Length", f"{avg_length:.0f} chars")
            with col4:
                avg_words = unique_questions_df['question_words'].mean()
                st.metric("📖 Avg Words", f"{avg_words:.0f}")
            
            st.markdown("---")
            
            # Search and filter options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            
            with col2:
                min_variants = st.selectbox("📊 Min variants", [1, 2, 3, 5, 10], index=0)
            
            # Filter the dataframe
            filtered_df = unique_questions_df.copy()
            
            if search_term:
                filtered_df = filtered_df[filtered_df['best_question'].str.contains(search_term, case=False, na=False)]
            
            filtered_df = filtered_df[filtered_df['total_variants'] >= min_variants]
            
            st.markdown(f"### 📋 Showing {len(filtered_df)} questions")
            
            # Display the unique questions
            if not filtered_df.empty:
                # Rename columns for better display
                display_df = filtered_df.copy()
                display_df = display_df.rename(columns={
                    'uid': 'UID',
                    'best_question': 'Best Question (English Format)',
                    'total_variants': 'Total Variants',
                    'question_length': 'Character Count',
                    'question_words': 'Word Count'
                })
                
                st.dataframe(
                    display_df,
                    column_config={
                        "UID": st.column_config.TextColumn("UID", width="small"),
                        "Best Question (English Format)": st.column_config.TextColumn("Best Question (English Format)", width="large"),
                        "Total Variants": st.column_config.NumberColumn("Total Variants", width="small"),
                        "Character Count": st.column_config.NumberColumn("Characters", width="small"),
                        "Word Count": st.column_config.NumberColumn("Words", width="small")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Download options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📥 Download Unique Questions (CSV)",
                        display_df.to_csv(index=False),
                        f"unique_questions_bank_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        "📥 Download Full Details (CSV)",
                        unique_questions_df.to_csv(index=False),
                        f"unique_questions_full_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ No questions match your current filters.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Unique questions bank failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked. Contact your Snowflake admin.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced View Surveys Page
elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ View Surveys on SurveyMonkey")
    st.markdown("*Browse and analyze your SurveyMonkey surveys*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">❌ SurveyMonkey token is missing in secrets configuration.</div>', unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("🔄 Fetching surveys from SurveyMonkey..."):
            surveys = get_surveys(token)
            
        if not surveys:
            st.markdown('<div class="warning-card">⚠️ No surveys found or invalid API response.</div>', unsafe_allow_html=True)
        else:
            # Display survey metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Surveys", len(surveys))
            with col2:
                recent_surveys = [s for s in surveys if s.get('date_created', '').startswith('2024') or s.get('date_created', '').startswith('2025')]
                st.metric("🆕 Recent (2024-2025)", len(recent_surveys))
            with col3:
                st.metric("🔄 Status", "Connected")
            
            st.markdown("---")
            
            # Survey selection interface
            choices = {s["title"]: s["id"] for s in surveys}
            survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
            survey_id_title_choices.sort(key=lambda x: int(x.split(" - ")[0]), reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_survey = st.selectbox("🎯 Choose Survey by Title", [""] + list(choices.keys()), index=0)
            with col2:
                selected_survey_ids = st.multiselect(
                    "📋 Select Multiple Surveys (ID/Title)",
                    survey_id_title_choices,
                    default=[],
                    help="Select one or more surveys by ID and title"
                )
            
            # Process selected surveys
            selected_survey_ids_from_title = []
            if selected_survey:
                selected_survey_ids_from_title.append(choices[selected_survey])
            
            all_selected_survey_ids = list(set(selected_survey_ids_from_title + [
                s.split(" - ")[0] for s in selected_survey_ids
            ]))
            
            if all_selected_survey_ids:
                combined_questions = []
                progress_bar = st.progress(0)
                
                for i, survey_id in enumerate(all_selected_survey_ids):
                    with st.spinner(f"🔄 Fetching survey questions for ID {survey_id}..."):
                        survey_json = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                    progress_bar.progress((i + 1) / len(all_selected_survey_ids))
                
                st.session_state.df_target = pd.DataFrame(combined_questions)
                
                if st.session_state.df_target.empty:
                    st.markdown('<div class="warning-card">⚠️ No questions found in the selected survey(s).</div>', unsafe_allow_html=True)
                else:
                    # Display analysis metrics
                    st.markdown("### 📊 Survey Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_questions = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                        st.metric("❓ Questions", total_questions)
                    with col2:
                        total_choices = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == True])
                        st.metric("📝 Choices", total_choices)
                    with col3:
                        headings_count = len(st.session_state.df_target[st.session_state.df_target["question_category"] == "Heading"])
                        st.metric("📋 Headings", headings_count)
                    with col4:
                        unique_surveys = st.session_state.df_target["survey_id"].nunique()
                        st.metric("📊 Surveys", unique_surveys)
                    
                    st.markdown("---")
                    
                    # Display options
                    col1, col2 = st.columns(2)
                    with col1:
                        show_main_only = st.checkbox("📋 Show only main questions", value=False)
                    with col2:
                        question_filter = st.selectbox("🔍 Filter by category", 
                                                     ["All", "Main Question/Multiple Choice", "Heading"])
                    
                    # Filter and display data
                    display_df = st.session_state.df_target.copy()
                    
                    if show_main_only:
                        display_df = display_df[display_df["is_choice"] == False]
                    
                    if question_filter != "All":
                        display_df = display_df[display_df["question_category"] == question_filter]
                    
                    display_df["survey_id_title"] = display_df.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                    
                    st.markdown(f"### 📋 Survey Questions ({len(display_df)} items)")
                    
                    st.dataframe(
                        display_df[["survey_id_title", "heading_0", "position", "is_choice", "parent_question", "schema_type", "question_category"]],
                        column_config={
                            "survey_id_title": st.column_config.TextColumn("Survey ID/Title", width="medium"),
                            "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                            "position": st.column_config.NumberColumn("Position", width="small"),
                            "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                            "parent_question": st.column_config.TextColumn("Parent Question", width="medium"),
                            "schema_type": st.column_config.TextColumn("Schema Type", width="small"),
                            "question_category": st.column_config.TextColumn("Category", width="small")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Download option
                    st.download_button(
                        "📥 Download Survey Data",
                        display_df.to_csv(index=False),
                        f"survey_data_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ Select a survey to view questions and analysis.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"SurveyMonkey processing failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced View Question Bank Page
elif st.session_state.page == "view_question_bank":
    st.markdown("## 📖 View Question Bank")
    st.markdown("*Complete question repository with UIDs in ascending order*")
    
    try:
        with st.spinner("🔄 Fetching Snowflake question bank..."):
            df_reference = run_snowflake_reference_query()
        
        if df_reference.empty:
            st.markdown('<div class="warning-card">⚠️ No data retrieved from Snowflake.</div>', unsafe_allow_html=True)
        else:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Questions", len(df_reference))
            with col2:
                unique_uids = df_reference['uid'].nunique()
                st.metric("🆔 Unique UIDs", unique_uids)
            with col3:
                avg_variants = len(df_reference) / unique_uids if unique_uids > 0 else 0
                st.metric("📝 Avg Variants/UID", f"{avg_variants:.1f}")
            
            st.markdown("---")
            
            # Search and filter
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            with col2:
                uid_filter = st.text_input("🆔 Filter by UID", placeholder="Enter UID...")
            
            # Apply filters
            filtered_df = df_reference.copy()
            
            if search_query:
                filtered_df = filtered_df[filtered_df['heading_0'].str.contains(search_query, case=False, na=False)]
            
            if uid_filter:
                filtered_df = filtered_df[filtered_df['uid'].astype(str).str.contains(uid_filter, case=False, na=False)]
            
            st.markdown(f"### 📋 Question Bank ({len(filtered_df)} questions)")
            
            st.dataframe(
                filtered_df,
                column_config={
                    "uid": st.column_config.TextColumn("UID", width="small"),
                    "heading_0": st.column_config.TextColumn("Question", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download option
            st.download_button(
                "📥 Download Question Bank",
                filtered_df.to_csv(index=False),
                f"question_bank_{uuid4()}.csv",
                "text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        logger.error(f"Snowflake processing failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked. Contact your Snowflake admin or wait 15–30 minutes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Configure Survey Page
elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Configure Survey from SurveyMonkey")
    st.markdown("*Match survey questions with UIDs and configure settings*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">❌ SurveyMonkey token is missing in secrets configuration.</div>', unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("🔄 Fetching surveys..."):
            surveys = get_surveys(token)
            
        if not surveys:
            st.markdown('<div class="warning-card">⚠️ No surveys found or invalid API response.</div>', unsafe_allow_html=True)
        else:
            choices = {s["title"]: s["id"] for s in surveys}
            survey_id_title_choices = [f"{s['id']} - {s['title']}" for s in surveys]
            survey_id_title_choices.sort(key=lambda x: int(x.split(" - ")[0]), reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_survey = st.selectbox("🎯 Choose Survey", [""] + list(choices.keys()), index=0)
            with col2:
                selected_survey_ids = st.multiselect(
                    "📋 SurveyID/Title",
                    survey_id_title_choices,
                    default=[],
                    help="Select one or more surveys by ID and title"
                )
            
            selected_survey_ids_from_title = []
            if selected_survey:
                selected_survey_ids_from_title.append(choices[selected_survey])
            
            all_selected_survey_ids = list(set(selected_survey_ids_from_title + [
                s.split(" - ")[0] for s in selected_survey_ids
            ]))
            
            if all_selected_survey_ids:
                combined_questions = []
                progress_bar = st.progress(0)
                
                for i, survey_id in enumerate(all_selected_survey_ids):
                    with st.spinner(f"🔄 Processing survey {survey_id}..."):
                        survey_json = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                    progress_bar.progress((i + 1) / len(all_selected_survey_ids))
            
                st.session_state.df_target = pd.DataFrame(combined_questions)
                
                if st.session_state.df_target.empty:
                    st.markdown('<div class="warning-card">⚠️ No questions found in the selected survey(s).</div>', unsafe_allow_html=True)
                else:
                    # Run UID matching
                    try:
                        with st.spinner("🔄 Matching questions to UIDs..."):
                            st.session_state.df_reference = run_snowflake_reference_query()
                            st.session_state.df_final = run_uid_match(st.session_state.df_reference, st.session_state.df_target)
                            st.session_state.uid_changes = {}
                            
                        # Display matching results
                        matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                        
                        st.markdown("### 📊 Configuration Results")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Match Rate", f"{matched_percentage}%")
                        with col2:
                            total_q = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                            st.metric("❓ Questions", total_q)
                        with col3:
                            total_c = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == True])
                            st.metric("📝 Choices", total_c)
                        with col4:
                            st.metric("🔄 Status", "✅ Processed")
                            
                    except Exception as e:
                        logger.error(f"UID matching failed: {e}")
                        if "250001" in str(e) or "invalid identifier" in str(e).lower():
                            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: Account locked or schema incorrect. UID matching disabled but editing available.</div>', unsafe_allow_html=True)
                            st.session_state.df_reference = None
                            st.session_state.df_final = st.session_state.df_target.copy()
                            st.session_state.df_final["Final_UID"] = None
                            st.session_state.df_final["configured_final_UID"] = None
                            st.session_state.df_final["Change_UID"] = None
                            st.session_state.df_final["survey_id_title"] = st.session_state.df_final.apply(
                                lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                                axis=1
                            )
                            st.session_state.uid_changes = {}
                        else:
                            st.markdown(f'<div class="warning-card">❌ UID matching failed: {e}</div>', unsafe_allow_html=True)
                            raise
                    
                    # Display configuration interface
                    st.markdown("---")
                    st.markdown("### ⚙️ Configure Questions & Download")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_main_only = st.checkbox("📋 Show only main questions", value=True)
                    with col2:
                        search_query = st.text_input("🔍 Search questions", placeholder="Type to filter...")
                    
                    # Filter and display results
                    display_df = st.session_state.df_final.copy()
                    
                    if show_main_only:
                        display_df = display_df[display_df["is_choice"] == False]
                    
                    if search_query:
                        display_df = display_df[display_df["heading_0"].str.contains(search_query, case=False, na=False)]
                    
                    # Add survey ID/title column
                    display_df["survey_id_title"] = display_df.apply(
                        lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                        axis=1
                    )
                    
                    st.markdown(f"### 📋 Survey Configuration ({len(display_df)} items)")
                    
                    # Display configuration table
                    config_columns = ["survey_id_title", "heading_0", "position", "is_choice", "schema_type", "configured_final_UID", "question_category"]
                    config_columns = [col for col in config_columns if col in display_df.columns]
                    
                    st.dataframe(
                        display_df[config_columns],
                        column_config={
                            "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
                            "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                            "position": st.column_config.NumberColumn("Position", width="small"),
                            "is_choice": st.column_config.CheckboxColumn("Choice", width="small"),
                            "schema_type": st.column_config.TextColumn("Type", width="small"),
                            "configured_final_UID": st.column_config.TextColumn("UID", width="small"),
                            "question_category": st.column_config.TextColumn("Category", width="small")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Export section
                    st.markdown("#### 📤 Export & Upload Options")
                    
                    # Prepare export data
                    export_columns = [
                        "survey_id", "survey_title", "heading_0", "configured_final_UID", "position",
                        "is_choice", "parent_question", "question_uid", "schema_type", "mandatory",
                        "mandatory_editable", "question_category"
                    ]
                    export_columns = [col for col in export_columns if col in st.session_state.df_final.columns]
                    export_df = st.session_state.df_final[export_columns].copy()
                    export_df = export_df.rename(columns={"configured_final_UID": "uid"})
                    
                    # Download and upload buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📥 Download Configuration (CSV)",
                            export_df.to_csv(index=False),
                            f"survey_configuration_{uuid4()}.csv",
                            "text/csv",
                            help="Download the complete survey configuration",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("🚀 Upload to Snowflake", use_container_width=True, type="primary"):
                            try:
                                with st.spinner("🔄 Uploading to Snowflake..."):
                                    with get_snowflake_engine().connect() as conn:
                                        export_df.to_sql(
                                            'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                                            conn,
                                            schema='DBT_SURVEY_MONKEY',
                                            if_exists='append',
                                            index=False
                                        )
                                    st.markdown('<div class="success-card">🎉 Successfully uploaded to Snowflake!</div>', unsafe_allow_html=True)
                                    st.balloons()
                            except Exception as e:
                                logger.error(f"Snowflake upload failed: {e}")
                                if "250001" in str(e):
                                    st.markdown('<div class="warning-card">🔒 Snowflake upload failed: User account is locked. Contact your Snowflake admin.</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="warning-card">❌ Snowflake upload failed: {e}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">ℹ️ Select a survey to start configuration.</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"SurveyMonkey processing failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Update Question Bank Page
elif st.session_state.page == "update_question_bank":
    st.markdown("## 🔄 Update Question Bank")
    st.markdown("*Match new questions with existing UIDs and update the database*")
    
    try:
        with st.spinner("🔄 Fetching Snowflake data..."):
            df_reference = run_snowflake_reference_query()
            df_target = run_snowflake_target_query()
        
        if df_reference.empty or df_target.empty:
            st.markdown('<div class="warning-card">⚠️ No data retrieved from Snowflake for matching.</div>', unsafe_allow_html=True)
        else:
            # Display initial metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Reference Questions", len(df_reference))
            with col2:
                st.metric("🎯 Target Questions", len(df_target))
            with col3:
                st.metric("🔄 Status", "Ready to Match")
            
            st.markdown("---")
            
            with st.spinner("🤖 Running UID matching algorithm..."):
                df_final = run_uid_match(df_reference, df_target)
            
            # Enhanced results display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                high_conf = len(df_final[df_final.get("Final_Match_Type", "") == "✅ High"])
                st.metric("✅ High Confidence", high_conf)
            with col2:
                low_conf = len(df_final[df_final.get("Final_Match_Type", "") == "⚠️ Low"]) 
                st.metric("⚠️ Low Confidence", low_conf)
            with col3:
                semantic = len(df_final[df_final.get("Final_Match_Type", "") == "🧠 Semantic"])
                st.metric("🧠 Semantic", semantic)
            with col4:
                no_match = len(df_final[df_final.get("Final_Match_Type", "") == "❌ No match"])
                st.metric("❌ No Match", no_match)
            
            st.markdown("---")
            
            # Filter controls
            st.markdown("### 🎛️ Filter Results")
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_filter = st.multiselect(
                    "🎯 Filter by Match Type",
                    ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                    default=["✅ High", "⚠️ Low", "🧠 Semantic"]
                )
            
            with col2:
                min_similarity = st.slider("📊 Minimum Similarity Score", 0.0, 1.0, 0.5, 0.05)
            
            # Apply filters
            filtered_df = df_final[df_final.get("Final_Match_Type", "").isin(confidence_filter)]
            if "Similarity" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Similarity"] >= min_similarity]
            
            st.markdown(f"### 📋 Matching Results ({len(filtered_df)} items)")
            
            # Enhanced display
            display_columns = ["heading_0", "Final_UID", "Final_Match_Type", "Similarity"]
            if "Semantic_Similarity" in filtered_df.columns:
                display_columns.append("Semantic_Similarity")
            if "Matched_Question" in filtered_df.columns:
                display_columns.append("Matched_Question")
            
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            display_df = filtered_df[available_columns].copy()
            display_df = display_df.rename(columns={
                "heading_0": "Target Question",
                "Final_UID": "Matched UID",
                "Final_Match_Type": "Match Type",
                "Similarity": "TF-IDF Score",
                "Semantic_Similarity": "Semantic Score",
                "Matched_Question": "Reference Question"
            })
            
            st.dataframe(
                display_df,
                column_config={
                    "Target Question": st.column_config.TextColumn("Target Question", width="large"),
                    "Matched UID": st.column_config.TextColumn("UID", width="small"),
                    "Match Type": st.column_config.TextColumn("Match Type", width="small"),
                    "TF-IDF Score": st.column_config.NumberColumn("TF-IDF", format="%.3f", width="small"),
                    "Semantic Score": st.column_config.NumberColumn("Semantic", format="%.3f", width="small"),
                    "Reference Question": st.column_config.TextColumn("Reference Question", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download option
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download All Results",
                    df_final.to_csv(index=False),
                    f"uid_matching_results_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📥 Download Filtered Results", 
                    filtered_df.to_csv(index=False),
                    f"uid_matches_filtered_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
    except Exception as e:
        logger.error(f"Question bank update failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked. Contact your Snowflake admin or wait 15–30 minutes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Create New Survey Page
elif st.session_state.page == "create_survey":
    st.markdown("## ➕ Create New Survey")
    st.markdown("*Build and deploy a new survey directly to SurveyMonkey*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">❌ SurveyMonkey token is missing in secrets configuration.</div>', unsafe_allow_html=True)
            st.stop()
        
        st.markdown("### 🎯 Survey Template Builder")
        
        with st.form("survey_template_form"):
            # Basic survey settings
            col1, col2 = st.columns(2)
            with col1:
                survey_title = st.text_input("📝 Survey Title", value="New Survey")
                survey_language = st.selectbox("🌐 Language", ["en", "es", "fr", "de"], index=0)
            with col2:
                num_pages = st.number_input("📄 Number of Pages", min_value=1, max_value=10, value=1)
                survey_theme = st.selectbox("🎨 Theme", ["Default", "Professional", "Modern"], index=0)
            
            # Survey settings
            st.markdown("#### ⚙️ Survey Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_progress_bar = st.checkbox("📊 Show Progress Bar", value=True)
            with col2:
                hide_asterisks = st.checkbox("⭐ Hide Required Asterisks", value=False)
            with col3:
                one_question_at_a_time = st.checkbox("1️⃣ One Question Per Page", value=False)
            
            # Pages and questions builder
            pages = []
            for i in range(num_pages):
                st.markdown(f"### 📄 Page {i+1}")
                
                col1, col2 = st.columns(2)
                with col1:
                    page_title = st.text_input(f"Page Title", value=f"Page {i+1}", key=f"page_title_{i}")
                with col2:
                    num_questions = st.number_input(
                        f"Questions on Page",
                        min_value=1,
                        max_value=10,
                        value=2,
                        key=f"num_questions_{i}"
                    )
                
                page_description = st.text_area(f"Page Description", value="", key=f"page_desc_{i}")
                
                questions = []
                for j in range(num_questions):
                    with st.expander(f"❓ Question {j+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            question_text = st.text_input("Question Text", value="", key=f"q_text_{i}_{j}")
                            question_type = st.selectbox(
                                "Question Type",
                                ["Single Choice", "Multiple Choice", "Open-Ended"],
                                key=f"q_type_{i}_{j}"
                            )
                        with col2:
                            is_required = st.checkbox("Required", key=f"q_required_{i}_{j}")
                            question_position = st.number_input("Position", min_value=1, value=j+1, key=f"q_pos_{i}_{j}")
                        
                        question_template = {
                            "heading": question_text,
                            "position": question_position,
                            "is_required": is_required
                        }
                        
                        if question_type == "Single Choice":
                            question_template["family"] = "single_choice"
                            question_template["subtype"] = "vertical"
                            num_choices = st.number_input(
                                "Number of Choices",
                                min_value=2,
                                max_value=10,
                                value=3,
                                key=f"num_choices_{i}_{j}"
                            )
                            choices = []
                            for k in range(num_choices):
                                choice_text = st.text_input(
                                    f"Choice {k+1}",
                                    value="",
                                    key=f"choice_{i}_{j}_{k}"
                                )
                                if choice_text:
                                    choices.append({"text": choice_text, "position": k + 1})
                            if choices:
                                question_template["choices"] = choices
                        
                        elif question_type == "Multiple Choice":
                            question_template["family"] = "multiple_choice"
                            question_template["subtype"] = "vertical"
                            num_choices = st.number_input(
                                "Number of Choices",
                                min_value=2,
                                max_value=10,
                                value=4,
                                key=f"num_choices_{i}_{j}"
                            )
                            choices = []
                            for k in range(num_choices):
                                choice_text = st.text_input(
                                    f"Choice {k+1}",
                                    value="",
                                    key=f"choice_{i}_{j}_{k}"
                                )
                                if choice_text:
                                    choices.append({"text": choice_text, "position": k + 1})
                            if choices:
                                question_template["choices"] = choices
                        
                        elif question_type == "Open-Ended":
                            question_template["family"] = "open_ended"
                            question_template["subtype"] = "essay"
                        
                        if question_text:
                            questions.append(question_template)
                
                if questions:
                    pages.append({
                        "title": page_title,
                        "description": page_description,
                        "questions": questions
                    })
            
            # Survey template compilation
            survey_template = {
                "title": survey_title,
                "language": survey_language,
                "pages": pages,
                "settings": {
                    "progress_bar": show_progress_bar,
                    "hide_asterisks": hide_asterisks,
                    "one_question_at_a_time": one_question_at_a_time
                },
                "theme": {
                    "name": survey_theme.lower(),
                    "font": "Arial",
                    "background_color": "#FFFFFF",
                    "question_color": "#000000",
                    "answer_color": "#000000"
                }
            }
            
            submit = st.form_submit_button("🚀 Create Survey", type="primary", use_container_width=True)
            
            if submit:
                if not survey_title or not pages:
                    st.markdown('<div class="warning-card">⚠️ Survey title and at least one page with questions are required.</div>', unsafe_allow_html=True)
                else:
                    st.session_state.survey_template = survey_template
                    
                    try:
                        with st.spinner("🔄 Creating survey in SurveyMonkey..."):
                            # Create survey
                            survey_id = create_survey(token, survey_template)
                            
                            # Create pages and questions
                            for page_template in survey_template["pages"]:
                                page_id = create_page(token, survey_id, page_template)
                                for question_template in page_template["questions"]:
                                    create_question(token, survey_id, page_id, question_template)
                            
                            st.markdown(f'<div class="success-card">🎉 Survey created successfully!<br>Survey ID: <strong>{survey_id}</strong></div>', unsafe_allow_html=True)
                            st.balloons()
                            
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Failed to create survey: {e}</div>', unsafe_allow_html=True)
        
        # Preview section
        if st.session_state.survey_template:
            st.markdown("---")
            st.markdown("### 👀 Survey Template Preview")
            
            with st.expander("🔍 View JSON Template"):
                st.json(st.session_state.survey_template)
            
            # Summary display
            template = st.session_state.survey_template
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Pages", len(template.get("pages", [])))
            with col2:
                total_questions = sum(len(page.get("questions", [])) for page in template.get("pages", []))
                st.metric("❓ Questions", total_questions)
            with col3:
                st.metric("🌐 Language", template.get("language", "en").upper())
            with col4:
                st.metric("📊 Progress Bar", "✅" if template.get("settings", {}).get("progress_bar") else "❌")
            
            # Download template
            st.download_button(
                "📥 Download Template",
                json.dumps(template, indent=2),
                f"survey_template_{uuid4()}.json",
                "application/json",
                use_container_width=True
            )
        
    except Exception as e:
        logger.error(f"Survey creation failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Navigation footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Return to Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

with col2:
    st.markdown("*Built with ❤️ using Streamlit*")

with col3:
    st.markdown(f"**Current Page:** {st.session_state.page.replace('_', ' ').title()}")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)import streamlit as st
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

# Synonym Mapping
DEFAULT_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

# Reference Heading Texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda."
]

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

# Normalization
def enhanced_normalize(text, synonym_map=DEFAULT_SYNONYM_MAP):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def get_best_question_for_uid(questions_list):
    """
    Select the best structured question from a list of questions with the same UID.
    Prioritizes English format and better structure.
    """
    if not questions_list:
        return None
    
    # Score each question based on quality indicators
    def score_question(question):
        score = 0
        text = str(question).lower()
        
        # Prefer questions that are complete sentences
        if text.endswith('?'):
            score += 10
        
        # Prefer questions with proper capitalization (indicates better formatting)
        if any(c.isupper() for c in question):
            score += 5
        
        # Prefer longer, more descriptive questions
        word_count = len(text.split())
        if 5 <= word_count <= 20:
            score += 8
        elif word_count > 20:
            score += 3
        
        # Avoid very short or incomplete questions
        if word_count < 3:
            score -= 10
        
        # Prefer questions without HTML tags or special formatting
        if '<' not in text and '>' not in text:
            score += 5
        
        # Prefer questions that don't contain common formatting artifacts
        artifacts = ['click here', 'please select', '...', 'n/a', 'other']
        if not any(artifact in text for artifact in artifacts):
            score += 3
        
        # Prefer questions that look like proper English
        english_indicators = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were']
        if any(indicator in text for indicator in english_indicators):
            score += 7
        
        return score
    
    # Score all questions and return the best one
    scored_questions = [(q, score_question(q)) for q in questions_list]
    best_question = max(scored_questions, key=lambda x: x[1])
    return best_question[0]

def create_unique_questions_bank(df_reference):
    """
    Create a unique questions bank with the best question for each UID.
    """
    if df_reference.empty:
        return pd.DataFrame()
    
    # Group by UID and get the best question for each
    unique_questions = []
    
    for uid in df_reference['uid'].unique():
        if pd.isna(uid):
            continue
            
        uid_questions = df_reference[df_reference['uid'] == uid]['heading_0'].tolist()
        best_question = get_best_question_for_uid(uid_questions)
        
        if best_question:
            unique_questions.append({
                'uid': uid,
                'best_question': best_question,
                'total_variants': len(uid_questions),
                'question_length': len(str(best_question)),
                'question_words': len(str(best_question).split())
            })
    
    unique_df = pd.DataFrame(unique_questions)
    
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

# Snowflake Queries
def run_snowflake_reference_query(limit=10000, offset=0):
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
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning(
                "🔒 Cannot fetch Snowflake data: User account is locked. "
                "UID matching is disabled. Please resolve the lockout and retry."
            )
        elif "invalid identifier" in str(e).lower():
            st.warning(
                "⚠️ Snowflake query failed due to invalid column. "
                "UID matching is disabled, but you can edit questions, search, and use Google Forms. "
                "Contact your Snowflake admin to verify table schema."
            )
        raise

def run_snowflake_target_query():
    query = """
        SELECT DISTINCT HEADING_0
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
                "🔒 Cannot fetch Snowflake data: User account is locked. "
                "Please resolve the lockout and retry."
            )
        raise

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

# UID Matching functions
def compute_tfidf_matches(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP):
    df_reference = df_reference[df_reference["heading_0"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["heading_0"].notna()].reset_index(drop=True)
    df_reference["norm_text"] = df_reference["heading_0"].apply(enhanced_normalize)
    df_target["norm_text"] = df_target["heading_0"].apply(enhanced_normalize)

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs = [], [], [], []
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
        matched_uids.append(df_reference.iloc[best_idx]["uid"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["heading_0"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    try:
        model = load_sentence_transformer()
        emb_target = model.encode(df_target["heading_0"].tolist(), convert_to_tensor=True)
        emb_ref = model.encode(df_reference["heading_0"].tolist(), convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb_target, emb_ref)

        sem_matches, sem_scores = [], []
        for i in range(len(df_target)):
            best_idx = cosine_scores[i].argmax().item()
            score = cosine_scores[i][best_idx].item()
            sem_matches.append(df_reference.iloc[best_idx]["uid"] if score >= SEMANTIC_THRESHOLD else None)
            sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

        df_target["Semantic_UID"] = sem_matches
        df_target["Semantic_Similarity"] = sem_scores
        return df_target
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}")
        st.error(f"🚨 Semantic matching failed: {e}")
        return df_target

def assign_match_type(row):
    if pd.notnull(row["Suggested_UID"]):
        return row["Match_Confidence"]
    return "🧠 Semantic" if pd.notnull(row["Semantic_UID"]) else "❌ No match"

def finalize_matches(df_target, df_reference):
    df_target["Final_UID"] = df_target["Suggested_UID"].combine_first(df_target["Semantic_UID"])
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target["Matched_Question"]
    df_target["Final_Match_Type"] = df_target.apply(assign_match_type, axis=1)
    
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
    
    return df_target

def detect_uid_conflicts(df_target):
    uid_conflicts = df_target.groupby("Final_UID")["heading_0"].nunique()
    duplicate_uids = uid_conflicts[uid_conflicts > 1].index
    df_target["UID_Conflict"] = df_target["Final_UID"].apply(
        lambda x: "⚠️ Conflict" if pd.notnull(x) and x in duplicate_uids else ""
    )
    return df_target

def run_uid_match(df_reference, df_target, synonym_map=DEFAULT_SYNONYM_MAP, batch_size=BATCH_SIZE):
    if df_reference.empty or df_target.empty:
        logger.warning("Empty input dataframes provided.")
        st.error("🚨 Input data is empty.")
        return pd.DataFrame()

    if len(df_target) > 10000:
        st.warning("⚠️ Large dataset detected. Processing may take time.")

    logger.info(f"Processing {len(df_target)} target questions against {len(df_reference)} reference questions.")
    df_results = []
    for start in range(0, len(df_target), batch_size):
        batch_target = df_target.iloc[start:start + batch_size].copy()
        with st.spinner(f"🔄 Processing batch {start//batch_size + 1}..."):
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
    st.markdown("### 🧠 UID Matcher Pro")
    st.markdown("Navigate through the application")
    
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
    if st.button("⚙️ Configure Survey", use_container_width=True):
        st.session_state.page = "configure_survey"
        st.rerun()
    if st.button("➕ Create New Survey", use_container_width=True):
        st.session_state.page = "create_survey"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank section
    st.markdown("**📚 Question Bank**")
    if st.button("📖 View Question Bank", use_container_width=True):
        st.session_state.page = "view_question_bank"
        st.rerun()
    if st.button("⭐ Unique Questions Bank", use_container_width=True):
        st.session_state.page = "unique_question_bank"
        st.rerun()
    if st.button("🔄 Update Question Bank", use_container_width=True):
        st.session_state.page = "update_question_bank"
        st.rerun()
    
    st.markdown("---")
    
    # Quick links
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")