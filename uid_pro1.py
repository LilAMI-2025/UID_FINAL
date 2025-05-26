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

