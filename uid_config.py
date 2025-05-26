import streamlit as st
import pandas as pd
import requests
import json
import logging
import uuid
from datetime import datetime
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

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
            with get_snowflake_engine().connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE WHERE UID IS NOT NULL"))
                count = result.fetchone()[0]
                st.metric("📊 Total UIDs", f"{count:,}")
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
    
    with status_col2:
        try:
            token = st.secrets.get("surveymonkey", {}).get("token", None)
            if token:
                get_surveys(token)
                st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">❌ SurveyMonkey: No Token</div>', unsafe_allow_html=True)
        except Exception as e:
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
            
            col1, col2 = st.columns([2, 1])
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            with col2:
                min_variants = st.selectbox("📊 Min variants", [1, 2, 3, 5, 10], index=0)
            
            filtered_df = unique_questions_df.copy()
            if search_term:
                filtered_df = filtered_df[filtered_df['best_question'].str.contains(search_term, case=False, na=False)]
            filtered_df = filtered_df[filtered_df['total_variants'] >= min_variants]
            
            st.markdown(f"### 📋 Showing {len(filtered_df)} questions")
            
            if not filtered_df.empty:
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

# View Surveys Page
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Surveys", len(surveys))
            with col2:
                recent_surveys = [s for s in surveys if s.get('date_created', '').startswith('2024') or s.get('date_created', '').startswith('2025')]
                st.metric("🆕 Recent (2024-2025)", len(recent_surveys))
            with col3:
                st.metric("🔄 Status", "Connected")
            
            st.markdown("---")
            
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_main_only = st.checkbox("📋 Show only main questions", value=False)
                    with col2:
                        question_filter = st.selectbox("🔍 Filter by category", 
                                                     ["All", "Main Question/Multiple Choice", "Heading"])
                    
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

# Configure Survey Page
elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Configure Survey with UIDs")
    st.markdown("*Match survey questions to existing UIDs and customize assignments*")
    
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
                    with st.spinner("🤖 Running UID matching algorithm..."):
                        df_reference = run_snowflake_reference_query()
                        st.session_state.df_final = run_uid_match(df_reference, st.session_state.df_target)
                    
                    matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                    
                    st.markdown("### 📊 Matching Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_questions = len(st.session_state.df_final[st.session_state.df_final["is_choice"] == False])
                        st.metric("❓ Questions", total_questions)
                    with col2:
                        matched_questions = len(st.session_state.df_final[
                            (st.session_state.df_final["is_choice"] == False) & 
                            (st.session_state.df_final["Final_UID"].notna())
                        ])
                        st.metric("✅ Matched", matched_questions)
                    with col3:
                        st.metric("📈 Match Rate", f"{matched_percentage}%")
                    with col4:
                        conflicts = len(st.session_state.df_final[st.session_state.df_final["UID_Conflict"] == "⚠️ Conflict"])
                        st.metric("⚠️ Conflicts", conflicts)
                    
                    st.markdown("---")
                    
                    st.markdown("### 🎛️ Customization")
                    col1, col2 = st.columns(2)
                    with col1:
                        config_filter = st.selectbox(
                            "🔍 Filter Display",
                            ["All Questions", "Main Questions Only", "Mandatory Only", "With UIDs Only"],
                            index=0
                        )
                    with col2:
                        search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
                    
                    config_columns = [
                        "survey_id_title", "heading_0", "position", "is_choice", "parent_question",
                        "schema_type", "mandatory", "configured_final_UID", "question_category"
                    ]
                    config_columns = [col for col in config_columns if col in st.session_state.df_final.columns]
                    config_df = st.session_state.df_final[config_columns].copy()
                    
                    if search_term:
                        config_df = config_df[config_df["heading_0"].str.contains(search_term, case=False, na=False)]
                    
                    if config_filter == "Main Questions Only":
                        config_df = config_df[config_df["is_choice"] == False]
                    elif config_filter == "Mandatory Only":
                        config_df = config_df[config_df["mandatory"] == True]
                    elif config_filter == "With UIDs Only":
                        config_df = config_df[config_df["configured_final_UID"].notna()]
                    
                    config_df = config_df.rename(columns={
                        "heading_0": "Question/Choice",
                        "configured_final_UID": "Assigned UID"
                    })
                    
                    st.dataframe(
                        config_df,
                        column_config={
                            "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
                            "Question/Choice": st.column_config.TextColumn("Question/Choice", width="large"),
                            "position": st.column_config.NumberColumn("Position", width="small"),
                            "is_choice": st.column_config.CheckboxColumn("Choice", width="small"),
                            "parent_question": st.column_config.TextColumn("Parent", width="medium"),
                            "schema_type": st.column_config.TextColumn("Type", width="small"),
                            "mandatory": st.column_config.CheckboxColumn("Required", width="small"),
                            "Assigned UID": st.column_config.TextColumn("UID", width="small"),
                            "question_category": st.column_config.TextColumn("Category", width="small")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    st.markdown("#### 📤 Export & Upload Options")
                    export_columns = [
                        "survey_id", "survey_title", "heading_0", "configured_final_UID", "position",
                        "is_choice", "parent_question", "question_uid", "schema_type", "mandatory",
                        "mandatory_editable", "question_category"
                    ]
                    export_columns = [col for col in export_columns if col in st.session_state.df_final.columns]
                    export_df = st.session_state.df_final[export_columns].copy()
                    export_df = export_df.rename(columns={"configured_final_UID": "uid"})
                    
                    st.markdown("##### 👀 Snowflake Upload Preview")
                    preview_df = export_df.copy()
                    main_questions_df = preview_df[preview_df["is_choice"] == False].copy()
                    preview_df["Main_Question_UID"] = preview_df.apply(
                        lambda row: main_questions_df[main_questions_df["heading_0"] == row["parent_question"]]["uid"].iloc[0]
                        if row["is_choice"] and pd.notnull(row["parent_question"]) and not main_questions_df[main_questions_df["heading_0"] == row["parent_question"]].empty
                        else row["uid"],
                        axis=1
                    )
                    preview_df["Main_Question_Position"] = preview_df.apply(
                        lambda row: main_questions_df[main_questions_df["heading_0"] == row["parent_question"]]["position"].iloc[0]
                        if row["is_choice"] and pd.notnull(row["parent_question"]) and not main_questions_df[main_questions_df["heading_0"] == row["parent_question"]].empty
                        else row["position"],
                        axis=1
                    )
                    
                    preview_display_df = preview_df[["survey_id", "survey_title", "heading_0", "Main_Question_Position", "Main_Question_UID"]].head(10)
                    preview_display_df = preview_display_df.rename(columns={
                        "survey_id": "SurveyID",
                        "survey_title": "SurveyName", 
                        "heading_0": "Question Info",
                        "Main_Question_Position": "Position",
                        "Main_Question_UID": "UID"
                    })
                    
                    st.dataframe(preview_display_df, hide_index=True, use_container_width=True)
                    st.caption(f"Showing first 10 rows of {len(preview_df)} total records")
                    
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

# Create New Survey Page
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
            col1, col2 = st.columns(2)
            with col1:
                survey_title = st.text_input("📝 Survey Title", value="New Survey")
                survey_language = st.selectbox("🌐 Language", ["en", "es", "fr", "de"], index=0)
            with col2:
                num_pages = st.number_input("📄 Number of Pages", min_value=1, max_value=10, value=1)
                survey_theme = st.selectbox("🎨 Theme", ["Default", "Professional", "Modern"], index=0)
            
            st.markdown("#### ⚙️ Survey Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_progress_bar = st.checkbox("📊 Show Progress Bar", value=True)
            with col2:
                hide_asterisks = st.checkbox("⭐ Hide Required Asterisks", value=False)
            with col3:
                one_question_at_a_time = st.checkbox("1️⃣ One Question Per Page", value=False)
            
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
                                ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
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
                            choice_cols = st.columns(min(num_choices, 3))
                            for k in range(num_choices):
                                col_idx = k % len(choice_cols)
                                with choice_cols[col_idx]:
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
                            choice_cols = st.columns(min(num_choices, 3))
                            for k in range(num_choices):
                                col_idx = k % len(choice_cols)
                                with choice_cols[col_idx]:
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
                            subtype_options = ["essay", "single", "numerical"]
                            question_template["subtype"] = st.selectbox(
                                "Open-Ended Type",
                                subtype_options,
                                key=f"oe_type_{i}_{j}"
                            )
                        
                        elif question_type == "Matrix":
                            question_template["family"] = "matrix"
                            question_template["subtype"] = "rating"
                            col1, col2 = st.columns(2)
                            with col1:
                                num_rows = st.number_input(
                                    "Number of Rows",
                                    min_value=2,
                                    max_value=10,
                                    value=3,
                                    key=f"num_rows_{i}_{j}"
                                )
                                rows = []
                                for k in range(num_rows):
                                    row_text = st.text_input(
                                        f"Row {k+1}",
                                        value="",
                                        key=f"row_{i}_{j}_{k}"
                                    )
                                    if row_text:
                                        rows.append({"text": row_text, "position": k + 1})
                            with col2:
                                num_matrix_choices = st.number_input(
                                    "Number of Rating Options",
                                    min_value=2,
                                    max_value=10,
                                    value=5,
                                    key=f"num_matrix_choices_{i}_{j}"
                                )
                                matrix_choices = []
                                for k in range(num_matrix_choices):
                                    choice_text = st.text_input(
                                        f"Rating {k+1}",
                                        value="",
                                        key=f"rating_{i}_{j}_{k}"
                                    )
                                    if choice_text:
                                        matrix_choices.append({"text": choice_text, "position": k + 1})
                            if rows and matrix_choices:
                                question_template["rows"] = rows
                                question_template["choices"] = matrix_choices
                        
                        if question_text:
                            questions.append(question_template)
                
                if questions:
                    pages.append({
                        "title": page_title,
                        "description": page_description,
                        "questions": questions
                    })
            
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
                            survey_id = create_survey(token, survey_template)
                            for page_template in survey_template["pages"]:
                                page_id = create_page(token, survey_id, page_template)
                                for question_template in page_template["questions"]:
                                    create_question(token, survey_id, page_id, question_template)
                            st.markdown(f'<div class="success-card">🎉 Survey created successfully!<br>Survey ID: <strong>{survey_id}</strong></div>', unsafe_allow_html=True)
                            st.balloons()
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Failed to create survey: {e}</div>', unsafe_allow_html=True)
        
        if st.session_state.survey_template:
            st.markdown("---")
            st.markdown("### 👀 Survey Template Preview")
            with st.expander("🔍 View JSON Template"):
                st.json(st.session_state.survey_template)
            
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

# Update Question Bank Page
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
            
            filtered_df = df_final[df_final.get("Final_Match_Type", "").isin(confidence_filter)]
            if "Similarity" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Similarity"] >= min_similarity]
            
            st.markdown(f"### 📋 Matching Results ({len(filtered_df)} items)")
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
