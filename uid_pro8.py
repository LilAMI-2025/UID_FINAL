"is_choice": st.column_config.CheckboxColumn("Choice", width="small"),
                                "schema_type": st.column_config.TextColumn("Type", width="small"),
                                "configured_final_UID": st.column_config.TextColumn("UID", width="small"),
                                "Final_Match_Type": st.column_config.TextColumn("Match Type", width="small"),
                                "Final_Governance": st.column_config.TextColumn("Governance", width="small"),
                                "question_category": st.column_config.TextColumn("Q Category", width="small"),
                                "survey_category": st.column_config.TextColumn("S Category", width="small")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown("---")
                        
                        # Enhanced export section
                        st.markdown("#### 📤 Enhanced Export & Upload Options")
                        
                        # Prepare enhanced export data
                        export_columns = [
                            "survey_id", "survey_title", "heading_0", "configured_final_UID", "position",
                            "is_choice", "parent_question", "question_uid", "schema_type", "mandatory",
                            "mandatory_editable", "question_category", "survey_category", "Final_Match_Type",
                            "Final_Governance", "Similarity", "Semantic_Similarity"
                        ]
                        export_columns = [col for col in export_columns if col in st.session_state.df_final.columns]
                        export_df = st.session_state.df_final[export_columns].copy()
                        export_df = export_df.rename(columns={"configured_final_UID": "uid"})
                        
                        # Generate governance report
                        governance_report = st.session_state.df_final[st.session_state.df_final.get("Final_Governance", "✅") == "⚠️"].copy()
                        
                        # Download and upload buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                "📥 Download Enhanced Configuration",
                                export_df.to_csv(index=False),
                                f"enhanced_survey_config_{uuid4()}.csv",
                                "text/csv",
                                help="Download complete configuration with governance and matching details",
                                use_container_width=True
                            )
                        
                        with col2:
                            if not governance_violations.empty:
                                st.download_button(
                                    "⚖️ Download Governance Report",
                                    governance_report.to_csv(index=False),
                                    f"governance_violations_{uuid4()}.csv",
                                    "text/csv",
                                    help="Download questions violating governance rules",
                                    use_container_width=True
                                )
                            else:
                                st.success("✅ No governance violations")
                        
                        with col3:
                            if st.button("🚀 Upload to Snowflake", use_container_width=True, type="primary"):
                                try:
                                    with st.spinner("🔄 Uploading enhanced configuration to Snowflake..."):
                                        # Add governance compliance check before upload
                                        if enforce_governance and not governance_violations.empty:
                                            st.error("❌ Cannot upload: Governance violations detected. Please resolve conflicts first.")
                                        else:
                                            with get_snowflake_engine().connect() as conn:
                                                export_df.to_sql(
                                                    'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE',
                                                    conn,
                                                    schema='DBT_SURVEY_MONKEY',
                                                    if_exists='append',
                                                    index=False
                                                )
                                            st.markdown('<div class="success-card">🎉 Successfully uploaded enhanced configuration to Snowflake!</div>', unsafe_allow_html=True)
                                            st.balloons()
                                except Exception as e:
                                    logger.error(f"Snowflake upload failed: {e}")
                                    if "250001" in str(e):
                                        st.markdown('<div class="warning-card">🔒 Snowflake upload failed: User account is locked. Contact your Snowflake admin.</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="warning-card">❌ Snowflake upload failed: {e}</div>', unsafe_allow_html=True)
                        
                        # Enhanced analytics section
                        if st.expander("📊 Advanced Analytics & Insights"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 🎯 Matching Quality Analysis")
                                match_type_counts = st.session_state.df_final["Final_Match_Type"].value_counts()
                                for match_type, count in match_type_counts.items():
                                    percentage = (count / len(st.session_state.df_final)) * 100
                                    st.write(f"• **{match_type}**: {count} ({percentage:.1f}%)")
                            
                            with col2:
                                st.markdown("### 📊 Category Distribution")
                                if "survey_category" in st.session_state.df_final.columns:
                                    category_counts = st.session_state.df_final["survey_category"].value_counts()
                                    for category, count in category_counts.items():
                                        percentage = (count / len(st.session_state.df_final)) * 100
                                        st.write(f"• **{category}**: {count} ({percentage:.1f}%)")
                            
                            # Conflict analysis
                            if conflicts > 0:
                                st.markdown("### ⚠️ Conflict Analysis")
                                conflict_df = st.session_state.df_final[st.session_state.df_final.get("UID_Conflict", "") == "⚠️ Conflict"]
                                st.write(f"Found {len(conflict_df)} questions with UID conflicts")
                                
                                if st.button("🔧 Auto-Resolve Conflicts"):
                                    # Implement conflict resolution logic
                                    st.info("🔄 Auto-conflict resolution is not implemented yet. Manual review recommended.")
                            
                    except Exception as e:
                        logger.error(f"Enhanced UID matching failed: {e}")
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
                            st.markdown(f'<div class="warning-card">❌ Enhanced UID matching failed: {e}</div>', unsafe_allow_html=True)
                            raise
            else:
                st.markdown('<div class="info-card">ℹ️ Select a survey to start enhanced configuration.</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Enhanced SurveyMonkey processing failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced View Question Bank Page
elif st.session_state.page == "view_question_bank":
    st.markdown("## 📖 Enhanced View Question Bank")
    st.markdown("*Complete question repository with governance compliance and quality insights*")
    
    try:
        with st.spinner("🔄 Fetching ALL Snowflake question bank data..."):
            df_reference = get_all_reference_questions()
        
        if df_reference.empty:
            st.markdown('<div class="warning-card">⚠️ No data retrieved from Snowflake.</div>', unsafe_allow_html=True)
        else:
            # Enhanced metrics with governance
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Reference Questions", len(df_reference))
            with col2:
                st.metric("🎯 Target Questions", len(df_target))
            with col3:
                # Governance compliance of reference data
                uid_counts = df_reference['uid'].value_counts()
                compliant_rate = (len(uid_counts[uid_counts <= UID_GOVERNANCE['max_variations_per_uid']]) / len(uid_counts)) * 100
                st.metric("⚖️ Reference Compliance", f"{compliant_rate:.1f}%")
            with col4:
                st.metric("🔄 Status", "Ready to Match")
            
            st.markdown("---")
            
            with st.spinner("🤖 Running enhanced UID matching algorithm..."):
                df_final = run_uid_match(df_reference, df_target, ENHANCED_SYNONYM_MAP)
            
            # Enhanced results display with governance
            col1, col2, col3, col4, col5 = st.columns(5)
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
            with col5:
                # Governance compliance of matches
                governance_compliant = len(df_final[df_final.get("Final_Governance", "✅") == "✅"])
                total_matched = len(df_final[df_final["Final_UID"].notna()])
                governance_rate = (governance_compliant / total_matched * 100) if total_matched > 0 else 0
                st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
            
            st.markdown("---")
            
            # Enhanced filter controls
            st.markdown("### 🎛️ Enhanced Filter Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence_filter = st.multiselect(
                    "🎯 Filter by Match Type",
                    ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                    default=["✅ High", "⚠️ Low", "🧠 Semantic"]
                )
            
            with col2:
                min_similarity = st.slider("📊 Minimum Similarity Score", 0.0, 1.0, 0.5, 0.05)
            
            with col3:
                governance_filter = st.selectbox("⚖️ Governance Filter", ["All", "Compliant Only", "Violations Only"])
            
            # Apply enhanced filters
            filtered_df = df_final[df_final.get("Final_Match_Type", "").isin(confidence_filter)]
            
            if "Similarity" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Similarity"] >= min_similarity]
            
            if governance_filter == "Compliant Only":
                filtered_df = filtered_df[filtered_df.get("Final_Governance", "✅") == "✅"]
            elif governance_filter == "Violations Only":
                filtered_df = filtered_df[filtered_df.get("Final_Governance", "✅") == "⚠️"]
            
            st.markdown(f"### 📋 Enhanced Matching Results ({len(filtered_df)} items)")
            
            # Enhanced display with governance information
            display_columns = ["heading_0", "Final_UID", "Final_Match_Type", "Final_Governance", "Similarity"]
            if "Semantic_Similarity" in filtered_df.columns:
                display_columns.append("Semantic_Similarity")
            if "Matched_Question" in filtered_df.columns:
                display_columns.append("Matched_Question")
            if "survey_category" in filtered_df.columns:
                display_columns.append("survey_category")
            
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            display_df = filtered_df[available_columns].copy()
            display_df = display_df.rename(columns={
                "heading_0": "Target Question",
                "Final_UID": "Matched UID",
                "Final_Match_Type": "Match Type",
                "Final_Governance": "Governance",
                "Similarity": "TF-IDF Score",
                "Semantic_Similarity": "Semantic Score",
                "Matched_Question": "Reference Question",
                "survey_category": "Category"
            })
            
            st.dataframe(
    display_df[config_columns],
    column_config={
        "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
        "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
        "position": st.column_config.NumberColumn("Position", width="small"),
        "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
        "schema_type": st.column_config.TextColumn("Schema Type", width="small"),
        "configured_final_UID": st.column_config.TextColumn("Final UID", width="medium"),
        "Final_Match_Type": st.column_config.TextColumn("Match Type", width="medium"),
        "Final_Governance": st.column_config.TextColumn("Governance", width="small"),
        "question_category": st.column_config.TextColumn("Question Category", width="small"),
        "survey_category": st.column_config.TextColumn("Survey Category", width="small")
    },
    hide_index=True,
    use_container_width=True
)
            
            # Enhanced download options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📥 Download All Results",
                    df_final.to_csv(index=False),
                    f"enhanced_uid_matching_results_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📥 Download Filtered Results", 
                    filtered_df.to_csv(index=False),
                    f"enhanced_uid_matches_filtered_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Generate governance report
                governance_violations = df_final[df_final.get("Final_Governance", "✅") == "⚠️"]
                if not governance_violations.empty:
                    st.download_button(
                        "⚖️ Download Governance Report",
                        governance_violations.to_csv(index=False),
                        f"matching_governance_violations_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.success("✅ No governance violations in matches")
                
    except Exception as e:
        logger.error(f"Enhanced question bank update failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked. Contact your Snowflake admin or wait 15–30 minutes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Create New Survey Page (keeping existing functionality)
elif st.session_state.page == "create_survey":
    st.markdown("## ➕ Enhanced Create New Survey")
    st.markdown("*Build and deploy a new survey with automatic categorization*")
    
    try:
        token = st.secrets.get("surveymonkey", {}).get("token", None)
        if not token:
            st.markdown('<div class="warning-card">❌ SurveyMonkey token is missing in secrets configuration.</div>', unsafe_allow_html=True)
            st.stop()
        
        st.markdown("### 🎯 Enhanced Survey Template Builder")
        
        with st.form("enhanced_survey_template_form"):
            # Enhanced basic survey settings
            col1, col2 = st.columns(2)
            with col1:
                survey_title = st.text_input("📝 Survey Title", value="New Survey")
                survey_language = st.selectbox("🌐 Language", ["en", "es", "fr", "de"], index=0)
            with col2:
                num_pages = st.number_input("📄 Number of Pages", min_value=1, max_value=10, value=1)
                # Auto-detect category based on title
                detected_category = categorize_survey(survey_title)
                survey_category = st.selectbox("📊 Survey Category", 
                                             list(SURVEY_CATEGORIES.keys()) + ["Other", "Unknown"], 
                                             index=list(SURVEY_CATEGORIES.keys()).index(detected_category) if detected_category in SURVEY_CATEGORIES.keys() else len(SURVEY_CATEGORIES))
            
            # Enhanced survey settings
            st.markdown("#### ⚙️ Enhanced Survey Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_progress_bar = st.checkbox("📊 Show Progress Bar", value=True)
                hide_asterisks = st.checkbox("⭐ Hide Required Asterisks", value=False)
            with col2:
                one_question_at_a_time = st.checkbox("1️⃣ One Question Per Page", value=False)
                apply_governance = st.checkbox("⚖️ Apply Governance Rules", value=True)
            with col3:
                auto_assign_uids = st.checkbox("🤖 Auto-Assign UIDs", value=False)
                quality_threshold = st.slider("🎯 Quality Threshold", 0.0, 20.0, UID_GOVERNANCE['quality_score_threshold'], 0.5)
            
            # Enhanced pages and questions builder
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
                        
                        # Enhanced question quality preview
                        if question_text:
                            quality_score = score_question_quality(question_text)
                            quality_color = "🟢" if quality_score >= quality_threshold else "🟡" if quality_score >= quality_threshold/2 else "🔴"
                            st.write(f"Quality Score: {quality_color} {quality_score:.1f}")
                        
                        question_template = {
                            "heading": question_text,
                            "position": question_position,
                            "is_required": is_required,
                            "quality_score": score_question_quality(question_text) if question_text else 0
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
            
            # Enhanced survey template compilation
            survey_template = {
                "title": survey_title,
                "language": survey_language,
                "category": survey_category,
                "pages": pages,
                "settings": {
                    "progress_bar": show_progress_bar,
                    "hide_asterisks": hide_asterisks,
                    "one_question_at_a_time": one_question_at_a_time,
                    "apply_governance": apply_governance,
                    "auto_assign_uids": auto_assign_uids,
                    "quality_threshold": quality_threshold
                },
                "theme": {
                    "name": "enhanced",
                    "font": "Arial",
                    "background_color": "#FFFFFF",
                    "question_color": "#000000",
                    "answer_color": "#000000"
                }
            }
            
            submit = st.form_submit_button("🚀 Create Enhanced Survey", type="primary", use_container_width=True)
            
            if submit:
                if not survey_title or not pages:
                    st.markdown('<div class="warning-card">⚠️ Survey title and at least one page with questions are required.</div>', unsafe_allow_html=True)
                else:
                    # Quality check before creation
                    total_quality_score = sum([q.get("quality_score", 0) for page in pages for q in page.get("questions", [])])
                    avg_quality = total_quality_score / sum([len(page.get("questions", [])) for page in pages]) if pages else 0
                    
                    if apply_governance and avg_quality < quality_threshold:
                        st.warning(f"⚠️ Average question quality ({avg_quality:.1f}) is below threshold ({quality_threshold}). Consider improving question quality.")
                    
                    st.session_state.survey_template = survey_template
                    
                    try:
                        with st.spinner("🔄 Creating enhanced survey in SurveyMonkey..."):
                            # Create survey
                            survey_id = create_survey(token, survey_template)
                            
                            # Create pages and questions
                            for page_template in survey_template["pages"]:
                                page_id = create_page(token, survey_id, page_template)
                                for question_template in page_template["questions"]:
                                    create_question(token, survey_id, page_id, question_template)
                            
                            st.markdown(f'<div class="success-card">🎉 Enhanced survey created successfully!<br>Survey ID: <strong>{survey_id}</strong><br>Category: <strong>{survey_category}</strong><br>Average Quality: <strong>{avg_quality:.1f}</strong></div>', unsafe_allow_html=True)
                            st.balloons()
                            
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Failed to create survey: {e}</div>', unsafe_allow_html=True)
        
        # Enhanced preview section
        if st.session_state.survey_template:
            st.markdown("---")
            st.markdown("### 👀 Enhanced Survey Template Preview")
            
            # Enhanced summary display
            template = st.session_state.survey_template
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📄 Pages", len(template.get("pages", [])))
            with col2:
                total_questions = sum(len(page.get("questions", [])) for page in template.get("pages", []))
                st.metric("❓ Questions", total_questions)
            with col3:
                st.metric("📊 Category", template.get("category", "Unknown"))
            with col4:
                avg_quality = sum([q.get("quality_score", 0) for page in template.get("pages", []) for q in page.get("questions", [])]) / total_questions if total_questions > 0 else 0
                st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
            with col5:
                governance_icon = "✅" if template.get("settings", {}).get("apply_governance", False) else "❌"
                st.metric("⚖️ Governance", governance_icon)
            
            with st.expander("🔍 View Enhanced JSON Template"):
                st.json(template)
            
            # Enhanced download template
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Enhanced Template",
                    json.dumps(template, indent=2),
                    f"enhanced_survey_template_{uuid4()}.json",
                    "application/json",
                    use_container_width=True
                )
            with col2:
                # Generate quality report
                quality_report = []
                for page_idx, page in enumerate(template.get("pages", [])):
                    for q_idx, question in enumerate(page.get("questions", [])):
                        quality_report.append({
                            "page": page_idx + 1,
                            "question": q_idx + 1,
                            "text": question.get("heading", ""),
                            "quality_score": question.get("quality_score", 0),
                            "type": question.get("family", "unknown")
                        })
                
                if quality_report:
                    quality_df = pd.DataFrame(quality_report)
                    st.download_button(
                        "🎯 Download Quality Report",
                        quality_df.to_csv(index=False),
                        f"survey_quality_report_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
    except Exception as e:
        logger.error(f"Enhanced survey creation failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Data Quality Management Page
elif st.session_state.page == "data_quality":
    st.markdown("## 🧹 Enhanced Data Quality Management")
    st.markdown("*Comprehensive data quality analysis with governance compliance*")
    
    try:
        with st.spinner("🔄 Loading data for enhanced quality analysis..."):
            df_reference = get_all_reference_questions()
        
        if df_reference.empty:
            st.markdown('<div class="warning-card">⚠️ No data available for analysis.</div>', unsafe_allow_html=True)
        else:
            # Use the enhanced create_data_quality_dashboard function
            cleaned_df = create_data_quality_dashboard(df_reference)
            
    except Exception as e:
        st.error(f"❌ Enhanced data quality analysis failed: {e}")

# Navigation footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Return to Dashboard", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

with col2:
    st.markdown("*Enhanced with ❤️ using Streamlit & AI*")

with col3:
    current_page = st.session_state.page.replace('_', ' ').title()
    st.markdown(f"**Current Page:** {current_page}")

# Enhanced footer with governance info
st.markdown("---")
st.markdown("### ⚖️ Governance Settings")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"**Max Variations per UID:** {UID_GOVERNANCE['max_variations_per_uid']}")
with col2:
    st.markdown(f"**Semantic Threshold:** {UID_GOVERNANCE['semantic_similarity_threshold']}")
with col3:
    st.markdown(f"**Auto-Consolidate Threshold:** {UID_GOVERNANCE['auto_consolidate_threshold']}")
with col4:
    st.markdown(f"**Quality Threshold:** {UID_GOVERNANCE['quality_score_threshold']}")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True), col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Total Questions", len(df_reference))
            
            with col2:
                unique_uids = df_reference['uid'].nunique()
                st.metric("🆔 Unique UIDs", unique_uids)
            
            with col3:
                avg_variants = len(df_reference) / unique_uids if unique_uids > 0 else 0
                st.metric("📝 Avg Variants/UID", f"{avg_variants:.1f}")
            
            with col4:
                # Governance compliance check
                uid_counts = df_reference['uid'].value_counts()
                compliant_uids = len(uid_counts[uid_counts <= UID_GOVERNANCE['max_variations_per_uid']])
                compliance_rate = (compliant_uids / unique_uids) * 100 if unique_uids > 0 else 0
                st.metric("⚖️ Governance Rate", f"{compliance_rate:.1f}%")
            
            with col5:
                # Data completeness
                completeness = 100  # All fetched data is complete
                st.metric("✅ Data Loaded", f"{completeness:.0f}%")
            
            # Quick governance alert
            violations = uid_counts[uid_counts > UID_GOVERNANCE['max_variations_per_uid']]
            if not violations.empty:
                st.warning(f"⚖️ Governance Alert: {len(violations)} UIDs exceed the maximum variation limit ({UID_GOVERNANCE['max_variations_per_uid']} per UID)")
            
            st.markdown("---")
            
            # Enhanced search and filter
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_query = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            
            with col2:
                uid_filter = st.text_input("🆔 Filter by UID", placeholder="Enter UID...")
            
            with col3:
                sort_by = st.selectbox("🔄 Sort by", ["UID (ascending)", "UID (descending)", "Question length", "Survey title"])
            
            # Additional filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                governance_filter = st.selectbox("⚖️ Governance Filter", ["All", "Compliant Only", "Violations Only"])
            
            with col2:
                if 'survey_title' in df_reference.columns:
                    category_filter = st.selectbox("📊 Survey Category", ["All"] + sorted([categorize_survey(title) for title in df_reference['survey_title'].unique() if pd.notna(title)]))
                else:
                    category_filter = "All"
            
            with col3:
                variation_filter = st.selectbox("📝 Variation Count", ["All", "Single (1)", "Few (2-5)", "Many (6-20)", "Excessive (>20)"])
            
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
            
            st.markdown(f"### 📋 Enhanced Question Bank ({len(filtered_df)} questions showing all variations)")
            
            # Add enhanced grouping information
            if not filtered_df.empty:
                uid_counts_display = filtered_df['uid'].value_counts().sort_index()
                top_uids = uid_counts_display.head(5)
                
                # Enhanced info with governance status
                governance_violations_display = len(uid_counts_display[uid_counts_display > UID_GOVERNANCE['max_variations_per_uid']])
                
                info_text = f"💡 Showing variations across {len(uid_counts_display)} UIDs. "
                info_text += f"UIDs with most variations: {', '.join([f'{uid}({count})' for uid, count in top_uids.items()])}. "
                if governance_violations_display > 0:
                    info_text += f"⚖️ {governance_violations_display} UIDs violate governance rules."
                
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
                    "📥 Download Filtered Question Bank",
                    filtered_df.to_csv(index=False),
                    f"enhanced_question_bank_{uuid4()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Generate governance report
                governance_violations_df = df_reference.groupby('uid').size().reset_index(columns=['variation_count'])
                governance_violations_df = governance_violations_df[governance_violations_df['variation_count'] > UID_GOVERNANCE['max_variations_per_uid']]
                
                if not governance_violations_df.empty:
                    st.download_button(
                        "⚖️ Download Governance Violations",
                        governance_violations_df.to_csv(index=False),
                        f"governance_violations_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.success("✅ No governance violations")
            
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
                        "🎯 Download Quality Analysis",
                        quality_df.to_csv(index=False),
                        f"quality_analysis_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
    except Exception as e:
        logger.error(f"Enhanced Snowflake processing failed: {e}")
        if "250001" in str(e):
            st.markdown('<div class="warning-card">🔒 Snowflake connection failed: User account is locked. Contact your Snowflake admin or wait 15–30 minutes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced View Surveys Page (keeping existing functionality)
elif st.session_state.page == "view_surveys":
    st.markdown("## 👁️ Enhanced View Surveys")
    st.markdown("*Browse and analyze your SurveyMonkey surveys with categorization*")
    
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
            # Enhanced survey metrics with categorization
            survey_categories = [categorize_survey(s.get('title', '')) for s in surveys]
            category_counts = pd.Series(survey_categories).value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Surveys", len(surveys))
            with col2:
                recent_surveys = [s for s in surveys if s.get('date_created', '').startswith('2024') or s.get('date_created', '').startswith('2025')]
                st.metric("🆕 Recent (2024-2025)", len(recent_surveys))
            with col3:
                st.metric("📊 Categories", len(category_counts))
            with col4:
                most_common_category = category_counts.index[0] if not category_counts.empty else "None"
                st.metric("🏆 Top Category", most_common_category)
            
            # Category breakdown
            if not category_counts.empty:
                st.markdown("### 📊 Survey Categories")
                category_cols = st.columns(min(4, len(category_counts)))
                for i, (category, count) in enumerate(category_counts.head(4).items()):
                    with category_cols[i]:
                        st.metric(f"📋 {category}", count)
            
            st.markdown("---")
            
            # Enhanced survey selection interface
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
            
            # Category filter
            category_filter = st.selectbox("📊 Filter by Category", ["All"] + sorted(category_counts.index.tolist()))
            
            # Process selected surveys
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
                    with st.spinner(f"🔄 Fetching survey questions for ID {survey_id}..."):
                        survey_json = get_survey_details(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                    progress_bar.progress((i + 1) / len(all_selected_survey_ids))
                
                st.session_state.df_target = pd.DataFrame(combined_questions)
                
                if st.session_state.df_target.empty:
                    st.markdown('<div class="warning-card">⚠️ No questions found in the selected survey(s).</div>', unsafe_allow_html=True)
                else:
                    # Enhanced analysis metrics
                    st.markdown("### 📊 Enhanced Survey Analysis")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
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
                    with col5:
                        # Add survey categories
                        st.session_state.df_target["survey_category"] = st.session_state.df_target["survey_title"].apply(categorize_survey)
                        unique_categories = st.session_state.df_target["survey_category"].nunique()
                        st.metric("🏷️ Categories", unique_categories)
                    
                    st.markdown("---")
                    
                    # Enhanced display options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_main_only = st.checkbox("📋 Show only main questions", value=False)
                    with col2:
                        question_filter = st.selectbox("🔍 Filter by Q category", 
                                                     ["All", "Main Question/Multiple Choice", "Heading"])
                    with col3:
                        survey_category_filter = st.selectbox("📊 Filter by S category",
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
                    
                    st.markdown(f"### 📋 Enhanced Survey Questions ({len(display_df)} items)")
                    
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
                        "📥 Download Enhanced Survey Data",
                        display_df.to_csv(index=False),
                        f"enhanced_survey_data_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ Select a survey or category to view questions and analysis.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Enhanced SurveyMonkey processing failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)

# Enhanced Update Question Bank Page (keeping existing functionality with governance)
elif st.session_state.page == "update_question_bank":
    st.markdown("## 🔄 Enhanced Update Question Bank")
    st.markdown("*Match new questions with existing UIDs using enhanced algorithms and governance rules*")
    
    try:
        with st.spinner("🔄 Fetching Snowflake data..."):
            df_reference = get_all_reference_questions()
            df_target = run_snowflake_target_query()
        
        if df_reference.empty or df_target.empty:
            st.markdown('<div class="warning-card">⚠️ No data retrieved from Snowflake for matching.</div>', unsafe_allow_html=True)
        else:
            # Enhanced initial metrics
            col1, col2, col3, col4 # Enhanced Configure Survey Page with Governance
elif st.session_state.page == "configure_survey":
    st.markdown("## ⚙️ Enhanced Configure Survey")
    st.markdown("*Match survey questions with UIDs using advanced semantic matching and governance rules*")
    
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
            
            # Enhanced matching options
            st.markdown("### ⚙️ Enhanced Matching Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_semantic_matching = st.checkbox("🧠 Enable Semantic Matching", value=True)
                semantic_threshold = st.slider("Semantic Threshold", 0.5, 0.95, SEMANTIC_THRESHOLD, 0.05)
            
            with col2:
                enforce_governance = st.checkbox("⚖️ Enforce Governance Rules", value=True)
                max_variations = st.number_input("Max Variations per UID", 1, 100, UID_GOVERNANCE['max_variations_per_uid'])
            
            with col3:
                auto_resolve_conflicts = st.checkbox("🔧 Auto-Resolve Conflicts", value=False)
                quality_threshold = st.slider("Quality Threshold", 0.0, 20.0, UID_GOVERNANCE['quality_score_threshold'], 0.5)
            
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
                    # Run enhanced UID matching
                    try:
                        with st.spinner("🔄 Running enhanced UID matching..."):
                            st.session_state.df_reference = get_all_reference_questions()
                            
                            # Update governance settings
                            if enforce_governance:
                                UID_GOVERNANCE['max_variations_per_uid'] = max_variations
                                UID_GOVERNANCE['quality_score_threshold'] = quality_threshold
                                UID_GOVERNANCE['semantic_similarity_threshold'] = semantic_threshold
                            
                            # Enhanced synonym mapping for this session
                            session_synonym_map = ENHANCED_SYNONYM_MAP.copy()
                            
                            st.session_state.df_final = run_uid_match(st.session_state.df_reference, st.session_state.df_target, session_synonym_map)
                            st.session_state.uid_changes = {}
                            
                        # Enhanced matching results
                        matched_percentage = calculate_matched_percentage(st.session_state.df_final)
                        
                        st.markdown("### 📊 Enhanced Configuration Results")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("📊 Match Rate", f"{matched_percentage}%")
                        
                        with col2:
                            total_q = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                            st.metric("❓ Questions", total_q)
                        
                        with col3:
                            governance_compliant = len(st.session_state.df_final[st.session_state.df_final.get("Final_Governance", "✅") == "✅"])
                            total_matched = len(st.session_state.df_final[st.session_state.df_final["Final_UID"].notna()])
                            governance_rate = (governance_compliant / total_matched * 100) if total_matched > 0 else 0
                            st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
                        
                        with col4:
                            semantic_matches = len(st.session_state.df_final[st.session_state.df_final.get("Final_Match_Type", "") == "🧠 Semantic"])
                            st.metric("🧠 Semantic Matches", semantic_matches)
                        
                        with col5:
                            conflicts = len(st.session_state.df_final[st.session_state.df_final.get("UID_Conflict", "") == "⚠️ Conflict"])
                            st.metric("⚠️ Conflicts", conflicts)
                        
                        # Governance violations alert
                        governance_violations = st.session_state.df_final[st.session_state.df_final.get("Final_Governance", "✅") == "⚠️"]
                        if not governance_violations.empty and enforce_governance:
                            st.warning(f"⚖️ Found {len(governance_violations)} governance violations. These UIDs exceed the maximum variation limit.")
                        
                        # Display enhanced configuration interface
                        st.markdown("---")
                        st.markdown("### ⚙️ Enhanced Configuration & Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            show_main_only = st.checkbox("📋 Show only main questions", value=True)
                        with col2:
                            search_query = st.text_input("🔍 Search questions", placeholder="Type to filter...")
                        with col3:
                            match_type_filter = st.selectbox("🎯 Match Type", ["All", "✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"])
                        
                        # Filter and display results with enhanced columns
                        display_df = st.session_state.df_final.copy()
                        
                        if show_main_only:
                            display_df = display_df[display_df["is_choice"] == False]
                        
                        if search_query:
                            display_df = display_df[display_df["heading_0"].str.contains(search_query, case=False, na=False)]
                        
                        if match_type_filter != "All":
                            display_df = display_df[display_df.get("Final_Match_Type", "") == match_type_filter]
                        
                        # Add enhanced columns
                        display_df["survey_id_title"] = display_df.apply(
                            lambda x: f"{x['survey_id']} - {x['survey_title']}" if pd.notnull(x['survey_id']) and pd.notnull(x['survey_title']) else "",
                            axis=1
                        )
                        
                        st.markdown(f"### 📋 Enhanced Survey Configuration ({len(display_df)} items)")
                        
                        # Display configuration table with governance info
                        config_columns = [
                            "survey_id_title", "heading_0", "position", "is_choice", "schema_type", 
                            "configured_final_UID", "Final_Match_Type", "Final_Governance", "question_category", "survey_category"
                        ]
                        config_columns = [col for col in config_columns if col in display_df.columns]
                        
                        st.dataframe(
                            display_df[config_columns],
                            column_config={
                                "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
                                "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
                                "position": st.column_config.NumberColumn("Position", width="small"),
                                "

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
    if st.button("📊 Categorized Questions", use_container_width=True):
        st.session_state.page = "categorized_questions"
        st.rerun()
    if st.button("🔄 Update Question Bank", use_container_width=True):
        st.session_state.page = "update_question_bank"
        st.rerun()
    if st.button("🧹 Data Quality Management", use_container_width=True):
        st.session_state.page = "data_quality"
        st.rerun()
    
    st.markdown("---")
    
    # Governance section
    st.markdown("**⚖️ Governance**")
    st.markdown(f"• Max variations per UID: {UID_GOVERNANCE['max_variations_per_uid']}")
    st.markdown(f"• Semantic threshold: {UID_GOVERNANCE['semantic_similarity_threshold']}")
    st.markdown(f"• Quality threshold: {UID_GOVERNANCE['quality_score_threshold']}")
    
    st.markdown("---")
    
    # Quick links
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

# App UI with enhanced styling
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets or "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing secrets configuration for Snowflake or SurveyMonkey.</div>', unsafe_allow_html=True)
    st.stop()

# Home Page with Enhanced Dashboard
if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher Pro")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Governance", "Enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced features highlight
    st.markdown("## 🚀 Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Enhanced UID Matching")
        st.markdown("• **Semantic Matching**: AI-powered question similarity")
        st.markdown("• **Governance Rules**: Automatic compliance checking")
        st.markdown("• **Conflict Detection**: Real-time duplicate identification")
        st.markdown("• **Quality Scoring**: Advanced question assessment")
        
        if st.button("🔧 Configure Survey with Enhanced Matching", use_container_width=True):
            st.session_state.page = "configure_survey"
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Survey Categorization")
        st.markdown("• **Auto-Categorization**: Smart survey type detection")
        st.markdown("• **Category Filters**: Application, GROW, Impact, etc.")
        st.markdown("• **Cross-Category Analysis**: Compare question patterns")
        st.markdown("• **Quality by Category**: Category-specific insights")
        
        if st.button("📊 View Categorized Questions", use_container_width=True):
            st.session_state.page = "categorized_questions"
            st.rerun()
    
    st.markdown("---")
    
    # Quick actions grid
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 SurveyMonkey Operations")
        if st.button("👁️ View & Analyze Surveys", use_container_width=True):
            st.session_state.page = "view_surveys"
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
    
    # System status with governance
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
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
    
    with status_col3:
        st.markdown('<div class="success-card">✅ Governance: Active</div>', unsafe_allow_html=True)
        st.markdown(f"Max variations: {UID_GOVERNANCE['max_variations_per_uid']}")

# Enhanced Unique Questions Bank Page
elif st.session_state.page == "unique_question_bank":
    st.markdown("## ⭐ Enhanced Unique Questions Bank")
    st.markdown("*Best structured question for each UID with governance compliance and quality scoring*")
    
    try:
        with st.spinner("🔄 Loading ALL question bank data and creating unique questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                st.info(f"📊 Loaded {len(df_reference)} total question variants from database")
                
                # Create unique questions bank
                unique_questions_df = create_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.markdown('<div class="warning-card">⚠️ No unique questions found in the database.</div>', unsafe_allow_html=True)
        else:
            # Enhanced summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🆔 Unique UIDs", len(unique_questions_df))
            with col2:
                st.metric("📝 Total Variants", unique_questions_df['total_variants'].sum())
            with col3:
                governance_compliant = len(unique_questions_df[unique_questions_df['governance_compliant'] == True])
                st.metric("⚖️ Governance Compliant", f"{governance_compliant}/{len(unique_questions_df)}")
            with col4:
                avg_quality = unique_questions_df['quality_score'].mean()
                st.metric("🎯 Avg Quality Score", f"{avg_quality:.1f}")
            with col5:
                categories = unique_questions_df['survey_category'].nunique()
                st.metric("📊 Categories", categories)
            
            st.markdown("---")
            
            # Enhanced search and filter options
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_term = st.text_input("🔍 Search questions", placeholder="Type to filter questions...")
            
            with col2:
                min_variants = st.selectbox("📊 Min variants", [1, 2, 3, 5, 10, 20], index=0)
            
            with col3:
                quality_filter = st.selectbox("🎯 Quality Filter", ["All", "High (>10)", "Medium (5-10)", "Low (<5)"])
            
            # Additional filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                governance_filter = st.selectbox("⚖️ Governance", ["All", "Compliant Only", "Violations Only"])
            
            with col2:
                category_filter = st.selectbox("📊 Category", ["All"] + sorted(unique_questions_df['survey_category'].unique().tolist()))
            
            with col3:
                show_variants = st.checkbox("👀 Show all variants", value=False)
            
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
            
            st.markdown(f"### 📋 Showing {len(filtered_df)} unique questions")
            
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
                display_df['Governance'] = display_df['Governance'].apply(lambda x: "✅" if x else "❌")
                
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
                        "📥 Download Filtered Results (CSV)",
                        display_df.to_csv(index=False),
                        f"unique_questions_filtered_{uuid4()}.csv",
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
                
                with col3:
                    # Generate governance report
                    governance_violations = unique_questions_df[unique_questions_df['governance_compliant'] == False]
                    if not governance_violations.empty:
                        st.download_button(
                            "⚖️ Download Governance Report",
                            governance_violations.to_csv(index=False),
                            f"governance_violations_{uuid4()}.csv",
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

# New Categorized Questions Page
elif st.session_state.page == "categorized_questions":
    st.markdown("## 📊 Categorized Questions Bank")
    st.markdown("*Questions organized by survey categories with detailed analysis*")
    
    try:
        with st.spinner("🔄 Loading and categorizing questions..."):
            df_reference = get_all_reference_questions()
            
            if df_reference.empty:
                st.markdown('<div class="warning-card">⚠️ No reference data found in the database.</div>', unsafe_allow_html=True)
            else:
                unique_questions_df = create_unique_questions_bank(df_reference)
        
        if unique_questions_df.empty:
            st.markdown('<div class="warning-card">⚠️ No categorized questions found.</div>', unsafe_allow_html=True)
        else:
            # Category overview
            category_stats = unique_questions_df.groupby('survey_category').agg({
                'uid': 'count',
                'total_variants': 'sum',
                'quality_score': 'mean',
                'governance_compliant': lambda x: (x == True).sum()
            }).round(2)
            
            category_stats.columns = ['Questions', 'Total Variants', 'Avg Quality', 'Governance Compliant']
            category_stats = category_stats.sort_values('Questions', ascending=False)
            
            st.markdown("### 📊 Category Overview")
            
            # Display category metrics
            categories = list(SURVEY_CATEGORIES.keys()) + ['Other', 'Unknown', 'Mixed']
            cols = st.columns(min(4, len(categories)))
            
            for i, category in enumerate(categories):
                if category in category_stats.index:
                    count = category_stats.loc[category, 'Questions']
                    with cols[i % 4]:
                        st.metric(f"📋 {category}", count)
            
            st.markdown("---")
            
            # Detailed category statistics
            st.markdown("### 📈 Detailed Category Statistics")
            st.dataframe(
                category_stats,
                column_config={
                    "Questions": st.column_config.NumberColumn("Questions", width="small"),
                    "Total Variants": st.column_config.NumberColumn("Total Variants", width="small"),
                    "Avg Quality": st.column_config.NumberColumn("Avg Quality Score", format="%.1f", width="small"),
                    "Governance Compliant": st.column_config.NumberColumn("Governance Compliant", width="small")
                },
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Category filter and detailed view
            st.markdown("### 🔍 Detailed Category Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_category = st.selectbox(
                    "📊 Select Category for Detailed View",
                    ["All"] + sorted(unique_questions_df['survey_category'].unique().tolist())
                )
            
            with col2:
                sort_by = st.selectbox(
                    "🔄 Sort by",
                    ["UID", "Quality Score", "Total Variants", "Question Length"]
                )
            
            # Filter by selected category
            if selected_category == "All":
                filtered_df = unique_questions_df.copy()
            else:
                filtered_df = unique_questions_df[unique_questions_df['survey_category'] == selected_category].copy()
            
            # Apply sorting
            if sort_by == "UID":
                try:
                    filtered_df['uid_numeric'] = pd.to_numeric(filtered_df['uid'], errors='coerce')
                    filtered_df = filtered_df.sort_values(['uid_numeric', 'uid'], na_position='last')
                    filtered_df = filtered_df.drop('uid_numeric', axis=1)
                except:
                    filtered_df = filtered_df.sort_values('uid')
            elif sort_by == "Quality Score":
                filtered_df = filtered_df.sort_values('quality_score', ascending=False)
            elif sort_by == "Total Variants":
                filtered_df = filtered_df.sort_values('total_variants', ascending=False)
            elif sort_by == "Question Length":
                filtered_df = filtered_df.sort_values('question_length', ascending=False)
            
            st.markdown(f"### 📋 {selected_category} Questions ({len(filtered_df)} items)")
            
            if not filtered_df.empty:
                # Category-specific insights
                if selected_category != "All":
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_quality = filtered_df['quality_score'].mean()
                        st.metric("🎯 Avg Quality", f"{avg_quality:.1f}")
                    
                    with col2:
                        total_variants = filtered_df['total_variants'].sum()
                        st.metric("📝 Total Variants", total_variants)
                    
                    with col3:
                        governance_rate = (filtered_df['governance_compliant'] == True).sum() / len(filtered_df) * 100
                        st.metric("⚖️ Governance Rate", f"{governance_rate:.1f}%")
                    
                    with col4:
                        avg_length = filtered_df['question_length'].mean()
                        st.metric("📏 Avg Length", f"{avg_length:.0f} chars")
                
                # Display questions
                display_df = filtered_df[['uid', 'best_question', 'survey_category', 'total_variants', 'quality_score', 'governance_compliant']].copy()
                display_df['governance_compliant'] = display_df['governance_compliant'].apply(lambda x: "✅" if x else "❌")
                
                display_df = display_df.rename(columns={
                    'uid': 'UID',
                    'best_question': 'Question',
                    'survey_category': 'Category',
                    'total_variants': 'Variants',
                    'quality_score': 'Quality',
                    'governance_compliant': 'Governance'
                })
                
          st.dataframe(
    display_df[config_columns],
    column_config={
        "survey_id_title": st.column_config.TextColumn("Survey", width="medium"),
        "heading_0": st.column_config.TextColumn("Question/Choice", width="large"),
        "position": st.column_config.NumberColumn("Position", width="small"),
        "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
        "schema_type": st.column_config.TextColumn("Schema Type", width="small"),
        "configured_final_UID": st.column_config.TextColumn("Final UID", width="medium"),
        "Final_Match_Type": st.column_config.TextColumn("Match Type", width="medium"),
        "Final_Governance": st.column_config.TextColumn("Governance", width="small"),
        "question_category": st.column_config.TextColumn("Question Category", width="small"),
        "survey_category": st.column_config.TextColumn("Survey Category", width="small")
    },
    hide_index=True,
    use_container_width=True
)
                                # Download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        f"📥 Download {selected_category} Questions",
                        filtered_df.to_csv(index=False),
                        f"{selected_category.lower()}_questions_{uuid4()}.csv",
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
                    # Cross-category comparison
                    comparison_df = unique_questions_df.groupby('survey_category').agg({
                        'quality_score': ['mean', 'std', 'min', 'max'],
                        'total_variants': ['mean', 'sum'],
                        'governance_compliant': lambda x: (x == True).sum() / len(x) * 100
                    }).round(2)
                    
                    st.download_button(
                        "📈 Download Cross-Category Analysis",
                        comparison_df.to_csv(),
                        f"cross_category_analysis_{uuid4()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.markdown('<div class="info-card">ℹ️ No questions found in the selected category.</div>', unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Categorized questions failed: {e}")
        st.markdown(f'<div class="warning-card">❌ Error: {e}</div>', unsafe_allow_html=True)
