# Simple form for adding questions
        with st.form("add_question_form"):
            question_text = st.text_area("📝 Question Text:")
            question_type = st.selectbox("📊 Question Type:", ["Single Choice", "Multiple Choice", "Open-Ended"])
            
            if st.form_submit_button("➕ Add Question"):
                if question_text.strip():
                    st.success("✅ Question would be added to manual survey!")
                else:
                    st.error("❌ Please enter question text")

def render_create_survey():
    st.markdown("## ➕ Create New Survey")
    st.markdown("*Create a new survey in SurveyMonkey*")
    
    token = st.secrets.get("surveymonkey", {}).get("token", None)
    if not token:
        st.markdown('<div class="warning-card">⚠️ SurveyMonkey token not configured.</div>', unsafe_allow_html=True)
        return
    
    with st.form("create_survey_form"):
        survey_title = st.text_input("📝 Survey Title:")
        survey_description = st.text_area("📄 Survey Description (optional):")
        survey_language = st.selectbox("🌐 Language:", ["en", "es", "fr", "de"])
        
        if st.form_submit_button("🚀 Create Survey", type="primary"):
            if survey_title.strip():
                st.success(f"✅ Would create survey: '{survey_title}'")
            else:
                st.error("❌ Please enter a survey title")

# Create sidebar
create_sidebar()

# App Header
st.markdown('<div class="main-header">🧠 UID Matcher Pro: Enhanced with Governance & Categories</div>', unsafe_allow_html=True)

# Secrets Validation
if "snowflake" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing Snowflake configuration in secrets.</div>', unsafe_allow_html=True)
    st.markdown("Please configure your Snowflake credentials in the secrets to use database features.")

if "surveymonkey" not in st.secrets:
    st.markdown('<div class="warning-card">⚠️ Missing SurveyMonkey configuration in secrets.</div>', unsafe_allow_html=True)
    st.markdown("Please configure your SurveyMonkey token in the secrets to use API features.")

# Main page routing
if st.session_state.page == "home":
    render_home_page()

elif st.session_state.page == "view_surveys":
    render_view_surveys()

elif st.session_state.page == "configure_survey":
    render_configure_survey()

elif st.session_state.page == "create_survey":
    render_create_survey()

elif st.session_state.page == "view_question_bank":
    render_view_question_bank()

elif st.session_state.page == "unique_question_bank":
    render_unique_question_bank()

elif st.session_state.page == "categorized_questions":
    render_categorized_questions()

else:
    # Default fallback for any undefined pages
    st.markdown("## 🔄 Page Under Development")
    st.markdown("This page is currently being developed. Please use the sidebar to navigate to available pages.")
    
    # Show available pages
    st.markdown("### Available Pages:")
    st.markdown("• **🏠 Home Dashboard** - Main overview and system status")
    st.markdown("• **👁️ View Surveys** - Browse SurveyMonkey surveys")
    st.markdown("• **⚙️ Configure Survey** - Upload and configure surveys")
    st.markdown("• **➕ Create Survey** - Create new surveys")
    st.markdown("• **📖 View Question Bank** - Browse all questions")
    st.markdown("• **⭐ Unique Questions Bank** - Best questions per UID")
    st.markdown("• **📊 Categorized Questions** - Questions by category")
    
    if st.button("🏠 Return to Home", type="primary"):
        st.session_state.page = "home"
        st.rerun()

# Footer information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <small>
        🧠 UID Matcher Pro v2.0 | 
        Enhanced with AI-powered semantic matching, governance compliance, and survey categorization
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)


