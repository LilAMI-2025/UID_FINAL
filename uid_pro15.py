import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple
import re
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="Advanced UID Matcher",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    
    .metric-card.blue { border-left-color: #3b82f6; }
    .metric-card.yellow { border-left-color: #f59e0b; }
    .metric-card.red { border-left-color: #ef4444; }
    .metric-card.green { border-left-color: #10b981; }
    
    .category-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 1rem;
    }
    
    .category-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .conflict-high { 
        background: #fef2f2; 
        border: 1px solid #fecaca; 
        border-radius: 8px; 
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .conflict-medium { 
        background: #fffbeb; 
        border: 1px solid #fed7aa; 
        border-radius: 8px; 
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .conflict-low { 
        background: #f0fdf4; 
        border: 1px solid #bbf7d0; 
        border-radius: 8px; 
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .survey-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }
    
    .survey-card.selected {
        border-color: #8b5cf6;
        background: #f3f4f6;
    }
    
    .sidebar-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_question_bank' not in st.session_state:
    st.session_state.processed_question_bank = None
if 'selected_surveys' not in st.session_state:
    st.session_state.selected_surveys = []
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = None

# Mock data for demonstration (replace with your actual data sources)
@st.cache_data
def load_mock_question_bank():
    """Load mock question bank data"""
    return pd.DataFrame([
        {'uid': '243', 'heading_0': 'What is your age?', 'count': 1000},
        {'uid': '333', 'heading_0': 'What is your age?', 'count': 512},
        {'uid': '156', 'heading_0': 'What is your name?', 'count': 850},
        {'uid': '789', 'heading_0': 'What is your sector?', 'count': 750},
        {'uid': '456', 'heading_0': 'What sector are you from?', 'count': 380},
        {'uid': '612', 'heading_0': 'How satisfied are you?', 'count': 920},
        {'uid': '234', 'heading_0': 'Rate your experience', 'count': 670},
        {'uid': '567', 'heading_0': 'Rate your overall experience', 'count': 420},
        {'uid': '891', 'heading_0': 'What is your company size?', 'count': 540},
        {'uid': '445', 'heading_0': 'Company size?', 'count': 290},
        {'uid': '678', 'heading_0': 'How long have you been in business?', 'count': 630},
        {'uid': '123', 'heading_0': 'Years in business?', 'count': 410}
    ])

@st.cache_data
def load_mock_survey_data():
    """Load mock survey data from SurveyMonkey"""
    return pd.DataFrame([
        {'id': '001', 'title': 'Pre Programme Assessment 2024', 'category': 'Pre programme', 'questions': 25, 'responses': 1200},
        {'id': '002', 'title': 'Application Form - Tech Cohort', 'category': 'Application', 'questions': 18, 'responses': 850},
        {'id': '003', 'title': 'GROW Leadership Assessment', 'category': 'GROW', 'questions': 32, 'responses': 640},
        {'id': '004', 'title': 'Enrollment Verification Survey', 'category': 'Enrollment', 'questions': 12, 'responses': 920},
        {'id': '005', 'title': 'Progress Review - Mid Term', 'category': 'Progress Review', 'questions': 28, 'responses': 780},
        {'id': '006', 'title': 'Impact Measurement Q4', 'category': 'Impact', 'questions': 22, 'responses': 560},
        {'id': '007', 'title': 'Feedback Collection - Training', 'category': 'Feedback', 'questions': 15, 'responses': 1100},
        {'id': '008', 'title': 'Pulse Check Weekly', 'category': 'Pulse', 'questions': 8, 'responses': 1450},
        {'id': '009', 'title': 'Application Review Process', 'category': 'Application', 'questions': 20, 'responses': 720},
        {'id': '010', 'title': 'GROW Coaching Evaluation', 'category': 'GROW', 'questions': 15, 'responses': 380}
    ])

def categorize_survey_from_title(title: str) -> str:
    """Categorize surveys based on title keywords"""
    title_lower = title.lower()
    
    # Define category keywords
    category_keywords = {
        'Application': ['application', 'apply', 'submission'],
        'Pre programme': ['pre programme', 'pre-programme', 'pre assessment', 'pre-assessment'],
        'Enrollment': ['enrollment', 'enrolment', 'verification', 'registration'],
        'Progress Review': ['progress', 'review', 'mid term', 'mid-term', 'milestone'],
        'Impact': ['impact', 'outcome', 'result', 'measurement'],
        'GROW': ['GROW'],  # Must be in CAPS
        'Feedback': ['feedback', 'evaluation', 'rating'],
        'Pulse': ['pulse', 'check', 'weekly', 'daily', 'quick']
    }
    
    # Check for GROW first (case sensitive)
    if 'GROW' in title:
        return 'GROW'
    
    # Check other categories
    for category, keywords in category_keywords.items():
        if category != 'GROW':  # Skip GROW as we checked it above
            for keyword in keywords:
                if keyword in title_lower:
                    return category
    
    return 'Other'

def process_question_bank(df_questions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Process question bank to find optimal UIDs and detect conflicts"""
    
    # Normalize questions for grouping
    df_questions['normalized_question'] = df_questions['heading_0'].str.lower().str.strip()
    
    # Group by normalized question
    question_groups = df_questions.groupby('normalized_question')
    
    processed_data = []
    conflict_summary = {
        'total_questions': 0,
        'conflicted_questions': 0,
        'total_conflicts': 0,
        'high_risk_conflicts': 0
    }
    
    for question, group in question_groups:
        # Sort by count descending to get optimal UID
        sorted_group = group.sort_values('count', ascending=False)
        optimal_row = sorted_group.iloc[0]
        conflicts = sorted_group.iloc[1:] if len(sorted_group) > 1 else pd.DataFrame()
        
        # Calculate conflict metrics
        has_conflicts = len(conflicts) > 0
        high_risk = False
        
        if has_conflicts:
            # High risk if any competing UID has >70% of optimal count
            high_risk = any(conflicts['count'] > optimal_row['count'] * 0.7)
        
        processed_data.append({
            'question': optimal_row['heading_0'],
            'optimal_uid': optimal_row['uid'],
            'count': optimal_row['count'],
            'has_conflicts': has_conflicts,
            'num_conflicts': len(conflicts),
            'high_risk': high_risk,
            'conflicts': conflicts[['uid', 'count']].to_dict('records') if has_conflicts else []
        })
        
        # Update summary
        conflict_summary['total_questions'] += 1
        if has_conflicts:
            conflict_summary['conflicted_questions'] += 1
            conflict_summary['total_conflicts'] += len(conflicts)
            if high_risk:
                conflict_summary['high_risk_conflicts'] += 1
    
    return pd.DataFrame(processed_data), conflict_summary

def create_dashboard_metrics(conflict_summary: Dict, total_surveys: int):
    """Create dashboard metrics cards"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>📊 Total Questions</h3>
            <h2>{conflict_summary['total_questions']}</h2>
            <p>Questions in database</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card yellow">
            <h3>⚠️ Conflicts</h3>
            <h2>{conflict_summary['conflicted_questions']}</h2>
            <p>Questions with UID conflicts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card red">
            <h3>🚨 High Risk</h3>
            <h2>{conflict_summary['high_risk_conflicts']}</h2>
            <p>Critical conflicts to resolve</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card green">
            <h3>📋 Surveys</h3>
            <h2>{total_surveys}</h2>
            <p>Available surveys</p>
        </div>
        """, unsafe_allow_html=True)

def create_conflict_chart(processed_df: pd.DataFrame):
    """Create conflict visualization chart"""
    
    # Prepare data for visualization
    conflict_data = processed_df.groupby('has_conflicts').size().reset_index()
    conflict_data['status'] = conflict_data['has_conflicts'].map({True: 'Has Conflicts', False: 'No Conflicts'})
    
    # Create pie chart
    fig = px.pie(
        conflict_data, 
        values=0, 
        names='status',
        title="UID Conflict Distribution",
        color_discrete_map={'Has Conflicts': '#ef4444', 'No Conflicts': '#10b981'}
    )
    
    fig.update_layout(
        showlegend=True,
        height=300,
        font=dict(size=12)
    )
    
    return fig

def create_category_overview(survey_df: pd.DataFrame):
    """Create category overview visualization"""
    
    # Count surveys and questions by category
    category_stats = survey_df.groupby('category').agg({
        'id': 'count',
        'questions': 'sum',
        'responses': 'sum'
    }).reset_index()
    category_stats.columns = ['category', 'survey_count', 'total_questions', 'total_responses']
    
    # Create category icons mapping
    category_icons = {
        'Application': '👥',
        'Pre programme': '📚',
        'Enrollment': '🎯',
        'Progress Review': '📈',
        'Impact': '🏆',
        'GROW': '🌱',
        'Feedback': '💬',
        'Pulse': '💓'
    }
    
    # Create category cards
    cols = st.columns(4)
    for i, (_, row) in enumerate(category_stats.iterrows()):
        with cols[i % 4]:
            icon = category_icons.get(row['category'], '📊')
            st.markdown(f"""
            <div class="category-card">
                <div style="font-size: 2em;">{icon}</div>
                <h4>{row['category']}</h4>
                <p><strong>{row['survey_count']}</strong> surveys</p>
                <p><strong>{row['total_questions']}</strong> questions</p>
                <p><strong>{row['total_responses']}</strong> responses</p>
            </div>
            """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Advanced UID Matcher</h1>
        <p>Intelligent question analysis and optimal UID assignment system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "🗄️ Question Bank", "📋 Survey Analysis"],
        index=0
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    question_bank_df = load_mock_question_bank()
    survey_df = load_mock_survey_data()
    
    # Add categories to surveys
    survey_df['category'] = survey_df['title'].apply(categorize_survey_from_title)
    
    # Process question bank
    if st.session_state.processed_question_bank is None:
        processed_df, conflict_summary = process_question_bank(question_bank_df)
        st.session_state.processed_question_bank = processed_df
        st.session_state.conflict_summary = conflict_summary
    else:
        processed_df = st.session_state.processed_question_bank
        conflict_summary = st.session_state.conflict_summary
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(processed_df, conflict_summary, survey_df)
    elif page == "🗄️ Question Bank":
        show_question_bank(processed_df, conflict_summary)
    elif page == "📋 Survey Analysis":
        show_survey_analysis(survey_df)

def show_dashboard(processed_df: pd.DataFrame, conflict_summary: Dict, survey_df: pd.DataFrame):
    """Show dashboard page"""
    
    st.header("📊 Dashboard Overview")
    
    # Metrics
    create_dashboard_metrics(conflict_summary, len(survey_df))
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_conflict_chart(processed_df), use_container_width=True)
    
    with col2:
        # Top conflicted questions
        top_conflicts = processed_df[processed_df['has_conflicts']].nlargest(5, 'num_conflicts')
        
        fig = px.bar(
            top_conflicts,
            x='num_conflicts',
            y='question',
            orientation='h',
            title="Top 5 Questions with Most Conflicts",
            color='high_risk',
            color_discrete_map={True: '#ef4444', False: '#f59e0b'}
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🔄 Recent Activity")
    
    activities = [
        {"time": "2 minutes ago", "action": "Question bank analysis completed", "status": "success"},
        {"time": "15 minutes ago", "action": "8 UID conflicts resolved", "status": "success"},
        {"time": "1 hour ago", "action": "3 high-risk conflicts detected in GROW category", "status": "warning"},
        {"time": "2 hours ago", "action": "New survey 'Impact Measurement Q4' processed", "status": "info"}
    ]
    
    for activity in activities:
        status_color = {
            "success": "#10b981",
            "warning": "#f59e0b", 
            "info": "#3b82f6"
        }[activity["status"]]
        
        st.markdown(f"""
        <div style="border-left: 4px solid {status_color}; padding-left: 1rem; margin: 0.5rem 0; background: #f8fafc; padding: 1rem; border-radius: 0 8px 8px 0;">
            <strong>{activity["time"]}</strong><br>
            {activity["action"]}
        </div>
        """, unsafe_allow_html=True)

def show_question_bank(processed_df: pd.DataFrame, conflict_summary: Dict):
    """Show question bank page"""
    
    st.header("🗄️ Intelligent Question Bank")
    
    # Conflict summary dashboard
    st.subheader("⚠️ UID Conflict Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="conflict-high">
            <h4>🚨 High Risk Conflicts</h4>
            <h2>{conflict_summary['high_risk_conflicts']}</h2>
            <p>Competing UIDs with similar usage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="conflict-medium">
            <h4>⚠️ Total Conflicts</h4>
            <h2>{conflict_summary['total_conflicts']}</h2>
            <p>UIDs requiring resolution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        resolution_rate = round(((conflict_summary['total_questions'] - conflict_summary['conflicted_questions']) / conflict_summary['total_questions']) * 100)
        st.markdown(f"""
        <div class="conflict-low">
            <h4>✅ Resolution Rate</h4>
            <h2>{resolution_rate}%</h2>
            <p>Questions with optimal UIDs</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Search and filter
    st.subheader("🔍 Search & Filter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("Search questions...", placeholder="Enter search term")
    
    with col2:
        conflict_filter = st.selectbox(
            "Filter by status",
            ["All", "Has Conflicts", "No Conflicts", "High Risk"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Usage Count", "Question", "Conflicts"]
        )
    
    # Filter data
    filtered_df = processed_df.copy()
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['question'].str.contains(search_term, case=False) |
            filtered_df['optimal_uid'].str.contains(search_term, case=False)
        ]
    
    if conflict_filter == "Has Conflicts":
        filtered_df = filtered_df[filtered_df['has_conflicts']]
    elif conflict_filter == "No Conflicts":
        filtered_df = filtered_df[~filtered_df['has_conflicts']]
    elif conflict_filter == "High Risk":
        filtered_df = filtered_df[filtered_df['high_risk']]
    
    # Sort data
    if sort_by == "Usage Count":
        filtered_df = filtered_df.sort_values('count', ascending=False)
    elif sort_by == "Question":
        filtered_df = filtered_df.sort_values('question')
    elif sort_by == "Conflicts":
        filtered_df = filtered_df.sort_values('num_conflicts', ascending=False)
    
    # Display optimized question bank
    st.subheader("📋 Optimized Question Bank")
    
    for _, row in filtered_df.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
            
            with col1:
                st.markdown(f"""
                <div style="background: #3b82f6; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: bold;">
                    {row['optimal_uid']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write(f"**{row['question']}**")
                st.write(f"Usage: {row['count']:,}")
            
            with col3:
                if row['has_conflicts']:
                    if row['high_risk']:
                        st.error("🚨 High Risk")
                    else:
                        st.warning("⚠️ Conflicts")
                else:
                    st.success("✅ Optimal")
            
            with col4:
                if row['has_conflicts']:
                    for conflict in row['conflicts']:
                        st.write(f"UID {conflict['uid']}: {conflict['count']:,}")
                else:
                    st.write("No conflicts")
            
            st.divider()

def show_survey_analysis(survey_df: pd.DataFrame):
    """Show survey analysis page"""
    
    st.header("📋 Survey Analysis by Category")
    
    # Category overview
    st.subheader("📊 Category Overview")
    create_category_overview(survey_df)
    
    # Search and filter
    st.subheader("🔍 Survey Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("Search surveys...", placeholder="Search by title or category")
    
    with col2:
        categories = ['All'] + sorted(survey_df['category'].unique())
        selected_category = st.selectbox("Filter by category", categories)
    
    # Filter surveys
    filtered_surveys = survey_df.copy()
    
    if search_term:
        filtered_surveys = filtered_surveys[
            filtered_surveys['title'].str.contains(search_term, case=False) |
            filtered_surveys['category'].str.contains(search_term, case=False)
        ]
    
    if selected_category != 'All':
        filtered_surveys = filtered_surveys[filtered_surveys['category'] == selected_category]
    
    # Survey selection
    st.subheader("📝 Select Surveys for UID Assignment")
    
    if 'selected_survey_ids' not in st.session_state:
        st.session_state.selected_survey_ids = []
    
    # Display surveys as selectable cards
    for _, survey in filtered_surveys.iterrows():
        is_selected = survey['id'] in st.session_state.selected_survey_ids
        
        card_class = "survey-card selected" if is_selected else "survey-card"
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{survey['title']}</h4>
                    <p><span style="background: #3b82f6; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8em;">{survey['category']}</span></p>
                    <p>📋 {survey['questions']} questions • 👥 {survey['responses']} responses</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"{'✅ Selected' if is_selected else '➕ Select'}", key=f"select_{survey['id']}"):
                    if is_selected:
                        st.session_state.selected_survey_ids.remove(survey['id'])
                    else:
                        st.session_state.selected_survey_ids.append(survey['id'])
                    st.rerun()
            
            with col3:
                st.button(f"👁️ Preview", key=f"preview_{survey['id']}")
    
    # Selected surveys summary
    if st.session_state.selected_survey_ids:
        st.subheader("✅ Selected Surveys")
        
        selected_surveys = filtered_surveys[filtered_surveys['id'].isin(st.session_state.selected_survey_ids)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Selected Surveys", len(selected_surveys))
            st.metric("Total Questions", selected_surveys['questions'].sum())
        
        with col2:
            st.metric("Total Responses", selected_surveys['responses'].sum())
            if st.button("🚀 Process Selected Surveys", type="primary"):
                st.success(f"Processing {len(selected_surveys)} surveys for UID assignment...")
                # Here you would trigger the UID assignment process
        
        # Category breakdown
        st.subheader("📊 Category Breakdown")
        category_breakdown = selected_surveys.groupby('category').agg({
            'id': 'count',
            'questions': 'sum',
            'responses': 'sum'
        }).reset_index()
        
        for _, cat_row in category_breakdown.iterrows():
            st.write(f"**{cat_row['category']}**: {cat_row['id']} surveys, {cat_row['questions']} questions, {cat_row['responses']} responses")

if __name__ == "__main__":
    main()