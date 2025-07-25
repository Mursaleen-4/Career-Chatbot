"""
🚀 AI-Powered Career Oracle - Next-Gen Career Guidance Platform
An innovative, AI-driven career discovery system with advanced analytics,
visual insights, and personalized recommendations.
"""

import streamlit as st
import joblib
import pandas as pd
import string
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import base64
from io import BytesIO
from nltk.stem import WordNetLemmatizer
import nltk
import json
from wordcloud import WordCloud
from collections import Counter
import re

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CareerChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.vectorizer = None
        self.dataset = None
        
    def load_model_and_data(self):
        """Load the trained model, vectorizer, and dataset"""
        try:
            self.model = joblib.load('intent_model.pkl')
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.dataset = pd.read_csv('career_guidance_dataset.csv')
            return True
        except FileNotFoundError as e:
            st.error(f"Error loading model files: {e}")
            st.error("Please run train_model.py first to train the model.")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text same as training"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
        
        return text
    
    def predict_career_role(self, question):
        """Predict career role and get relevant information"""
        processed_question = self.preprocess_text(question)
        question_vectorized = self.vectorizer.transform([processed_question])
        
        # Get prediction and probabilities
        prediction = self.model.predict(question_vectorized)[0]
        probabilities = self.model.predict_proba(question_vectorized)[0]
        confidence = probabilities.max()
        
        # Get top 3 predictions
        top_indices = probabilities.argsort()[-3:][::-1]
        top_roles = [self.model.classes_[i] for i in top_indices]
        top_confidences = [probabilities[i] for i in top_indices]
        
        return prediction, confidence, top_roles, top_confidences
    
    def get_role_information(self, role):
        """Get sample answers for a specific role"""
        # Handle both column name formats
        role_col = 'Role' if 'Role' in self.dataset.columns else 'role'
        answer_col = 'Answer' if 'Answer' in self.dataset.columns else 'answer'
        
        role_data = self.dataset[self.dataset[role_col] == role]
        if not role_data.empty:
            return role_data[answer_col].iloc[0]  # Return first answer for the role
        return "No specific information available for this role."

# Custom CSS for futuristic styling
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 20px 60px rgba(118, 75, 162, 0.4); }
    }
    
    .oracle-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00f5ff, #ff00f5, #f5ff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,245,255,0.5);
        margin-bottom: 1rem;
        animation: textShine 3s infinite;
    }
    
    @keyframes textShine {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        color: #ffffff;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    .neural-bg {
        background: linear-gradient(45deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(0,245,255,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .neural-bg::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,245,255,0.1), transparent);
        animation: slide 3s infinite;
    }
    
    @keyframes slide {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255,255,255,0.2);
        transform: scale(1.05);
    }
    
    .chat-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #00f5ff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .career-path {
        background: rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .career-path:hover {
        background: rgba(102,126,234,0.1);
        transform: translateX(5px);
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        background: #00f5ff;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(0.8); opacity: 1; }
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(0,245,255,0.3) !important;
        border-radius: 25px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 15px 20px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00f5ff !important;
        box-shadow: 0 0 20px rgba(0,245,255,0.3) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3) !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .floating-particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(0,245,255,0.6);
        border-radius: 50%;
        animation: float 6s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .career-radar {
        background: rgba(0,0,0,0.02);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(102,126,234,0.2);
    }
    
    .typing-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #00f5ff;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .success-animation {
        animation: successPulse 0.6s ease-out;
    }
    
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

# Generate AI insights based on career predictions
def generate_ai_insights(prediction, confidence, user_question):
    insights = []
    
    # Confidence-based insights
    if confidence > 0.8:
        insights.append(f"🎯 **Strong Match Detected**: Your interests align exceptionally well with {prediction}. This suggests a natural fit for this career path.")
    elif confidence > 0.6:
        insights.append(f"✨ **Good Alignment**: {prediction} shows promising compatibility with your expressed interests.")
    else:
        insights.append(f"🔍 **Exploratory Match**: While {prediction} is suggested, consider providing more specific details about your interests for better accuracy.")
    
    # Interest-based insights
    keywords = user_question.lower().split()
    if any(word in keywords for word in ['data', 'analysis', 'analytics', 'statistics']):
        insights.append("📊 **Data-Driven Profile**: Your interest in data suggests strong analytical thinking - valuable in today's data-driven economy.")
    
    if any(word in keywords for word in ['creative', 'design', 'art', 'visual']):
        insights.append("🎨 **Creative Mindset**: Your creative inclinations indicate strong right-brain thinking - essential for innovation.")
    
    if any(word in keywords for word in ['people', 'help', 'team', 'social']):
        insights.append("👥 **People-Oriented**: Your focus on human interaction suggests strong interpersonal skills - highly valued across industries.")
    
    if any(word in keywords for word in ['technology', 'tech', 'programming', 'software']):
        insights.append("💻 **Technology Enthusiast**: Your tech interests position you well in the rapidly growing digital economy.")
    
    return insights

# Generate career development roadmap
def generate_career_roadmap(prediction):
    roadmaps = {
        'Data Scientist': {
            'skills': ['Python/R Programming', 'Machine Learning', 'Statistics', 'Data Visualization', 'SQL'],
            'timeline': ['Learn Programming (3-6 months)', 'Master Statistics (6-9 months)', 'Build ML Projects (9-12 months)', 'Gain Industry Experience (1-2 years)'],
            'salary_range': '$70K - $150K+',
            'growth_rate': '25% (Much faster than average)'
        },
        'Software Engineer': {
            'skills': ['Programming Languages', 'System Design', 'Algorithms', 'Version Control', 'Testing'],
            'timeline': ['Master Programming (6 months)', 'Learn Frameworks (6-9 months)', 'Build Portfolio (9-12 months)', 'Industry Experience (1-2 years)'],
            'salary_range': '$60K - $140K+',
            'growth_rate': '22% (Much faster than average)'
        },
        'UX/UI Designer': {
            'skills': ['Design Thinking', 'Prototyping', 'User Research', 'Design Tools', 'Psychology'],
            'timeline': ['Learn Design Principles (3 months)', 'Master Tools (6 months)', 'Build Portfolio (9 months)', 'Gain Experience (1-2 years)'],
            'salary_range': '$50K - $120K+',
            'growth_rate': '18% (Much faster than average)'
        }
    }
    
    return roadmaps.get(prediction, {
        'skills': ['Industry-specific skills', 'Communication', 'Problem-solving', 'Continuous Learning'],
        'timeline': ['Foundation Building (6 months)', 'Skill Development (1 year)', 'Experience Gaining (2 years)'],
        'salary_range': 'Varies by industry',
        'growth_rate': 'Industry dependent'
    })

# Create interactive visualizations
def create_confidence_radar(top_roles, top_confidences):
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=top_confidences,
        theta=top_roles,
        fill='toself',
        name='Confidence Scores',
        line_color='rgba(0,245,255,0.8)',
        fillcolor='rgba(0,245,255,0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%'
            )
        ),
        showlegend=False,
        title="Career Match Confidence Radar",
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_career_timeline_viz(roadmap):
    fig = go.Figure()
    
    timeline = roadmap['timeline']
    y_pos = list(range(len(timeline)))
    
    fig.add_trace(go.Scatter(
        x=[i*3 for i in range(len(timeline))],
        y=y_pos,
        mode='markers+lines+text',
        marker=dict(size=15, color='rgba(0,245,255,0.8)'),
        line=dict(color='rgba(102,126,234,0.8)', width=3),
        text=timeline,
        textposition='right',
        name='Career Timeline'
    ))
    
    fig.update_layout(
        title="Your Career Development Roadmap",
        xaxis_title="Timeline (Months)",
        yaxis=dict(showticklabels=False),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="🚀 AI Career Oracle",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Floating particles background
    st.markdown("""
    <div class="floating-particles">
        <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
        <div class="particle" style="left: 20%; animation-delay: 1s;"></div>
        <div class="particle" style="left: 30%; animation-delay: 2s;"></div>
        <div class="particle" style="left: 40%; animation-delay: 3s;"></div>
        <div class="particle" style="left: 50%; animation-delay: 4s;"></div>
        <div class="particle" style="left: 60%; animation-delay: 5s;"></div>
        <div class="particle" style="left: 70%; animation-delay: 1.5s;"></div>
        <div class="particle" style="left: 80%; animation-delay: 2.5s;"></div>
        <div class="particle" style="left: 90%; animation-delay: 3.5s;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Revolutionary header
    st.markdown("""
    <div class="main-header">
        <h1 class="oracle-title">🔮 AI CAREER ORACLE</h1>
        <p class="subtitle">Next-Generation Career Intelligence Platform</p>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">Powered by Advanced Machine Learning & Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = CareerChatbot()
    
    # Load model and data with progress
    with st.spinner("🧠 Initializing AI Oracle..."):
        if not chatbot.load_model_and_data():
            st.stop()
        time.sleep(0.5)  # Dramatic effect
        st.success("✨ AI Oracle Ready!")
    
    # Initialize session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'interests': [],
            'skill_areas': [],
            'career_exploration_count': 0,
            'dominant_traits': []
        }
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'Standard'
    
    # Futuristic sidebar
    with st.sidebar:
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                🧠 NEURAL COMMAND CENTER
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis mode selector
        st.session_state.analysis_mode = st.selectbox(
            "🔬 Analysis Mode",
            ["Standard", "Deep Insight", "Career Path Explorer", "Skill Matcher"]
        )
        
        # Live stats
        st.markdown("""
        <div class="neural-bg">
            <h3 style="color: #00f5ff;">⚡ Live Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        role_col = 'Role' if 'Role' in chatbot.dataset.columns else 'role'
        total_careers = chatbot.dataset[role_col].nunique()
        total_questions = len(chatbot.dataset)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Careers", total_careers, delta="Active")
        with col2:
            st.metric("💭 Queries", total_questions, delta="Trained")
        
        st.metric("🔥 Sessions", len(st.session_state.chat_history), delta="Current")
        
        # User profile insights
        if st.session_state.user_profile['career_exploration_count'] > 0:
            st.markdown("""
            <div class="neural-bg">
                <h3 style="color: #00f5ff;">👤 Your Profile</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"🔍 **Explorations:** {st.session_state.user_profile['career_exploration_count']}")
            if st.session_state.user_profile['dominant_traits']:
                st.write(f"🧬 **Traits:** {', '.join(st.session_state.user_profile['dominant_traits'][:3])}")
        
        # Quick start examples
        st.markdown("""
        <div class="neural-bg">
            <h3 style="color: #00f5ff;">🚀 Quick Start</h3>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "I love analyzing data and finding patterns",
            "I want to design beautiful user interfaces",
            "I'm passionate about cybersecurity and protecting systems",
            "I enjoy building mobile applications",
            "I want to work with artificial intelligence"
        ]
        
        for i, example in enumerate(example_questions[:3]):
            if st.button(f"💡 {example[:30]}...", key=f"example_{i}"):
                st.session_state.selected_example = example
    
    # Main interface with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Oracle Chat", "📊 Analytics Dashboard", "🛤️ Career Roadmap", "🎯 Skill Matcher"])
    
    with tab1:
        # AI Oracle Chat Interface
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                <span class="pulse-dot"></span>ASK THE ORACLE
            </h2>
            <p style="text-align: center; color: rgba(255,255,255,0.8);">Speak your career desires, and let AI reveal your destiny...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced input with voice-like interface
        user_question = st.text_area(
            "🎤 Describe your career interests, passions, or goals:",
            placeholder="Example: I'm fascinated by data science and want to use machine learning to solve real-world problems. I enjoy programming in Python and have a strong mathematical background...",
            height=120,
            key="main_input"
        )
        
        # Use example if selected
        if 'selected_example' in st.session_state:
            user_question = st.session_state.selected_example
            del st.session_state.selected_example
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 CONSULT THE ORACLE", type="primary", use_container_width=True):
                if user_question:
                    # Dramatic processing sequence
                    with st.spinner("🧠 Neural networks analyzing..."):
                        time.sleep(0.5)
                    
                    with st.spinner("🔍 Pattern recognition in progress..."):
                        time.sleep(0.3)
                    
                    with st.spinner("✨ Generating insights..."):
                        prediction, confidence, top_roles, top_confidences = chatbot.predict_career_role(user_question)
                        time.sleep(0.2)
                    
                    # Update user profile
                    st.session_state.user_profile['career_exploration_count'] += 1
                    
                    # Analyze traits from question
                    keywords = user_question.lower().split()
                    if any(word in keywords for word in ['data', 'analysis', 'statistics']):
                        st.session_state.user_profile['dominant_traits'].append('Analytical')
                    if any(word in keywords for word in ['creative', 'design', 'art']):
                        st.session_state.user_profile['dominant_traits'].append('Creative')
                    if any(word in keywords for word in ['people', 'team', 'social']):
                        st.session_state.user_profile['dominant_traits'].append('Social')
                    if any(word in keywords for word in ['technology', 'programming', 'software']):
                        st.session_state.user_profile['dominant_traits'].append('Technical')
                    
                    # Remove duplicates and limit to 5 traits
                    st.session_state.user_profile['dominant_traits'] = list(set(st.session_state.user_profile['dominant_traits']))[-5:]
                    
                    # Add to chat history with timestamp
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now(),
                        'question': user_question,
                        'prediction': prediction,
                        'confidence': confidence,
                        'top_roles': top_roles,
                        'top_confidences': top_confidences,
                        'analysis_mode': st.session_state.analysis_mode
                    })
                    
                    # Success animation
                    st.success("🎉 Oracle Consultation Complete!")
                    
                    # Main prediction card with enhanced styling
                    st.markdown(f"""
                    <div class="prediction-card success-animation">
                        <h2 style="margin: 0; color: white; font-family: 'Orbitron', monospace;">
                            🎯 PRIMARY DESTINY: {prediction}
                        </h2>
                        <div style="margin-top: 1rem;">
                            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                                <h4 style="margin: 0; color: #00f5ff;">Oracle Confidence</h4>
                                <div style="font-size: 2rem; font-weight: bold;">{confidence:.1%}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Career information
                    role_info = chatbot.get_role_information(prediction)
                    st.markdown(f"""
                    <div class="career-path">
                        <h4>📖 Career Insight</h4>
                        <p>{role_info}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interactive confidence radar
                    st.markdown("### 🎯 Career Match Analysis")
                    radar_fig = create_confidence_radar(top_roles, top_confidences)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Top suggestions with enhanced cards
                    st.markdown("### 🚀 Alternative Career Paths")
                    cols = st.columns(3)
                    for i, (role, conf) in enumerate(zip(top_roles, top_confidences)):
                        with cols[i]:
                            rank_emoji = ["🥇", "🥈", "🥉"][i]
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin: 0; color: #00f5ff;">{rank_emoji} Rank {i+1}</h4>
                                <h3 style="margin: 0.5rem 0; color: white;">{role}</h3>
                                <div style="font-size: 1.5rem; font-weight: bold; color: #00f5ff;">{conf:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # AI Insights based on analysis mode
                    insights = generate_ai_insights(prediction, confidence, user_question)
                    if st.session_state.analysis_mode == "Deep Insight":
                        st.markdown("### 🧠 Deep AI Analysis")
                        for insight in insights:
                            st.markdown(f"""
                            <div class="ai-insight">
                                {insight}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Career roadmap for primary prediction
                    if st.session_state.analysis_mode == "Career Path Explorer":
                        roadmap = generate_career_roadmap(prediction)
                        st.markdown("### 🛤️ Your Career Development Path")
                        
                        # Timeline visualization
                        timeline_fig = create_career_timeline_viz(roadmap)
                        st.plotly_chart(timeline_fig, use_container_width=True)
                        
                        # Skills and stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 🎯 Key Skills to Develop")
                            for skill in roadmap['skills']:
                                st.markdown(f"<div class='career-path'>• {skill}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 📈 Career Stats")
                            st.metric("💰 Salary Range", roadmap['salary_range'])
                            st.metric("📊 Growth Rate", roadmap['growth_rate'])
                
                else:
                    st.warning("🔮 The Oracle requires your question to provide guidance...")
    
    with tab2:
        # Analytics Dashboard
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                📊 CAREER ANALYTICS NEXUS
            </h2>
            <p style="text-align: center; color: rgba(255,255,255,0.8);">Real-time insights into career trends and your exploration journey</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            # User's career exploration analytics
            st.markdown("### 🔍 Your Career Exploration Journey")
            
            # Create exploration timeline
            exploration_data = []
            for chat in st.session_state.chat_history:
                exploration_data.append({
                    'timestamp': chat['timestamp'],
                    'career': chat['prediction'],
                    'confidence': chat['confidence']
                })
            
            df_exploration = pd.DataFrame(exploration_data)
            
            # Confidence over time chart
            fig_timeline = px.line(df_exploration, x='timestamp', y='confidence', 
                                 title='Career Match Confidence Over Time',
                                 markers=True)
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Career interest distribution
            career_counts = df_exploration['career'].value_counts()
            fig_pie = px.pie(values=career_counts.values, names=career_counts.index,
                           title='Your Career Interest Distribution')
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_confidence = df_exploration['confidence'].mean()
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
            with col2:
                unique_careers = df_exploration['career'].nunique()
                st.metric("🎆 Careers Explored", unique_careers)
            with col3:
                best_match = df_exploration.loc[df_exploration['confidence'].idxmax(), 'career']
                st.metric("🌅 Best Match", best_match)
            with col4:
                exploration_span = (df_exploration['timestamp'].max() - df_exploration['timestamp'].min()).days
                st.metric("⏱️ Days Active", max(1, exploration_span))
        
        else:
            st.info("🚀 Start exploring careers to see your personalized analytics dashboard!")
        
        # Global career trends (simulated data for demonstration)
        st.markdown("### 🌍 Global Career Trends")
        
        # Create simulated trending careers data
        trending_careers = {
            'AI/ML Engineer': {'trend': 95, 'growth': '+45%'},
            'Data Scientist': {'trend': 92, 'growth': '+38%'},
            'Cybersecurity Analyst': {'trend': 88, 'growth': '+31%'},
            'UX/UI Designer': {'trend': 85, 'growth': '+28%'},
            'Cloud Architect': {'trend': 82, 'growth': '+25%'}
        }
        
        cols = st.columns(len(trending_careers))
        for i, (career, data) in enumerate(trending_careers.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #00f5ff;">{career}</h4>
                    <div style="font-size: 1.2rem; color: white; margin: 0.5rem 0;">Trend Score: {data['trend']}</div>
                    <div style="color: #4CAF50; font-weight: bold;">{data['growth']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        # Career Roadmap Builder
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                🛤️ CAREER ROADMAP ARCHITECT
            </h2>
            <p style="text-align: center; color: rgba(255,255,255,0.8);">Build your personalized journey to career success</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Career selection for roadmap
        available_careers = ['Data Scientist', 'Software Engineer', 'UX/UI Designer', 'Machine Learning Engineer', 
                           'Cybersecurity Analyst', 'Product Manager', 'Business Analyst']
        
        selected_career = st.selectbox("🎯 Select a career to explore:", available_careers)
        
        if st.button("🚀 Generate Roadmap", type="primary"):
            roadmap = generate_career_roadmap(selected_career)
            
            # Advanced roadmap visualization
            st.markdown(f"### 🎆 Your Path to Becoming a {selected_career}")
            
            # Create interactive timeline
            timeline_fig = create_career_timeline_viz(roadmap)
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="career-radar">
                    <h3>🎯 Essential Skills Matrix</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for i, skill in enumerate(roadmap['skills']):
                    progress = 90 - (i * 15)  # Simulated importance
                    st.markdown(f"""
                    <div class="career-path">
                        <strong>{skill}</strong>
                        <div style="background: rgba(0,245,255,0.2); height: 8px; border-radius: 4px; margin-top: 5px;">
                            <div style="background: #00f5ff; height: 100%; width: {progress}%; border-radius: 4px;"></div>
                        </div>
                        <small>Importance: {progress}%</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="career-radar">
                    <h3>📊 Career Statistics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("💰 Salary Range", roadmap['salary_range'])
                st.metric("📈 Job Growth", roadmap['growth_rate'])
                st.metric("⏱️ Time to Master", "12-24 months")
                st.metric("🌟 Difficulty Level", "Intermediate")
    
    with tab4:
        # Skill Matcher
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                🎯 NEURAL SKILL MATCHER
            </h2>
            <p style="text-align: center; color: rgba(255,255,255,0.8);">Match your skills to perfect career opportunities</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Skill assessment interface
        st.markdown("### 🧠 Skill Assessment Matrix")
        
        skill_categories = {
            'Technical Skills': ['Programming', 'Data Analysis', 'System Design', 'Database Management', 'Cloud Computing'],
            'Creative Skills': ['UI/UX Design', 'Graphic Design', 'Creative Writing', 'Video Editing', 'Brand Strategy'],
            'Analytical Skills': ['Problem Solving', 'Statistical Analysis', 'Research', 'Critical Thinking', 'Pattern Recognition'],
            'Interpersonal Skills': ['Communication', 'Leadership', 'Team Collaboration', 'Negotiation', 'Mentoring']
        }
        
        user_skills = {}
        
        for category, skills in skill_categories.items():
            st.markdown(f"#### {category}")
            cols = st.columns(len(skills))
            
            for i, skill in enumerate(skills):
                with cols[i]:
                    rating = st.slider(f"{skill}", 0, 10, 5, key=f"{category}_{skill}")
                    user_skills[skill] = rating
        
        if st.button("🔮 ANALYZE SKILL PROFILE", type="primary"):
            # Create skill radar chart
            categories = list(skill_categories.keys())
            category_scores = []
            
            for category, skills in skill_categories.items():
                avg_score = np.mean([user_skills[skill] for skill in skills])
                category_scores.append(avg_score)
            
            # Skill radar visualization
            fig_skills = go.Figure()
            
            fig_skills.add_trace(go.Scatterpolar(
                r=category_scores,
                theta=categories,
                fill='toself',
                name='Your Skills',
                line_color='rgba(255,0,255,0.8)',
                fillcolor='rgba(255,0,255,0.2)'
            ))
            
            fig_skills.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=False,
                title="Your Skill Profile Radar",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_skills, use_container_width=True)
            
            # Career recommendations based on skills
            st.markdown("### 🎆 Skill-Based Career Recommendations")
            
            # Simple skill-to-career mapping logic
            tech_score = np.mean([user_skills[skill] for skill in skill_categories['Technical Skills']])
            creative_score = np.mean([user_skills[skill] for skill in skill_categories['Creative Skills']])
            analytical_score = np.mean([user_skills[skill] for skill in skill_categories['Analytical Skills']])
            interpersonal_score = np.mean([user_skills[skill] for skill in skill_categories['Interpersonal Skills']])
            
            recommendations = []
            
            if tech_score >= 7:
                recommendations.append(('Software Engineer', tech_score * 10, '💻'))
            if analytical_score >= 7:
                recommendations.append(('Data Scientist', analytical_score * 10, '📊'))
            if creative_score >= 7:
                recommendations.append(('UX/UI Designer', creative_score * 10, '🎨'))
            if interpersonal_score >= 7:
                recommendations.append(('Product Manager', interpersonal_score * 10, '💼'))
            
            if not recommendations:
                recommendations = [('Business Analyst', 75, '📊'), ('Project Manager', 70, '📋')]
            
            cols = st.columns(min(3, len(recommendations)))
            for i, (career, match_score, emoji) in enumerate(recommendations[:3]):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; color: #00f5ff;">{emoji} {career}</h3>
                        <div style="font-size: 1.5rem; font-weight: bold; color: white; margin: 1rem 0;">{match_score:.0f}% Match</div>
                        <div style="background: rgba(0,245,255,0.2); height: 8px; border-radius: 4px;">
                            <div style="background: #00f5ff; height: 100%; width: {match_score}%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Enhanced chat history with visual timeline
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("""
        <div class="neural-bg">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center;">
                📜 ORACLE CONSULTATION HISTORY
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            timestamp = chat['timestamp'].strftime("%Y-%m-%d %H:%M")
            with st.expander(f"🔮 Consultation {len(st.session_state.chat_history) - i}: {chat['question'][:50]}... ({timestamp})"):
                st.markdown(f"""
                <div class="chat-bubble">
                    <strong>💬 Question:</strong> {chat['question']}<br>
                    <strong>🎯 Oracle's Verdict:</strong> {chat['prediction']}<br>
                    <strong>✨ Confidence:</strong> {chat['confidence']:.1%}<br>
                    <strong>🔬 Mode:</strong> {chat['analysis_mode']}
                </div>
                """, unsafe_allow_html=True)
                
                # Mini radar for each consultation
                mini_radar = create_confidence_radar(chat['top_roles'], chat['top_confidences'])
                mini_radar.update_layout(height=300)
                st.plotly_chart(mini_radar, use_container_width=True)
        
        # Clear history with confirmation
        if st.button("🗑️ Clear All Consultation History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.user_profile = {
                'interests': [],
                'skill_areas': [],
                'career_exploration_count': 0,
                'dominant_traits': []
            }
            st.rerun()
    
    # Revolutionary footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 2rem;'>
        <h3 style='color: white; font-family: "Orbitron", monospace; margin: 0;'>🚀 AI CAREER ORACLE</h3>
        <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0;'>Powered by Advanced Machine Learning & Neural Networks</p>
        <p style='color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;'>Built with Streamlit, scikit-learn, Plotly & NLTK</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
