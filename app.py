import streamlit as st
import pandas as pd
import numpy as np
import re, io, os
import requests
import joblib  # For saving/loading models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aura · Personality Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────
# REFINED ELEGANCE CSS
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');

:root {
    --primary-rose: #d4a373;
    --soft-pink: #f9f1f0;
    --deep-berry: #4a3035;
    --accent: #e19d9d;
}

*, html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
}

h1, h2, h3, .playfair { 
    font-family: 'Playfair Display', serif !important; 
}

.stApp { 
    background-color: #ffffff;
    background-image: radial-gradient(#f3e7e9 1px, transparent 1px);
    background-size: 40px 40px;
}

/* HEADER SECTION */
.hero-container {
    padding: 100px 20px 60px 20px;
    text-align: center;
    background: #fffafa;
    border-bottom: 1px solid #eee;
    margin-bottom: 50px;
}

.hero-title {
    font-size: 4rem;
    font-weight: 700;
    color: var(--deep-berry);
    letter-spacing: -1px;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 400;
}

/* CARD SYSTEM */
.glass-card {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 0px; /* Sharp edges for a more modern, architectural look */
    padding: 40px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.03);
    margin-bottom: 20px;
}

.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--deep-berry);
    margin-bottom: 30px;
    border-bottom: 1px solid var(--accent);
    display: inline-block;
}

/* BUTTON */
div.stButton > button {
    background: var(--deep-berry) !important;
    color: white !important;
    border-radius: 0px !important;
    padding: 15px 40px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    transition: 0.4s !important;
    width: 100%;
}

div.stButton > button:hover {
    background: var(--primary-rose) !important;
    box-shadow: 0 5px 15px rgba(212, 163, 115, 0.3) !important;
}

/* RESULTS */
.result-box {
    border-left: 2px solid var(--primary-rose);
    padding-left: 25px;
    margin: 20px 0;
}

.result-type {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    color: var(--deep-berry);
}

.result-conf {
    font-size: 0.8rem;
    color: var(--accent);
    font-weight: 700;
    text-transform: uppercase;
}

textarea {
    border-radius: 0px !important;
    border: 1px solid #ddd !important;
    padding: 20px !important;
}

/* Removing Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource(show_spinner=False)
def get_ready_models():
    # To maximize speed, we use a fixed seed and a optimized TF-IDF
    FILE_ID = "1WpYslfFqGPdMHChZOrHjpcCsfehdvshN"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    
    # Simple request to get data
    df = pd.read_csv(url)
    df = df.sample(n=2000, random_state=42) # Faster training for UI smoothness
    
    models = {}
    for task_name, task_col in [('IE', 'I'), ('TF', 'T')]:
        df['target'] = df['type'].apply(lambda x: 1 if x[0 if task_name=='IE' else 2] == task_col else 0)
        
        # Improved Vectorizer: Stop Words help focus on "Personality signals"
        vec = TfidfVectorizer(stop_words='english', max_features=2500, ngram_range=(1, 2))
        X = vec.fit_transform(df['posts'].apply(preprocess))
        y = df['target']
        
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X, y)
        models[task_name] = (vec, clf)
    return models

# ──────────────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────────────

# Load models silently in the background
models = get_ready_models()

st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">Sophisticated Personality Insights</div>
        <div class="hero-title">Aura Analytics</div>
    </div>
""", unsafe_allow_html=True)

main_col_left, space, main_col_right = st.columns([1, 0.1, 1])

with main_col_left:
    st.markdown('<div class="section-label">The Input</div>', unsafe_allow_html=True)
    st.markdown("Please provide a sample of writing (at least 50 words) that reflects your natural thought process.")
    
    user_input = st.text_area("", placeholder="Reflect on your day, your goals, or what drives you...", height=300)
    
    analyze_btn = st.button("Unveil My Aura")

with main_col_right:
    st.markdown('<div class="section-label">The Analysis</div>', unsafe_allow_html=True)
    
    if analyze_btn:
        if len(user_input.split()) < 10:
            st.info("The depth of our analysis requires a bit more text.")
        else:
            # Prediction Logic
            results = {}
            for task in ['IE', 'TF']:
                vec, clf = models[task]
                processed = preprocess(user_input)
                vec_input = vec.transform([processed])
                prob = clf.predict_proba(vec_input)[0]
                results[task] = prob

            # Interpretation
            ie_res = "Introverted" if results['IE'][1] > 0.5 else "Extroverted"
            tf_res = "Thinking" if results['TF'][1] > 0.5 else "Feeling"
            
            st.markdown(f"""
                <div class="result-box">
                    <div class="result-conf">Orientation</div>
                    <div class="result-type">{ie_res}</div>
                    <p>Analysis suggests a {max(results['IE'])*100:.1f}% stylistic alignment.</p>
                </div>
                <div class="result-box">
                    <div class="result-conf">Decision Matrix</div>
                    <div class="result-type">{tf_res}</div>
                    <p>Analysis suggests a {max(results['TF'])*100:.1f}% stylistic alignment.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("Analysis Complete.")
    else:
        st.markdown("""
            <div style="color: #bbb; margin-top: 50px; text-align: center;">
                <i>Waiting for input...</i>
            </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
