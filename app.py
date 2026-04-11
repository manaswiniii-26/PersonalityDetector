import streamlit as st
import pandas as pd
import numpy as np
import re, io, os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, .playfair { font-family: 'Playfair Display', serif !important; }

.stApp { 
    background-color: #ffffff;
    background-image: radial-gradient(#f3e7e9 1px, transparent 1px);
    background-size: 40px 40px;
}

/* HEADER SECTION */
.hero-container {
    padding: 80px 20px 40px 20px;
    text-align: center;
    background: #fffafa;
    border-bottom: 1px solid #eee;
    margin-bottom: 40px;
}
.hero-title { font-size: 3.5rem; font-weight: 700; color: var(--deep-berry); letter-spacing: -1px; margin-bottom: 10px; }
.hero-subtitle { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 3px; font-weight: 400; }

/* SECTION LABELS */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: var(--deep-berry);
    margin: 40px 0 20px;
    border-bottom: 1px solid var(--accent);
    display: inline-block;
}

/* MBTI TYPE TILES */
.mbti-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
.mbti-tile {
    padding: 15px;
    border: 1px solid #f0f0f0;
    text-align: center;
    background: white;
    transition: 0.3s;
}
.mbti-tile:hover { border-color: var(--accent); background: var(--soft-pink); }
.mbti-code { font-weight: 700; color: var(--deep-berry); font-size: 1.1rem; }
.mbti-name { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 1px; }

/* METRICS CARD */
.metric-card {
    background: white;
    padding: 20px;
    border-left: 3px solid var(--primary-rose);
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
}
.metric-val { font-family: 'Playfair Display', serif; font-size: 1.8rem; color: var(--deep-berry); }
.metric-label { font-size: 0.7rem; text-transform: uppercase; color: var(--accent); font-weight: 700; }

/* BUTTON */
div.stButton > button {
    background: var(--deep-berry) !important;
    color: white !important;
    border-radius: 0px !important;
    padding: 12px 30px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    width: 100%;
}

textarea { border-radius: 0px !important; border: 1px solid #ddd !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# DATA & MODELS
# ──────────────────────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource(show_spinner=False)
def train_aura_engine():
    FILE_ID = "1WpYslfFqGPdMHChZOrHjpcCsfehdvshN"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    df = pd.read_csv(url).sample(n=2500, random_state=42)
    
    results = {}
    for task_name, task_idx in [('IE', 0), ('TF', 2)]:
        df['target'] = df['type'].apply(lambda x: 1 if x[task_idx] in ['E', 'F'] else 0)
        vec = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
        X = vec.fit_transform(df['posts'].apply(preprocess))
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        svm = SVC(kernel='linear', probability=True).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        
        results[task_name] = {
            'vec': vec, 'svm': svm, 'rf': rf,
            'svm_acc': accuracy_score(y_test, svm.predict(X_test)),
            'rf_acc': accuracy_score(y_test, rf.predict(X_test))
        }
    return results

# ──────────────────────────────────────────────────────
# UI CONTENT
# ──────────────────────────────────────────────────────

with st.spinner("Refining the analytical engine..."):
    engine = train_aura_engine()

st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">Personality Analytics</div>
        <div class="hero-title">Aura AI</div>
    </div>
""", unsafe_allow_html=True)

# --- SECTION 1: SYSTEM PERFORMANCE ---
st.markdown('<div class="section-label">Model Intelligence</div>', unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">SVM Accuracy (I/E)</div><div class="metric-val">{engine["IE"]["svm_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">RF Accuracy (I/E)</div><div class="metric-val">{engine["IE"]["rf_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">SVM Accuracy (T/F)</div><div class="metric-val">{engine["TF"]["svm_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">RF Accuracy (T/F)</div><div class="metric-val">{engine["TF"]["rf_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)

# --- SECTION 2: THE PREDICTOR ---
st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-label">The Input</div>', unsafe_allow_html=True)
    user_text = st.text_area("Share a sample of your writing...", height=250, placeholder="Describe your philosophy or a recent memory...")
    model_choice = st.radio("Intelligence Architecture", ["Support Vector Machine (SVM)", "Random Forest (RF)"], horizontal=True)
    predict_btn = st.button("Unveil Aura")

with right_col:
    st.markdown('<div class="section-label">The Analysis</div>', unsafe_allow_html=True)
    if predict_btn and user_text:
        model_key = 'svm' if "SVM" in model_choice else 'rf'
        
        final_results = {}
        for task in ['IE', 'TF']:
            vec = engine[task]['vec']
            clf = engine[task][model_key]
            prob = clf.predict_proba(vec.transform([preprocess(user_text)]))[0]
            final_results[task] = prob

        # Labels
        ie_type = "Extroverted" if final_results['IE'][1] > 0.5 else "Introverted"
        tf_type = "Feeling" if final_results['TF'][1] > 0.5 else "Thinking"
        
        st.markdown(f"""
            <div style="margin-bottom: 30px;">
                <div style="font-size: 0.8rem; color: var(--accent); font-weight: 700; text-transform: uppercase;">Primary Orientation</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 2.8rem; color: var(--deep-berry);">{ie_type}</div>
                <p style="color: #666;">Confidence Level: {max(final_results['IE'])*100:.1f}%</p>
            </div>
            <div>
                <div style="font-size: 0.8rem; color: var(--accent); font-weight: 700; text-transform: uppercase;">Cognitive Style</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 2.8rem; color: var(--deep-berry);">{tf_type}</div>
                <p style="color: #666;">Confidence Level: {max(final_results['TF'])*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting input for cognitive mapping...")

# --- SECTION 3: MBTI DIRECTORY ---
st.markdown('<div class="section-label">The 16 Archetypes</div>', unsafe_allow_html=True)
mbti_types = {
    "INTJ": "Architect", "INTP": "Logician", "ENTJ": "Commander", "ENTP": "Debater",
    "INFJ": "Advocate", "INFP": "Mediator", "ENFJ": "Protagonist", "ENFP": "Campaigner",
    "ISTJ": "Logistician", "ISFJ": "Defender", "ESTJ": "Executive", "ESFJ": "Consul",
    "ISTP": "Virtuoso", "ISFP": "Adventurer", "ESTP": "Entrepreneur", "ESFP": "Entertainer"
}

st.markdown('<div class="mbti-grid">', unsafe_allow_html=True)
for code, name in mbti_types.items():
    st.markdown(f'<div class="mbti-tile"><div class="mbti-code">{code}</div><div class="mbti-name">{name}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
