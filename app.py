"""
Personality Prediction from Text using Machine Learning
========================================================
A simple Streamlit web app that predicts MBTI personality traits
(Introvert/Extrovert and Thinking/Feeling) from user-provided text.

Models Used: SVM and Random Forest (no deep learning!)
Dataset: MBTI Personality Dataset (Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Soft pastel aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import elegant Google Font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #fff0f5 0%, #f0f8ff 50%, #fff5fb 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit default elements ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #FFB6C1 0%, #ADD8E6 100%);
        border-radius: 24px;
        padding: 2.5rem 2rem 2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255,182,193,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 180px; height: 180px;
        background: rgba(255,255,255,0.18);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -50px; left: -30px;
        width: 140px; height: 140px;
        background: rgba(255,255,255,0.12);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.3rem;
        color: #fff;
        text-shadow: 0 2px 12px rgba(0,0,0,0.12);
        margin: 0 0 0.5rem 0;
        position: relative; z-index: 1;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.92);
        font-weight: 300;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
        position: relative; z-index: 1;
    }
    .hero-emoji {
        font-size: 2.5rem;
        margin-bottom: 0.6rem;
        display: block;
        position: relative; z-index: 1;
    }

    /* ── Soft card ── */
    .soft-card {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        box-shadow: 0 4px 24px rgba(200,180,210,0.15);
        border: 1px solid rgba(255,192,203,0.25);
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #b06080;
        margin: 0 0 0.8rem 0;
        letter-spacing: 0.02em;
    }

    /* ── Predict button styling ── */
    div.stButton > button {
        background: linear-gradient(135deg, #FFB6C1 0%, #f4a7c3 100%);
        color: #5a2a3a;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-radius: 50px;
        padding: 0.7rem 2.5rem;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(255,182,193,0.4);
        width: 100%;
        letter-spacing: 0.03em;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #f4a7c3 0%, #ADD8E6 100%);
        box-shadow: 0 6px 22px rgba(173,216,230,0.4);
        transform: translateY(-2px);
        color: #1a3a5c;
    }
    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Text area ── */
    textarea {
        border-radius: 14px !important;
        border: 1.5px solid #ffc0cb !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.97rem !important;
        background: rgba(255,255,255,0.9) !important;
        color: #3a2a3a !important;
        padding: 0.8rem !important;
        transition: border 0.2s ease !important;
    }
    textarea:focus {
        border: 1.5px solid #ADD8E6 !important;
        box-shadow: 0 0 0 3px rgba(173,216,230,0.2) !important;
    }

    /* ── Result cards ── */
    .result-card {
        background: linear-gradient(135deg, #fff0f5 0%, #f0f8ff 100%);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin: 0.7rem 0;
        border-left: 5px solid #FFC0CB;
        box-shadow: 0 3px 16px rgba(255,192,203,0.18);
    }
    .result-card.blue {
        border-left-color: #ADD8E6;
    }
    .result-trait {
        font-family: 'DM Serif Display', serif;
        font-size: 1.55rem;
        color: #6a3050;
        margin: 0 0 0.25rem 0;
    }
    .result-trait.blue-text {
        color: #2a5070;
    }
    .result-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #b07090;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }
    .result-label.blue-label {
        color: #5090b0;
    }
    .result-desc {
        font-size: 0.92rem;
        color: #5a4050;
        line-height: 1.6;
        margin: 0.4rem 0 0 0;
    }
    .confidence-pill {
        display: inline-block;
        background: rgba(255,192,203,0.3);
        border-radius: 50px;
        padding: 0.2rem 0.85rem;
        font-size: 0.82rem;
        color: #8a3050;
        font-weight: 600;
        margin-top: 0.4rem;
        border: 1px solid rgba(255,192,203,0.5);
    }
    .confidence-pill.blue-pill {
        background: rgba(173,216,230,0.3);
        color: #1a5070;
        border-color: rgba(173,216,230,0.5);
    }

    /* ── Summary banner ── */
    .summary-box {
        background: linear-gradient(135deg, #FFC0CB 0%, #ADD8E6 100%);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1.2rem 0;
        box-shadow: 0 4px 20px rgba(200,180,210,0.2);
    }
    .summary-box p {
        color: #fff;
        font-size: 1.08rem;
        line-height: 1.7;
        margin: 0;
        font-weight: 400;
        text-shadow: 0 1px 6px rgba(0,0,0,0.1);
    }
    .summary-box strong {
        font-weight: 600;
    }

    /* ── Metric chips ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.6rem;
    }
    .metric-chip {
        flex: 1;
        min-width: 120px;
        background: rgba(255,255,255,0.9);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,192,203,0.35);
        box-shadow: 0 2px 8px rgba(200,180,210,0.12);
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #b06080;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9a7090;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.2rem;
        color: #8a4060;
        margin: 1.8rem 0 0.8rem 0;
        text-align: center;
    }

    /* ── Info note ── */
    .info-note {
        background: rgba(173,216,230,0.2);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        font-size: 0.87rem;
        color: #3a6070;
        border: 1px solid rgba(173,216,230,0.4);
        line-height: 1.6;
    }

    /* ── Divider ── */
    hr.soft-divider {
        border: none;
        border-top: 1.5px solid rgba(255,192,203,0.35);
        margin: 1.5rem 0;
    }

    /* Streamlit selectbox / radio aesthetics */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stRadio"] > div {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase everything
    - Remove URLs (common in MBTI posts)
    - Remove special characters, keeping only letters and spaces
    - Strip extra whitespace
    """
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove everything that isn't a letter or space
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# TRAIN MODELS (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(csv_source: str):
    """
    Load the MBTI dataset, engineer features with TF-IDF,
    and train SVM + Random Forest models for two binary tasks:
      1. Introvert (I) vs Extrovert (E)
      2. Thinking (T) vs Feeling (F)

    csv_source can be:
      - A raw GitHub URL  (https://raw.githubusercontent.com/...)
      - A local file path (mbti_1.csv)

    Returns a dict with vectorizers and trained models.
    """
    import io, urllib.request

    if csv_source.startswith("http://") or csv_source.startswith("https://"):
        # Download from URL into memory — works on Streamlit Cloud
        with urllib.request.urlopen(csv_source) as resp:
            raw_bytes = resp.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        df = pd.read_csv(csv_source)

    # ── Clean posts ──
    df['clean_posts'] = df['posts'].apply(preprocess_text)

    # ── Binary labels ──
    # MBTI type is a 4-letter string, e.g. "INFJ"
    df['IE'] = df['type'].apply(lambda t: 0 if t[0] == 'I' else 1)   # 0=I, 1=E
    df['TF'] = df['type'].apply(lambda t: 0 if t[2] == 'T' else 1)   # 0=T, 1=F

    results = {}

    for task, label_col in [('IE', 'IE'), ('TF', 'TF')]:
        X = df['clean_posts']
        y = df[label_col]

        # ── TF-IDF vectorisation ──
        # max_features keeps it lightweight; ngram_range captures short phrases
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True        # log-scale TF to dampen very frequent terms
        )
        X_vec = vectorizer.fit_transform(X)

        # ── Train / test split ──
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── SVM ──
        svm = SVC(kernel='linear', probability=True, random_state=42, C=1.0)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        svm_acc   = accuracy_score(y_test, svm_preds)
        svm_cm    = confusion_matrix(y_test, svm_preds)

        # ── Random Forest ──
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc   = accuracy_score(y_test, rf_preds)
        rf_cm    = confusion_matrix(y_test, rf_preds)

        results[task] = {
            'vectorizer': vectorizer,
            'svm': svm,
            'rf': rf,
            'svm_acc': svm_acc,
            'rf_acc': rf_acc,
            'svm_cm': svm_cm,
            'rf_cm': rf_cm,
        }

    return results


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_personality(text: str, model_choice: str, models: dict):
    """
    Given raw text and the chosen model ('SVM' or 'Random Forest'),
    return predictions and confidence probabilities for both tasks.
    """
    clean = preprocess_text(text)
    out = {}

    for task in ['IE', 'TF']:
        vec   = models[task]['vectorizer']
        model = models[task]['svm'] if model_choice == 'SVM' else models[task]['rf']

        X = vec.transform([clean])
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        out[task] = {
            'label': pred,
            'proba': proba,
            'confidence': round(max(proba) * 100, 1)
        }

    return out


# ─────────────────────────────────────────────
# FRIENDLY EXPLANATIONS
# ─────────────────────────────────────────────
IE_EXPLANATIONS = {
    0: ("Introvert 🌙",
        "You seem to recharge through quiet time and inner reflection. "
        "You likely prefer deep one-on-one conversations over large social gatherings, "
        "and you think carefully before speaking."),
    1: ("Extrovert ☀️",
        "You seem to gain energy from being around people and social interaction. "
        "You're likely expressive, outgoing, and enjoy meeting new people and "
        "engaging in lively discussions.")
}

TF_EXPLANATIONS = {
    0: ("Thinking 🧠",
        "You tend to approach decisions with logic and objective analysis. "
        "You value fairness, honesty, and are comfortable making tough calls "
        "based on facts rather than feelings."),
    1: ("Feeling 💛",
        "You tend to lead with empathy and personal values when making decisions. "
        "You're attuned to others' emotions and care deeply about harmony "
        "and the human impact of choices.")
}

COMBO_SUMMARIES = {
    (0, 0): ("You may prefer deep focus, structured environments, and logical decision-making. "
             "You're likely thoughtful, analytical, and value quality over quantity in relationships."),
    (0, 1): ("You may be a quiet but deeply empathetic person who listens carefully and cares "
             "about the feelings of those close to you."),
    (1, 0): ("You may be assertive, expressive, and enjoy debating ideas. You likely bring "
             "energy and clear-headed thinking to group settings."),
    (1, 1): ("You may be warm, enthusiastic, and socially connected — someone who brings "
             "people together and leads with heart."),
}


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():

    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-emoji">🌸</span>
        <h1 class="hero-title">Personality Predictor</h1>
        <p class="hero-subtitle">
            Share a few thoughts and discover whether you lean 
            <strong>Introverted or Extroverted</strong>, 
            <strong>Thinking or Feeling</strong> — 
            powered by classic machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # DATASET SOURCE — update GITHUB_RAW_URL to your own repo URL
    # Example:
    #   https://raw.githubusercontent.com/YourName/YourRepo/main/mbti_1.csv
    #
    # How to get this URL:
    #   1. Go to your GitHub repo
    #   2. Click on mbti_1.csv
    #   3. Click the "Raw" button
    #   4. Copy that URL and paste it below
    # ──────────────────────────────────────────────────────────────────
    GITHUB_RAW_URL = (
        "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/mbti_1.csv"
    )
    LOCAL_PATH = "mbti_1.csv"

    # Auto-select source: use local file when running on your machine,
    # otherwise fetch from GitHub (works on Streamlit Cloud)
    if os.path.exists(LOCAL_PATH):
        csv_source = LOCAL_PATH
    else:
        csv_source = GITHUB_RAW_URL
        if "YOUR_USERNAME" in GITHUB_RAW_URL:
            st.error(
                "**Dataset not found.** \n\n"
                "You are running on Streamlit Cloud but the GitHub URL hasn't been set yet. "
                "Open `app.py` and replace `YOUR_USERNAME` and `YOUR_REPO` in the "
                "`GITHUB_RAW_URL` variable with your actual GitHub username and repo name."
            )
            st.stop()

    with st.spinner("🌸 Loading dataset and training models — this takes ~30–60 s the first time …"):
        models = load_and_train(csv_source)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Model accuracy overview ──
    st.markdown('<p class="section-header">📊 Model Accuracy</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">🟣 Introvert / Extrovert</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-value">{models['IE']['svm_acc']*100:.1f}%</div>
                <div class="metric-label">SVM</div>
            </div>
            <div class="metric-chip">
                <div class="metric-value">{models['IE']['rf_acc']*100:.1f}%</div>
                <div class="metric-label">Random Forest</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">🔵 Thinking / Feeling</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-value">{models['TF']['svm_acc']*100:.1f}%</div>
                <div class="metric-label">SVM</div>
            </div>
            <div class="metric-chip">
                <div class="metric-value">{models['TF']['rf_acc']*100:.1f}%</div>
                <div class="metric-label">Random Forest</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── User Input Section ──
    st.markdown('<p class="section-header">✍️ Tell Us About Yourself</p>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">💬 Your Text</p>', unsafe_allow_html=True)
    user_text = st.text_area(
        label="Enter your text",
        placeholder="Write freely — describe how you feel, think, or interact with the world. "
                    "The more you write, the better! (Aim for 3–5 sentences or more.)",
        height=160,
        label_visibility="collapsed"
    )

    # Model choice
    model_choice = st.radio(
        "Choose a model:",
        options=["SVM", "Random Forest"],
        horizontal=True,
        help="SVM is generally more accurate on text tasks. Random Forest is here for comparison."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔮  Predict My Personality", use_container_width=True)

    # ── Results Section ──
    if predict_btn:
        if not user_text.strip():
            st.warning("Please enter some text before predicting.")
        elif len(user_text.split()) < 5:
            st.warning("Please write at least a sentence or two for a meaningful prediction.")
        else:
            preds = predict_personality(user_text, model_choice, models)

            ie_label = preds['IE']['label']     # 0=I, 1=E
            tf_label = preds['TF']['label']     # 0=T, 1=F
            ie_conf  = preds['IE']['confidence']
            tf_conf  = preds['TF']['confidence']

            ie_name, ie_desc = IE_EXPLANATIONS[ie_label]
            tf_name, tf_desc = TF_EXPLANATIONS[tf_label]
            combo_summary    = COMBO_SUMMARIES[(ie_label, tf_label)]

            st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">🌸 Your Personality Profile</p>', unsafe_allow_html=True)

            # ── Combined summary ──
            st.markdown(f"""
            <div class="summary-box">
                <p>You seem more <strong>{ie_name}</strong> and <strong>{tf_name}</strong>.<br>
                {combo_summary}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Individual trait cards ──
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Energy Style</div>
                    <div class="result-trait">{ie_name}</div>
                    <div class="result-desc">{ie_desc}</div>
                    <span class="confidence-pill">Confidence: {ie_conf}%</span>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="result-card blue">
                    <div class="result-label blue-label">Decision Style</div>
                    <div class="result-trait blue-text">{tf_name}</div>
                    <div class="result-desc">{tf_desc}</div>
                    <span class="confidence-pill blue-pill">Confidence: {tf_conf}%</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability breakdown ──
            st.markdown('<div class="soft-card" style="margin-top:1.2rem;">', unsafe_allow_html=True)
            st.markdown('<p class="card-title">📈 Probability Breakdown</p>', unsafe_allow_html=True)

            ie_proba = preds['IE']['proba']
            tf_proba = preds['TF']['proba']

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Introvert ↔ Extrovert**")
                st.progress(float(ie_proba[0]), text=f"Introvert: {ie_proba[0]*100:.1f}%")
                st.progress(float(ie_proba[1]), text=f"Extrovert: {ie_proba[1]*100:.1f}%")
            with c2:
                st.markdown("**Thinking ↔ Feeling**")
                st.progress(float(tf_proba[0]), text=f"Thinking: {tf_proba[0]*100:.1f}%")
                st.progress(float(tf_proba[1]), text=f"Feeling: {tf_proba[1]*100:.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Disclaimer ──
            st.markdown("""
            <div class="info-note" style="margin-top:1rem;">
                🌿 <strong>Remember:</strong> This is a fun ML experiment trained on internet text — 
                it's not a clinical assessment. Personality is nuanced and complex. Take results 
                as a light-hearted reflection, not a definitive label!
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#b090a0; font-size:0.82rem; padding:0.5rem 0 1.5rem;">
        Built with 🌸 Streamlit · SVM &amp; Random Forest · TF-IDF Features<br>
        Dataset: MBTI Personality Type Dataset (Kaggle)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
