"""
Personality Prediction from Text using Machine Learning
========================================================
Models : SVM + Random Forest
Features: TF-IDF (word + bigrams)
Tasks   : Introvert/Extrovert  &  Thinking/Feeling
"""

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
    page_title="Personality Predictor ✨",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; box-sizing: border-box; }

.stApp { background: linear-gradient(160deg,#ffe4ec 0%,#fce4f3 40%,#ead6f5 100%); min-height:100vh; }

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

.block-container { max-width:800px !important; margin:0 auto !important; padding:2rem 1.5rem 4rem !important; }

/* HERO */
.hero {
    background: linear-gradient(135deg,#f472b6 0%,#e879f9 60%,#c084fc 100%);
    border-radius:28px; padding:3rem 2rem; text-align:center;
    margin-bottom:2.2rem; box-shadow:0 10px 40px rgba(244,114,182,.35);
    position:relative; overflow:hidden;
}
.hero::before { content:''; position:absolute; width:220px; height:220px;
    background:rgba(255,255,255,.12); border-radius:50%; top:-60px; right:-60px; }
.hero::after  { content:''; position:absolute; width:160px; height:160px;
    background:rgba(255,255,255,.08); border-radius:50%; bottom:-50px; left:-40px; }
.hero-icon  { font-size:3rem; display:block; margin-bottom:.6rem; position:relative; z-index:2; }
.hero-title { font-size:2.6rem; font-weight:800; color:#fff; margin:0 0 .6rem;
    line-height:1.2; position:relative; z-index:2; text-shadow:0 2px 12px rgba(0,0,0,.15); }
.hero-sub   { font-size:1.05rem; color:rgba(255,255,255,.92); font-weight:400;
    max-width:520px; margin:0 auto; line-height:1.7; position:relative; z-index:2; }

/* SECTION LABEL */
.sec-label { font-size:.72rem; font-weight:800; letter-spacing:.13em;
    text-transform:uppercase; color:#c026d3; margin:2rem 0 .7rem; text-align:center; }

/* CARD */
.card { background:rgba(255,255,255,.78); backdrop-filter:blur(14px);
    border-radius:22px; padding:1.6rem 1.8rem;
    box-shadow:0 4px 28px rgba(192,38,211,.1);
    border:1px solid rgba(244,114,182,.2); margin-bottom:1.2rem; }
.card-head { font-weight:700; font-size:1rem; color:#9d174d; margin-bottom:.8rem; }

/* CHIPS */
.chips { display:flex; gap:.9rem; flex-wrap:wrap; }
.chip  { flex:1; min-width:110px; background:linear-gradient(135deg,#fdf2f8,#fae8ff);
    border-radius:16px; padding:.9rem .8rem; text-align:center;
    border:1px solid rgba(244,114,182,.25); box-shadow:0 2px 10px rgba(192,38,211,.08); }
.chip-val { font-size:1.55rem; font-weight:800; color:#be185d; line-height:1; }
.chip-lbl { font-size:.72rem; font-weight:600; color:#9d174d; margin-top:.3rem; letter-spacing:.05em; }

/* TILES */
.tiles { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.tile  { flex:1; min-width:140px; background:linear-gradient(135deg,#fff,#fdf4ff);
    border-radius:18px; padding:1.2rem 1rem; text-align:center;
    border:1px solid rgba(216,180,254,.4); box-shadow:0 3px 14px rgba(192,38,211,.08); }
.tile-icon  { font-size:1.8rem; margin-bottom:.4rem; display:block; }
.tile-title { font-weight:700; font-size:.9rem; color:#7e22ce; }
.tile-desc  { font-size:.78rem; color:#6b7280; line-height:1.5; margin-top:.3rem; }

/* TEXTAREA */
textarea { border-radius:14px !important; border:2px solid #f9a8d4 !important;
    background:rgba(255,255,255,.95) !important; color:#3b0764 !important;
    font-size:.97rem !important; font-family:'Nunito',sans-serif !important; transition:border .2s !important; }
textarea:focus { border-color:#e879f9 !important; box-shadow:0 0 0 3px rgba(232,121,249,.18) !important; }

div[data-testid="stRadio"] label { font-weight:600 !important; color:#7e22ce !important; }

/* BUTTON */
div.stButton > button {
    background: linear-gradient(135deg,#f472b6 0%,#c084fc 100%) !important;
    color:#fff !important; font-family:'Nunito',sans-serif !important;
    font-weight:800 !important; font-size:1.1rem !important;
    border:none !important; border-radius:50px !important;
    padding:.75rem 2rem !important; width:100% !important;
    cursor:pointer !important; box-shadow:0 6px 24px rgba(244,114,182,.45) !important;
    transition:all .25s ease !important; letter-spacing:.04em !important;
}
div.stButton > button:hover { transform:translateY(-3px) !important; box-shadow:0 10px 32px rgba(192,38,211,.4) !important; }

/* RESULT CARDS */
.res-grid { display:flex; gap:1rem; margin:1rem 0; }
.res-card { flex:1; background:linear-gradient(135deg,#fdf2f8 0%,#fae8ff 100%);
    border-radius:20px; padding:1.4rem 1.3rem; border-left:5px solid #f472b6;
    box-shadow:0 4px 18px rgba(244,114,182,.18); }
.res-card.purple { border-left-color:#a855f7; }
.res-tag   { font-size:.7rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#be185d; margin-bottom:.35rem; }
.res-tag.pt { color:#7e22ce; }
.res-name  { font-size:1.5rem; font-weight:800; color:#831843; margin-bottom:.4rem; }
.res-name.pt { color:#581c87; }
.res-desc  { font-size:.87rem; color:#4b5563; line-height:1.6; }
.pill { display:inline-block; margin-top:.7rem; padding:.2rem .9rem; border-radius:50px;
    font-size:.78rem; font-weight:700; background:rgba(244,114,182,.18);
    color:#9d174d; border:1px solid rgba(244,114,182,.35); }
.pill.pt { background:rgba(168,85,247,.15); color:#6b21a8; border-color:rgba(168,85,247,.3); }

/* SUMMARY BANNER */
.summary { background:linear-gradient(135deg,#f472b6,#a855f7); border-radius:20px;
    padding:1.6rem 2rem; text-align:center; margin:1rem 0;
    box-shadow:0 6px 28px rgba(244,114,182,.3); }
.summary p { color:#fff; font-size:1.08rem; line-height:1.7; margin:0; font-weight:600; }

/* PROGRESS */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#f472b6,#c084fc) !important; border-radius:99px !important; }

/* NOTE */
.note { background:rgba(255,255,255,.65); border-radius:14px; padding:.9rem 1.2rem;
    font-size:.85rem; color:#6b7280; border:1px solid rgba(244,114,182,.25);
    line-height:1.6; margin-top:1rem; text-align:center; }

.divider { border:none; border-top:1.5px solid rgba(244,114,182,.25); margin:1.6rem 0; }
.footer  { text-align:center; font-size:.78rem; color:#c084fc; padding:1rem 0 .5rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────────────
# LOAD DATA  (only 3 000 rows for speed)
# ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    FILE_ID = "1WpYslfFqGPdMHChZOrHjpcCsfehdvshN"
    url     = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    session = requests.Session()
    resp    = session.get(url, stream=True, timeout=60)
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            resp = session.get(url + "&confirm=" + v, stream=True, timeout=60)
            break
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
    df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    df['clean'] = df['posts'].apply(preprocess)
    df['IE']    = df['type'].apply(lambda t: 0 if t[0] == 'I' else 1)
    df['TF']    = df['type'].apply(lambda t: 0 if t[2] == 'T' else 1)
    return df


# ──────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_df):
    results = {}
    for task in ['IE', 'TF']:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)
        X   = vec.fit_transform(_df['clean'])
        y   = _df[task]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        svm = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
        svm.fit(X_tr, y_tr)
        rf  = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        results[task] = dict(
            vec=vec, svm=svm, rf=rf,
            svm_acc=accuracy_score(y_te, svm.predict(X_te)),
            rf_acc =accuracy_score(y_te, rf.predict(X_te)),
        )
    return results


# ──────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────
def predict(text, model_key, models):
    clean = preprocess(text)
    out   = {}
    for task in ['IE', 'TF']:
        m     = models[task]['svm'] if model_key == 'SVM' else models[task]['rf']
        X     = models[task]['vec'].transform([clean])
        pred  = m.predict(X)[0]
        proba = m.predict_proba(X)[0]
        out[task] = dict(label=int(pred), proba=proba, conf=round(max(proba)*100, 1))
    return out


# ──────────────────────────────────────────────────────
# COPY
# ──────────────────────────────────────────────────────
IE_INFO = {
    0: ("🌙 Introvert", "You recharge through quiet time and deep reflection. You prefer meaningful one-on-one conversations and think carefully before speaking."),
    1: ("☀️ Extrovert", "You gain energy from people and social situations. You're expressive, outgoing, and love engaging with the world around you."),
}
TF_INFO = {
    0: ("🧠 Thinker", "You make decisions with logic and objective analysis. You value honesty and fairness, and you're comfortable with tough, fact-based choices."),
    1: ("💛 Feeler", "You lead with empathy and personal values. You're deeply attuned to others' emotions and care about harmony and human connection."),
}
COMBOS = {
    (0,0): "You tend to be quietly analytical — someone who thinks deeply before acting and values logic above all.",
    (0,1): "You're a gentle, empathetic soul who listens carefully and cares deeply about the people close to you.",
    (1,0): "You're energetic and direct — someone who loves debating ideas and bringing clear thinking to group settings.",
    (1,1): "You're warm and enthusiastic — someone who brings people together and leads every room with heart.",
}
MBTI_TYPES = {
    "INTJ":"The Architect","INTP":"The Thinker","ENTJ":"The Commander","ENTP":"The Debater",
    "INFJ":"The Advocate","INFP":"The Mediator","ENFJ":"The Protagonist","ENFP":"The Campaigner",
    "ISTJ":"The Logistician","ISFJ":"The Defender","ESTJ":"The Executive","ESFJ":"The Consul",
    "ISTP":"The Virtuoso","ISFP":"The Adventurer","ESTP":"The Entrepreneur","ESFP":"The Entertainer",
}


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────
def main():

    # HERO
    st.markdown("""
    <div class="hero">
        <span class="hero-icon">🌸</span>
        <h1 class="hero-title">Personality Predictor</h1>
        <p class="hero-sub">
            Write a few sentences and discover whether you lean
            <strong>Introverted or Extroverted</strong> &amp;
            <strong>Thinking or Feeling</strong>
        </p>
    </div>""", unsafe_allow_html=True)

    # WHAT IS MBTI
    st.markdown('<p class="sec-label">✨ What is MBTI?</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tiles">
        <div class="tile"><span class="tile-icon">🌙</span>
            <div class="tile-title">Introvert (I)</div>
            <div class="tile-desc">Recharges alone, thinks before speaking, values depth</div></div>
        <div class="tile"><span class="tile-icon">☀️</span>
            <div class="tile-title">Extrovert (E)</div>
            <div class="tile-desc">Recharges socially, loves conversation, energetic</div></div>
        <div class="tile"><span class="tile-icon">🧠</span>
            <div class="tile-title">Thinking (T)</div>
            <div class="tile-desc">Decides with logic, values fairness &amp; objectivity</div></div>
        <div class="tile"><span class="tile-icon">💛</span>
            <div class="tile-title">Feeling (F)</div>
            <div class="tile-desc">Decides with empathy, values harmony &amp; people</div></div>
    </div>""", unsafe_allow_html=True)

    # LOAD + TRAIN
    with st.spinner("🌸 Loading data and training models — about 15–20 seconds ..."):
        try:
            df     = load_data()
            models = train_models(df)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            st.stop()

    # ACCURACY
    st.markdown('<p class="sec-label">📊 Model Accuracy</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟣 Introvert / Extrovert**")
        st.markdown(f"""<div class="chips">
            <div class="chip"><div class="chip-val">{models['IE']['svm_acc']*100:.1f}%</div><div class="chip-lbl">SVM</div></div>
            <div class="chip"><div class="chip-val">{models['IE']['rf_acc']*100:.1f}%</div><div class="chip-lbl">Random Forest</div></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("**🔵 Thinking / Feeling**")
        st.markdown(f"""<div class="chips">
            <div class="chip"><div class="chip-val">{models['TF']['svm_acc']*100:.1f}%</div><div class="chip-lbl">SVM</div></div>
            <div class="chip"><div class="chip-val">{models['TF']['rf_acc']*100:.1f}%</div><div class="chip-lbl">Random Forest</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # EXPLORE 16 TYPES
    st.markdown('<p class="sec-label">🔍 Explore All 16 MBTI Types</p>', unsafe_allow_html=True)
    with st.expander("Click to explore"):
        cols = st.columns(4)
        for i, (code, name) in enumerate(MBTI_TYPES.items()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#fdf2f8,#fae8ff);
                    border-radius:12px;padding:.7rem .8rem;margin:.3rem 0;
                    border:1px solid rgba(244,114,182,.2);text-align:center;">
                    <div style="font-weight:800;color:#be185d;font-size:1rem;">{code}</div>
                    <div style="font-size:.75rem;color:#6b7280;">{name}</div>
                </div>""", unsafe_allow_html=True)

    # INPUT
    st.markdown('<p class="sec-label">✍️ Try It Yourself</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-head">💬 Share your thoughts</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        label="text", label_visibility="collapsed",
        placeholder="e.g. I love spending quiet evenings reading and reflecting on ideas. "
                    "I find large crowds draining but deep conversations energising ...",
        height=150,
    )
    model_choice = st.radio(
        "Choose your model:",
        ["SVM (recommended)", "Random Forest"],
        horizontal=True,
    )
    model_key = "SVM" if "SVM" in model_choice else "Random Forest"
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔮  Predict My Personality Type")

    # RESULTS
    if predict_btn:
        if not user_text.strip() or len(user_text.split()) < 5:
            st.warning("Please write at least 2–3 sentences for a good prediction!")
        else:
            with st.spinner("Analysing your text …"):
                preds = predict(user_text, model_key, models)

            ie_lbl = preds['IE']['label']
            tf_lbl = preds['TF']['label']
            ie_name, ie_desc = IE_INFO[ie_lbl]
            tf_name, tf_desc = TF_INFO[tf_lbl]
            combo = COMBOS[(ie_lbl, tf_lbl)]

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="sec-label">🌸 Your Personality Profile</p>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="summary">
                <p>You seem to be <strong>{ie_name}</strong> &amp; <strong>{tf_name}</strong><br>
                <span style="font-weight:400;font-size:.97rem;">{combo}</span></p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="res-grid">
                <div class="res-card">
                    <div class="res-tag">Energy Style</div>
                    <div class="res-name">{ie_name}</div>
                    <div class="res-desc">{ie_desc}</div>
                    <span class="pill">Confidence: {preds['IE']['conf']}%</span>
                </div>
                <div class="res-card purple">
                    <div class="res-tag pt">Decision Style</div>
                    <div class="res-name pt">{tf_name}</div>
                    <div class="res-desc">{tf_desc}</div>
                    <span class="pill pt">Confidence: {preds['TF']['conf']}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
            st.markdown('<div class="card-head">📈 Probability Breakdown</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            ip, ep = preds['IE']['proba']
            tp, fp = preds['TF']['proba']
            with c1:
                st.markdown("**🌙 Introvert vs ☀️ Extrovert**")
                st.progress(float(ip), text=f"Introvert {ip*100:.1f}%")
                st.progress(float(ep), text=f"Extrovert {ep*100:.1f}%")
            with c2:
                st.markdown("**🧠 Thinking vs 💛 Feeling**")
                st.progress(float(tp), text=f"Thinking {tp*100:.1f}%")
                st.progress(float(fp), text=f"Feeling  {fp*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="note">
                🌿 This is a fun ML experiment — not a clinical assessment.
                Personality is complex and nuanced. Enjoy the result as a reflection, not a label!
            </div>""", unsafe_allow_html=True)

    # FOOTER
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        Built with 🌸 Streamlit &nbsp;·&nbsp; SVM &amp; Random Forest
        &nbsp;·&nbsp; TF-IDF Features &nbsp;·&nbsp; MBTI Dataset (Kaggle)
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
