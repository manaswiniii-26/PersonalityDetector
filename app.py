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
# CSS - UNIFIED STYLE SECTION
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; box-sizing: border-box; }

.stApp { background: linear-gradient(160deg,#ffe4ec 0%,#fce4f3 40%,#ead6f5 100%); min-height:100vh; }

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

.block-container { max-width:800px !important; margin:0 auto !important; padding:2rem 1.5rem 4rem !important; }

/* HERO */
.hero-wrap { text-align:center; padding: 2rem 0; margin-bottom: 2rem; }
.hero-title { font-size:3rem; font-weight:800; color:#9d174d; margin:0; }
.hero-tagline { color: #be185d; font-size: 1.1rem; opacity: 0.8; }

/* SECTION WRAPPER */
.sec-wrap { margin-top: 2.5rem; margin-bottom: 1.5rem; text-align: center; }
.sec-title { font-size: 1.5rem; font-weight: 800; color: #7e22ce; }
.sec-sub { font-size: 0.9rem; color: #6b7280; }

/* CARDS & TILES */
.metric-card { background: white; padding: 1rem; border-radius: 15px; text-align: center; border: 1px solid #fce7f3; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.metric-label { font-size: 0.7rem; text-transform: uppercase; color: #9d174d; font-weight: 700; }
.metric-val { font-size: 1.2rem; font-weight: 800; color: #3b0764; }

/* RADIO BUTTONS - DARK TEXT FIX */
div[data-testid="stRadio"] label p {
    color: #3b0764 !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] label {
    color: #7e22ce !important;
}

/* TEXTAREA - DARK TEXT FIX */
textarea { 
    border-radius:14px !important; 
    border:2px solid #f9a8d4 !important;
    background:rgba(255,255,255,.95) !important; 
    color:#3b0764 !important;
    font-size:.97rem !important; 
}

/* NEXT QUOTE BUTTON */
div.stButton > button {
    background: linear-gradient(135deg,#f472b6 0%,#c084fc 100%) !important;
    color:#fff !important; font-weight:800 !important;
    border:none !important; border-radius:50px !important;
    padding:.75rem 2rem !important; width:100% !important;
    box-shadow:0 6px 24px rgba(244,114,182,.45) !important;
}

/* SPECTRUM TRACKS */
.spec-wrap { background: white; padding: 1.2rem; border-radius: 15px; margin-bottom: 1rem; border: 1px solid #f5f3ff; }
.spec-labels { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 700; color: #4b5563; margin-bottom: 0.5rem; }
.spec-track { background: #f3f4f6; height: 8px; border-radius: 10px; overflow: hidden; }
.spec-fill { background: linear-gradient(90deg, #f472b6, #c084fc); height: 100%; border-radius: 10px; transition: width 0.5s ease; }
.spec-note { font-size: 0.7rem; color: #9ca3af; margin-top: 0.4rem; }
.spec-badge { padding: 2px 6px; border-radius: 4px; font-size: 0.6rem; color: white; margin: 0 4px; }
.badge-live { background: #ec4899; }
.badge-ref { background: #a855f7; }

/* MBTI TILES */
.mbti-tile { background: white; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #f3f4f6; transition: 0.3s; margin-bottom: 10px;}
.mbti-tile.hl { border: 2px solid #f472b6; background: #fff1f2; transform: scale(1.05); box-shadow: 0 10px 20px rgba(244,114,182,0.2); }
.mbti-emoji { font-size: 1.5rem; }
.mbti-code { font-weight: 800; color: #3b0764; }
.mbti-name { font-size: 0.7rem; color: #6b7280; }

/* AFFIRMATION WRAP */
.affirmation-wrap { 
    background:rgba(255,255,255,.5); 
    border-radius:22px; padding:2rem; 
    text-align:center; border:1px solid rgba(244,114,182,.2);
}
.affirmation-quote { font-size:1.2rem; font-weight:700; color:#3b0764; font-style:italic; margin-bottom:1rem; }
.affirmation-sub { font-size:0.8rem; color:#9d174d; text-transform:uppercase; letter-spacing:1px; }

.divider { border:none; border-top:1.5px solid rgba(244,114,182,.25); margin:1.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# STATIC DATA
# ──────────────────────────────────────────────────────
MBTI_TYPES = {
    "INTJ": ("Architect",    "🏛️", "Strategic, independent, and quietly visionary."),
    "INTP": ("Logician",     "🔭", "Curious thinkers drawn to abstract theory."),
    "ENTJ": ("Commander",    "⚡", "Bold, decisive, and driven to lead."),
    "ENTP": ("Debater",      "💡", "Quick-witted, creative, loves a good challenge."),
    "INFJ": ("Advocate",     "🌸", "Empathetic visionaries with deep conviction."),
    "INFP": ("Mediator",     "🌙", "Idealistic, sensitive, and quietly creative."),
    "ENFJ": ("Protagonist",  "🌟", "Charismatic leaders who uplift everyone around them."),
    "ENFP": ("Campaigner",   "🎨", "Enthusiastic, imaginative, and free-spirited."),
    "ISTJ": ("Logistician",  "📋", "Reliable, thorough, and deeply responsible."),
    "ISFJ": ("Defender",     "🛡️", "Warm protectors who care for others quietly."),
    "ESTJ": ("Executive",    "📊", "Organised, logical, and clear about right and wrong."),
    "ESFJ": ("Consul",       "🤝", "Caring, sociable, and always there for others."),
    "ISTP": ("Virtuoso",     "🔧", "Observant craftspeople who love how things work."),
    "ISFP": ("Adventurer",   "🎭", "Gentle, spontaneous, and deeply artistic."),
    "ESTP": ("Entrepreneur", "🎯", "Energetic risk-takers who act in the moment."),
    "ESFP": ("Entertainer",  "🎉", "Spontaneous, fun-loving, and full of warmth."),
}

IE_DESCS = {
    "Extroverted": "Your writing suggests you draw energy from the world around you — connection, action, and shared experience.",
    "Introverted": "Your language carries the texture of inner life — reflective, layered, and oriented inward.",
}
TF_DESCS = {
    "Feeling":  "Emotional nuance flows through your writing. You lead with the heart and weigh decisions through a lens of values.",
    "Thinking": "Your writing shows systematic thinking — analytical, principled, and inclined toward logic over sentiment.",
}

AFFIRMATIONS = [
    "You are not your type — you are the whole ocean, and your type is just today's tide.",
    "Every letter is a tendency, not a prison. You contain multitudes.",
    "Your quiet moments are not empty — they are full of you.",
    "Sensitivity is not weakness. It is the ability to feel the world more fully.",
    "The most interesting people are always a little hard to classify.",
]

# ──────────────────────────────────────────────────────
# PREPROCESSING + MODEL TRAINING
# ──────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@st.cache_resource(show_spinner=False)
def train_engine():
    FILE_ID = "1WpYslfFqGPdMHChZOrHjpcCsfehdvshN"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    df = pd.read_csv(url).sample(n=2500, random_state=42)
    results = {}
    for task, idx, pos_label in [("IE", 0, "E"), ("TF", 2, "F")]:
        df["target"] = df["type"].apply(lambda x: 1 if x[idx] == pos_label else 0)
        vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
        X = vec.fit_transform(df["posts"].apply(preprocess))
        y = df["target"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        svm = SVC(kernel="linear", probability=True).fit(X_tr, y_tr)
        rf  = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        results[task] = {
            "vec": vec, "svm": svm, "rf": rf,
            "svm_acc": accuracy_score(y_te, svm.predict(X_te)),
            "rf_acc": accuracy_score(y_te, rf.predict(X_te)),
        }
    return results

def predict(engine, text: str, model_key: str) -> dict:
    clean = preprocess(text)
    probs = {}
    for task in ("IE", "TF"):
        vec = engine[task]["vec"]
        clf = engine[task][model_key]
        probs[task] = clf.predict_proba(vec.transform([clean]))[0][1]
    return probs

# ──────────────────────────────────────────────────────
# BOOT & UI
# ──────────────────────────────────────────────────────
with st.spinner("Calibrating Aura AI..."):
    engine = train_engine()

if "aff_idx" not in st.session_state:
    st.session_state["aff_idx"] = 0

# HERO
st.markdown('<div class="hero-wrap"><div class="hero-title">Aura <em>AI</em></div><div class="hero-tagline">Discover the patterns woven into your words.</div></div>', unsafe_allow_html=True)

# METRICS
st.markdown('<div class="sec-wrap"><div class="sec-title">Model Intelligence</div></div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
stats = [(c1, "SVM IE", engine['IE']['svm_acc']), (c2, "RF IE", engine['IE']['rf_acc']), (c3, "SVM TF", engine['TF']['svm_acc']), (c4, "RF TF", engine['TF']['rf_acc'])]
for col, label, acc in stats:
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-val">{acc*100:.1f}%</div></div>', unsafe_allow_html=True)

# PREDICTOR
st.markdown('<div class="sec-wrap"><div class="sec-title">Explore Your Aura</div></div>', unsafe_allow_html=True)
user_text = st.text_area("", height=200, placeholder="Type something about yourself...")
model_pick = st.radio("Intelligence architecture", ["SVM", "Random Forest"], horizontal=True)
run_btn = st.button("✦ Unveil Aura")

if run_btn and user_text.strip():
    mkey = "svm" if "SVM" in model_pick else "rf"
    probs = predict(engine, user_text, mkey)
    ie_type = "Extroverted" if probs["IE"] > 0.5 else "Introverted"
    tf_type = "Feeling" if probs["TF"] > 0.5 else "Thinking"
    st.session_state.update({"ran": True, "ie_type": ie_type, "tf_type": tf_type, "ie_bar_pct": round(probs["IE"]*100), "tf_bar_pct": round(probs["TF"]*100), "guessed": ("E" if ie_type == "Extroverted" else "I") + "NTJ"})

# SPECTRUM SECTION
if st.session_state.get("ran"):
    ie_bar = st.session_state["ie_bar_pct"]
    tf_bar = st.session_state["tf_bar_pct"]
    st.markdown(f"""
    <div class="spec-wrap">
        <div class="spec-labels"><span>Introversion</span><span>Extraversion</span></div>
        <div class="spec-track"><div class="spec-fill" style="width:{ie_bar}%"></div></div>
    </div>
    <div class="spec-wrap">
        <div class="spec-labels"><span>Thinking</span><span>Feeling</span></div>
        <div class="spec-track"><div class="spec-fill" style="width:{tf_bar}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ARCHETYPES GRID
st.markdown('<div class="sec-wrap"><div class="sec-title">The 16 Archetypes</div></div>', unsafe_allow_html=True)
guessed = st.session_state.get("guessed")
cols = st.columns(4)
for i, (code, (name, emoji, _)) in enumerate(MBTI_TYPES.items()):
    hl = " hl" if code == guessed else ""
    cols[i % 4].markdown(f'<div class="mbti-tile{hl}"><div class="mbti-emoji">{emoji}</div><div class="mbti-code">{code}</div><div class="mbti-name">{name}</div></div>', unsafe_allow_html=True)

# AFFIRMATION & BUTTON
st.markdown('<div class="sec-wrap"><div class="sec-title">A Note for You</div></div>', unsafe_allow_html=True)
quote = AFFIRMATIONS[st.session_state["aff_idx"]]
st.markdown(f'<div class="affirmation-wrap"><div class="affirmation-quote">"{quote}"</div><div class="affirmation-sub">A beginning, not a conclusion</div></div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 1.5, 2])
with btn_col:
    if st.button("Next quote →"):
        st.session_state["aff_idx"] = (st.session_state["aff_idx"] + 1) % len(AFFIRMATIONS)
        st.rerun()

st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
