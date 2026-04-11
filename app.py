import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Aura · Personality Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
h1,h2,h3,.playfair { font-family: 'Playfair Display', serif !important; }

.stApp {
    background: #fffbf9;
    background-image: radial-gradient(#f5e8ed 1px, transparent 1px);
    background-size: 36px 36px;
}

/* ── HERO ── */
.hero-wrap {
    padding: 72px 24px 48px;
    text-align: center;
    background: linear-gradient(160deg, #fff5f5 0%, #fdf0f8 50%, #f5f0ff 100%);
    border-bottom: 1px solid #f0e0e8;
    margin-bottom: 48px;
}
.hero-eyebrow {
    font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase;
    color: #c9899a; font-weight: 500; margin-bottom: 14px;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.6rem; font-weight: 600; color: #3a2830; line-height: 1.08;
}
.hero-title em { font-style: italic; color: #b07fa0; }
.hero-tagline {
    margin-top: 14px; font-size: 14px; color: #9a7080;
    font-weight: 300; max-width: 420px; margin-left: auto;
    margin-right: auto; line-height: 1.7;
}

/* ── SECTION HEADER ── */
.sec-header { margin: 48px 0 6px; }
.sec-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.35rem; color: #3a2830; font-weight: 600;
}
.sec-sub {
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: #c9899a; font-weight: 500; margin-bottom: 20px;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: white; border: 0.5px solid #f0e0e8;
    border-radius: 12px; padding: 20px 16px;
    border-top: 3px solid #d4a0b0;
}
.metric-card.lav { border-top-color: #b0a0d4; }
.metric-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem; color: #3a2830; line-height: 1; margin-bottom: 6px;
}
.metric-label {
    font-size: 10px; text-transform: uppercase;
    color: #c9899a; letter-spacing: 1.5px; font-weight: 500;
}

/* ── RESULTS ── */
.result-block { margin-bottom: 28px; }
.result-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: #c9899a; font-weight: 500; margin-bottom: 4px;
}
.result-value {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem; color: #3a2830; line-height: 1.1;
}
.result-desc { font-size: 13px; color: #9a7080; margin-top: 6px; font-style: italic; line-height: 1.65; }
.conf-bar-wrap { height: 5px; background: #f5eaee; border-radius: 3px; margin-top: 12px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #d4a0b0, #b07fa0); }
.conf-label { font-size: 11px; color: #c9899a; margin-top: 4px; text-align: right; }

/* ── SPECTRUM BARS ── */
.spec-wrap { margin-bottom: 22px; }
.spec-labels { display: flex; justify-content: space-between; font-size: 11px; color: #9a7080; margin-bottom: 6px; font-weight: 500; }
.spec-track { height: 8px; background: #f5eaee; border-radius: 4px; overflow: hidden; }
.spec-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #f0c0d0, #b07fa0); }
.spec-note { font-size: 11px; color: #c9899a; margin-top: 4px; text-align: right; }

/* ── INSIGHT CARDS ── */
.insight-card {
    background: white; border: 0.5px solid #f0e0e8;
    border-radius: 12px; padding: 20px;
}
.insight-icon { font-size: 22px; margin-bottom: 10px; }
.insight-title { font-size: 13px; font-weight: 600; color: #3a2830; margin-bottom: 4px; }
.insight-text { font-size: 12px; color: #9a7080; line-height: 1.65; }

/* ── MBTI GRID ── */
.mbti-tile {
    background: white; border: 0.5px solid #f0e0e8;
    border-radius: 10px; padding: 14px 8px; text-align: center;
    transition: border-color 0.2s;
}
.mbti-tile:hover { border-color: #d4a0b0; background: #fdf0f8; }
.mbti-tile.hl { border-color: #b07fa0 !important; background: #fdf0f8 !important; box-shadow: 0 0 0 2px #d4a0b050; }
.mbti-emoji { font-size: 20px; margin-bottom: 6px; }
.mbti-code { font-weight: 600; font-size: 0.95rem; color: #3a2830; }
.mbti-name { font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: #b09aa0; margin-top: 2px; }

/* ── AFFIRMATION ── */
.affirmation-wrap {
    background: linear-gradient(135deg, #fdf0f8 0%, #f5f0ff 100%);
    border: 0.5px solid #e8d0e8; border-radius: 16px; padding: 40px 32px;
    text-align: center;
}
.affirmation-quote {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.35rem; font-style: italic; color: #3a2830;
    line-height: 1.65; max-width: 560px; margin: 0 auto 10px;
}
.affirmation-sub { font-size: 10px; letter-spacing: 2px; color: #c9a8b0; text-transform: uppercase; margin-bottom: 6px; }

/* ── LETTER CHIPS ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.chip {
    padding: 4px 12px; border-radius: 20px; font-size: 11px;
    background: #fdf0f8; color: #8a4060; border: 0.5px solid #e8c0d0; font-weight: 500;
}

/* ── PLACEHOLDER ── */
.placeholder { text-align: center; padding: 48px 20px; color: #c9a8b0; font-style: italic; font-size: 14px; line-height: 1.7; }

/* ── BUTTON ── */
div.stButton > button {
    background: #3a2830 !important; color: white !important;
    border-radius: 8px !important; padding: 13px 28px !important;
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 2.5px !important; text-transform: uppercase !important;
    border: none !important; width: 100% !important;
}
div.stButton > button:hover { background: #5a3840 !important; }

textarea { border-radius: 8px !important; border: 0.5px solid #e8d8e0 !important; background: #fffaf9 !important; }
.stRadio > div { gap: 12px; }
.stRadio > div > label { border: 0.5px solid #e8d8e0; border-radius: 8px; padding: 10px 16px; background: white; cursor: pointer; font-size: 13px; color: #9a7080; }
.stRadio > div > label[data-checked="true"] { border-color: #b07fa0; background: #fdf0f8; color: #3a2830; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── DATA + MODELS ────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource(show_spinner=False)
def train_engine():
    FILE_ID = "1WpYslfFqGPdMHChZOrHjpcCsfehdvshN"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    df = pd.read_csv(url).sample(n=2500, random_state=42)
    results = {}
    for task_name, task_idx in [('IE', 0), ('TF', 2)]:
        df['target'] = df['type'].apply(lambda x: 1 if x[task_idx] in ['E', 'F'] else 0)
        vec = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
        X = vec.fit_transform(df['posts'].apply(preprocess))
        y = df['target']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        svm = SVC(kernel='linear', probability=True).fit(X_tr, y_tr)
        rf  = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
        results[task_name] = {
            'vec': vec, 'svm': svm, 'rf': rf,
            'svm_acc': accuracy_score(y_te, svm.predict(X_te)),
            'rf_acc':  accuracy_score(y_te, rf.predict(X_te))
        }
    return results

with st.spinner("Calibrating the analytical engine…"):
    engine = train_engine()


# ── MBTI DATA ────────────────────────────────────────────────────────────────
MBTI_TYPES = {
    "INTJ": ("Architect",   "🏛️", "Strategic, independent, and quietly visionary."),
    "INTP": ("Logician",    "🔭", "Curious thinkers drawn to abstract theory."),
    "ENTJ": ("Commander",   "⚡", "Bold, decisive, and driven to lead."),
    "ENTP": ("Debater",     "💡", "Quick-witted, creative, and loves a good challenge."),
    "INFJ": ("Advocate",    "🌸", "Empathetic visionaries with deep conviction."),
    "INFP": ("Mediator",    "🌙", "Idealistic, sensitive, and quietly creative."),
    "ENFJ": ("Protagonist", "🌟", "Charismatic leaders who uplift everyone around them."),
    "ENFP": ("Campaigner",  "🎨", "Enthusiastic, imaginative, and free-spirited."),
    "ISTJ": ("Logistician", "📋", "Reliable, thorough, and deeply responsible."),
    "ISFJ": ("Defender",    "🛡️", "Warm protectors who care for others quietly."),
    "ESTJ": ("Executive",   "📊", "Organized, logical, and clear about right and wrong."),
    "ESFJ": ("Consul",      "🤝", "Caring, sociable, and always there for others."),
    "ISTP": ("Virtuoso",    "🔧", "Observant craftspeople who love how things work."),
    "ISFP": ("Adventurer",  "🎭", "Gentle, spontaneous, and deeply artistic."),
    "ESTP": ("Entrepreneur","🎯", "Energetic risk-takers who act in the moment."),
    "ESFP": ("Entertainer", "🎉", "Spontaneous, fun-loving, and full of warmth.")
}

AFFIRMATIONS = [
    "You are not your type — you are the whole ocean, and your type is just today's tide.",
    "Every letter is a tendency, not a prison. You contain multitudes.",
    "Your quiet moments are not empty — they are full of you.",
    "Sensitivity is not weakness. It is the ability to feel the world more fully.",
    "The most interesting people are always a little hard to classify."
]

IE_DESCS = {
    'Extroverted': "Your words suggest you draw energy from the world around you — connection, action, and shared experience seem to fuel you.",
    'Introverted': "Your language carries the texture of inner life — reflective, layered, and oriented inward. Solitude is likely where you find yourself."
}
TF_DESCS = {
    'Feeling': "Emotional nuance flows through your writing. You lead with the heart and tend to weigh decisions through a lens of values and empathy.",
    'Thinking': "Your writing shows systematic thinking — analytical, principled, and inclined toward logic over sentiment."
}


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Personality Analytics</div>
    <div class="hero-title">Aura <em>AI</em></div>
    <div class="hero-tagline">Discover the patterns woven into your words — a quiet mirror for your inner world.</div>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1 · MODEL METRICS ────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">Model Intelligence</div><div class="sec-sub">Calibration metrics</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">SVM Accuracy · I/E</div><div class="metric-val">{engine["IE"]["svm_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">RF Accuracy · I/E</div><div class="metric-val">{engine["IE"]["rf_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card lav"><div class="metric-label">SVM Accuracy · T/F</div><div class="metric-val">{engine["TF"]["svm_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card lav"><div class="metric-label">RF Accuracy · T/F</div><div class="metric-val">{engine["TF"]["rf_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)


# ── SECTION 2 · PREDICTOR ────────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">Explore Your Aura</div><div class="sec-sub">Write freely — the model listens</div></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    user_text   = st.text_area("", height=220, placeholder="Describe a memory that shaped you, how you recharge, or your philosophy on life…")
    model_pick  = st.radio("Intelligence architecture", ["Support Vector Machine (SVM)", "Random Forest (RF)"], horizontal=True)
    run_btn     = st.button("✦ Unveil Aura")

with right:
    if run_btn and user_text.strip():
        mkey = 'svm' if "SVM" in model_pick else 'rf'
        probs = {}
        for task in ['IE', 'TF']:
            vec = engine[task]['vec']
            clf = engine[task][mkey]
            probs[task] = clf.predict_proba(vec.transform([preprocess(user_text)]))[0]

        ie_score = probs['IE'][1]
        tf_score = probs['TF'][1]
        ie_type  = "Extroverted" if ie_score > 0.5 else "Introverted"
        tf_type  = "Feeling"     if tf_score > 0.5 else "Thinking"
        ie_conf  = round(max(ie_score, 1 - ie_score) * 100, 1)
        tf_conf  = round(max(tf_score, 1 - tf_score) * 100, 1)

        first = 'E' if ie_score > 0.5 else 'I'
        third = 'F' if tf_score > 0.5 else 'T'
        guessed = first + 'N' + third + 'J'
        g_name  = MBTI_TYPES.get(guessed, ('—','',''))[0]
        g_desc  = MBTI_TYPES.get(guessed, ('','',''))[2]

        st.markdown(f"""
        <div class="result-block">
            <div class="result-label">Social Orientation</div>
            <div class="result-value">{ie_type}</div>
            <div class="result-desc">{IE_DESCS[ie_type]}</div>
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{ie_conf}%"></div></div>
            <div class="conf-label">{ie_conf}% confidence</div>
        </div>
        <div class="result-block">
            <div class="result-label">Cognitive Style</div>
            <div class="result-value">{tf_type}</div>
            <div class="result-desc">{TF_DESCS[tf_type]}</div>
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{tf_conf}%"></div></div>
            <div class="conf-label">{tf_conf}% confidence</div>
        </div>
        <div class="result-block">
            <div class="result-label">Your Archetype Leans Toward</div>
            <div class="result-value" style="font-size:1.5rem;">{guessed} · {g_name}</div>
            <div class="result-desc">{g_desc}</div>
            <div class="chip-row">
                <span class="chip">{'Extroverted' if first=='E' else 'Introverted'}</span>
                <span class="chip">Intuitive</span>
                <span class="chip">{'Feeling' if third=='F' else 'Thinking'}</span>
                <span class="chip">Judging</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state['ie_score'] = ie_score
        st.session_state['tf_score'] = tf_score
        st.session_state['guessed']  = guessed
    elif run_btn:
        st.info("Please share a few sentences above.")
    else:
        st.markdown('<div class="placeholder">Share a few sentences above and your cognitive portrait will emerge here…</div>', unsafe_allow_html=True)


# ── SECTION 3 · PERSONALITY DIMENSIONS ──────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">Personality Dimensions</div><div class="sec-sub">The four axes of type theory</div></div>', unsafe_allow_html=True)

ie_pct = round(st.session_state.get('ie_score', 0.5) * 100)
tf_pct = round(st.session_state.get('tf_score', 0.5) * 100)
ie_lbl = f"{ie_pct}% toward {'Extroverted' if ie_pct > 50 else 'Introverted'}" if 'ie_score' in st.session_state else "Run analysis to update"
tf_lbl = f"{tf_pct}% toward {'Feeling' if tf_pct > 50 else 'Thinking'}" if 'tf_score' in st.session_state else "Run analysis to update"

st.markdown(f"""
<div class="spec-wrap">
    <div class="spec-labels"><span>Introversion</span><span>Extraversion</span></div>
    <div class="spec-track"><div class="spec-fill" style="width:{ie_pct}%"></div></div>
    <div class="spec-note">{ie_lbl}</div>
</div>
<div class="spec-wrap">
    <div class="spec-labels"><span>Intuition</span><span>Sensing</span></div>
    <div class="spec-track"><div class="spec-fill" style="width:55%"></div></div>
    <div class="spec-note">Reference baseline</div>
</div>
<div class="spec-wrap">
    <div class="spec-labels"><span>Thinking</span><span>Feeling</span></div>
    <div class="spec-track"><div class="spec-fill" style="width:{tf_pct}%"></div></div>
    <div class="spec-note">{tf_lbl}</div>
</div>
<div class="spec-wrap">
    <div class="spec-labels"><span>Perceiving</span><span>Judging</span></div>
    <div class="spec-track"><div class="spec-fill" style="width:48%"></div></div>
    <div class="spec-note">Reference baseline</div>
</div>
""", unsafe_allow_html=True)


# ── SECTION 4 · DIMENSION INSIGHTS ──────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">Personality at a Glance</div><div class="sec-sub">What each dimension reveals</div></div>', unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)
for col, icon, title, text in [
    (i1, "🌙", "Energy source",       "Introverts recharge in solitude; extroverts draw energy from connection and social engagement."),
    (i2, "🌿", "Information style",   "Sensors trust concrete, lived experience. Intuitives seek patterns and meaning beneath the surface."),
    (i3, "⚖️", "Decision making",     "Thinkers weigh logic and fairness; Feelers prioritize harmony and the emotional weight a choice carries."),
    (i4, "🌊", "Lifestyle preference","Judgers prefer structure and closure; Perceivers stay open, adaptable, and comfortable with ambiguity."),
]:
    with col:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-icon">{icon}</div>
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)


# ── SECTION 5 · MBTI DIRECTORY ───────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">The 16 Archetypes</div><div class="sec-sub">Explore all personalities</div></div>', unsafe_allow_html=True)

guessed = st.session_state.get('guessed', None)
cols = st.columns(4)
for i, (code, (name, emoji, _)) in enumerate(MBTI_TYPES.items()):
    hl = ' hl' if code == guessed else ''
    with cols[i % 4]:
        st.markdown(f"""
        <div class="mbti-tile{hl}">
            <div class="mbti-emoji">{emoji}</div>
            <div class="mbti-code">{code}</div>
            <div class="mbti-name">{name}</div>
        </div>
        """, unsafe_allow_html=True)


# ── SECTION 6 · AFFIRMATIONS ─────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-title">A Note for You</div><div class="sec-sub">Gentle reminders</div></div>', unsafe_allow_html=True)

if 'aff_idx' not in st.session_state:
    st.session_state['aff_idx'] = 0

quote = AFFIRMATIONS[st.session_state['aff_idx']]
st.markdown(f"""
<div class="affirmation-wrap">
    <div class="affirmation-quote">"{quote}"</div>
    <div class="affirmation-sub">Each type is a beginning, not a conclusion</div>
</div>
""", unsafe_allow_html=True)

ba, bb, bc = st.columns([3,1,3])
with bb:
    if st.button("Next quote →"):
        st.session_state['aff_idx'] = (st.session_state['aff_idx'] + 1) % len(AFFIRMATIONS)
        st.rerun()

st.markdown('<div style="height: 80px;"></div>', unsafe_allow_html=True)
