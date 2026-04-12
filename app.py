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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aura · Personality Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, .playfair         { font-family: 'Playfair Display', serif !important; }

.stApp {
    background: #fffbf9;
    background-image: radial-gradient(#f5e8ed 1px, transparent 1px);
    background-size: 36px 36px;
}

/* HERO */
.hero-wrap {
    padding: 72px 24px 48px;
    text-align: center;
    background: linear-gradient(160deg, #fff5f5 0%, #fdf0f8 50%, #f5f0ff 100%);
    border-bottom: 1px solid #f0e0e8;
    margin-bottom: 48px;
}
.hero-eyebrow { font-size: 10px; letter-spacing: 3.5px; text-transform: uppercase; color: #c9899a; font-weight: 500; margin-bottom: 14px; }
.hero-title   { font-family: 'Playfair Display', serif !important; font-size: 3.6rem; font-weight: 600; color: #3a2830; line-height: 1.08; }
.hero-title em { font-style: italic; color: #b07fa0; }
.hero-tagline  { margin-top: 14px; font-size: 14px; color: #9a7080; font-weight: 300; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.7; }

/* SECTION HEADERS */
.sec-wrap  { margin: 48px 0 6px; }
.sec-title { font-family: 'Playfair Display', serif !important; font-size: 1.35rem; color: #3a2830; font-weight: 600; }
.sec-sub   { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: #c9899a; font-weight: 500; margin-bottom: 20px; }

/* METRIC CARDS */
.metric-card     { background: white; border: 0.5px solid #f0e0e8; border-radius: 12px; padding: 20px 16px; border-top: 3px solid #d4a0b0; }
.metric-card.lav { border-top-color: #b0a0d4; }
.metric-val      { font-family: 'Playfair Display', serif !important; font-size: 2rem; color: #3a2830; line-height: 1; margin-bottom: 6px; }
.metric-label    { font-size: 10px; text-transform: uppercase; color: #c9899a; letter-spacing: 1.5px; font-weight: 500; }

/* RESULT BLOCKS */
.result-block  { margin-bottom: 28px; }
.result-label  { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: #c9899a; font-weight: 500; margin-bottom: 4px; }
.result-value  { font-family: 'Playfair Display', serif !important; font-size: 2.4rem; color: #3a2830; line-height: 1.1; }
.result-desc   { font-size: 13px; color: #9a7080; margin-top: 6px; font-style: italic; line-height: 1.65; }
.conf-bar-wrap { height: 5px; background: #f5eaee; border-radius: 3px; margin-top: 12px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #f0c0d0, #b07fa0); }
.conf-lbl      { font-size: 11px; color: #c9899a; margin-top: 4px; text-align: right; }

/* SPECTRUM BARS */
.spec-wrap    { margin-bottom: 22px; }
.spec-labels  { display: flex; justify-content: space-between; font-size: 11px; color: #9a7080; margin-bottom: 6px; font-weight: 500; }
.spec-track   { height: 8px; background: #f5eaee; border-radius: 4px; overflow: hidden; }
.spec-fill    { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #f0c0d0, #b07fa0); }
.spec-note    { font-size: 11px; color: #c9899a; margin-top: 5px; text-align: right; }
.spec-badge   { display: inline-block; font-size: 9px; padding: 2px 8px; border-radius: 10px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-left: 4px; }
.badge-live   { background: #fde8f0; color: #a04060; }
.badge-ref    { background: #f0eef8; color: #8070a0; }

/* INSIGHT CARDS */
.insight-card  { background: white; border: 0.5px solid #f0e0e8; border-radius: 12px; padding: 20px; height: 100%; }
.insight-icon  { font-size: 22px; margin-bottom: 10px; }
.insight-title { font-size: 13px; font-weight: 600; color: #3a2830; margin-bottom: 4px; }
.insight-text  { font-size: 12px; color: #9a7080; line-height: 1.65; }

/* MBTI GRID */
.mbti-tile     { background: white; border: 0.5px solid #f0e0e8; border-radius: 10px; padding: 14px 8px; text-align: center; transition: border-color 0.2s; margin-bottom: 10px; }
.mbti-tile:hover { border-color: #d4a0b0; background: #fdf0f8; }
.mbti-tile.hl  { border-color: #b07fa0 !important; background: #fdf0f8 !important; box-shadow: 0 0 0 2px #d4a0b050; }
.mbti-emoji    { font-size: 20px; margin-bottom: 6px; }
.mbti-code     { font-weight: 600; font-size: 0.95rem; color: #3a2830; }
.mbti-name     { font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: #b09aa0; margin-top: 2px; }

/* CHIP TAGS */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.chip     { padding: 4px 12px; border-radius: 20px; font-size: 11px; background: #fdf0f8; color: #8a4060; border: 0.5px solid #e8c0d0; font-weight: 500; }

/* AFFIRMATION */
.affirmation-wrap  { background: linear-gradient(135deg, #fdf0f8 0%, #f5f0ff 100%); border: 0.5px solid #e8d0e8; border-radius: 16px; padding: 40px 32px; text-align: center; }
.affirmation-quote { font-family: 'Playfair Display', serif !important; font-size: 1.35rem; font-style: italic; color: #3a2830; line-height: 1.65; max-width: 560px; margin: 0 auto 10px; }
.affirmation-sub   { font-size: 10px; letter-spacing: 2px; color: #c9a8b0; text-transform: uppercase; }

/* PLACEHOLDER */
.placeholder { text-align: center; padding: 48px 20px; color: #c9a8b0; font-style: italic; font-size: 14px; line-height: 1.7; border: 0.5px dashed #e8d0d8; border-radius: 12px; background: #fffaf9; }

/* STREAMLIT BUTTON */
div.stButton > button {
    background: #3a2830 !important; color: white !important;
    border-radius: 8px !important; padding: 13px 28px !important;
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 2.5px !important; text-transform: uppercase !important;
    border: none !important; width: 100% !important;
}
div.stButton > button:hover { background: #5a3840 !important; }

textarea { border-radius: 8px !important; border: 0.5px solid #e8d8e0 !important; background: #fffaf9 !important; color: #3a2830 !important; }

/* Radio button labels */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p {
    color: #3a2830 !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING + MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


@st.cache_resource(show_spinner=False)
def train_engine():
    """
    Downloads the MBTI dataset and trains two binary classifiers:
      IE : Introvert (0) vs Extrovert (1)  — type letter index 0
      TF : Thinking  (0) vs Feeling  (1)   — type letter index 2

    Returns a dict with vectoriser, both models, and test accuracies per task.
    """
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
            "vec":     vec,
            "svm":     svm,
            "rf":      rf,
            "svm_acc": accuracy_score(y_te, svm.predict(X_te)),
            "rf_acc":  accuracy_score(y_te, rf.predict(X_te)),
        }
    return results


def predict(engine, text: str, model_key: str) -> dict:
    """
    Returns raw probabilities from the chosen classifier:
      ie_prob  →  P(Extrovert)   — 0 = strongly Introverted, 1 = strongly Extroverted
      tf_prob  →  P(Feeling)     — 0 = strongly Thinking,    1 = strongly Feeling
    """
    clean = preprocess(text)
    probs = {}
    for task in ("IE", "TF"):
        vec = engine[task]["vec"]
        clf = engine[task][model_key]
        probs[task] = clf.predict_proba(vec.transform([clean]))[0][1]
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Calibrating the analytical engine…"):
    engine = train_engine()

if "aff_idx" not in st.session_state:
    st.session_state["aff_idx"] = 0


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Personality Analytics</div>
    <div class="hero-title">Aura <em>AI</em></div>
    <div class="hero-tagline">
        Discover the patterns woven into your words —
        a quiet mirror for your inner world.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODEL METRICS  (live from trained models)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">Model Intelligence</div>
    <div class="sec-sub">Live calibration metrics from your dataset</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
tiles = [
    (c1, "SVM Accuracy · I/E", f"{engine['IE']['svm_acc']*100:.1f}%", ""),
    (c2, "RF Accuracy  · I/E", f"{engine['IE']['rf_acc']*100:.1f}%",  ""),
    (c3, "SVM Accuracy · T/F", f"{engine['TF']['svm_acc']*100:.1f}%", " lav"),
    (c4, "RF Accuracy  · T/F", f"{engine['TF']['rf_acc']*100:.1f}%",  " lav"),
]
for col, label, val, cls in tiles:
    with col:
        st.markdown(
            f'<div class="metric-card{cls}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-val">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">Explore Your Aura</div>
    <div class="sec-sub">Write freely — the model listens</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    user_text  = st.text_area(
        "",
        height=220,
        placeholder="Describe a memory that shaped you, how you recharge, or your philosophy on life…",
    )
    model_pick = st.radio(
        "Intelligence architecture",
        ["Support Vector Machine (SVM)", "Random Forest (RF)"],
        horizontal=True,
    )
    run_btn = st.button("✦ Unveil Aura")

# Run inference → persist results in session_state so they survive reruns
if run_btn:
    if user_text.strip():
        mkey  = "svm" if "SVM" in model_pick else "rf"
        probs = predict(engine, user_text, mkey)

        ie_prob = probs["IE"]   # P(Extrovert): 0→Introvert, 1→Extrovert
        tf_prob = probs["TF"]   # P(Feeling):   0→Thinking,  1→Feeling

        ie_type = "Extroverted" if ie_prob > 0.5 else "Introverted"
        tf_type = "Feeling"     if tf_prob > 0.5 else "Thinking"

        # Confidence = probability of the winning class
        ie_conf = round(max(ie_prob, 1 - ie_prob) * 100, 1)
        tf_conf = round(max(tf_prob, 1 - tf_prob) * 100, 1)

        # Spectrum bar %:
        #   IE bar → ie_prob * 100  (left=Introvert, right=Extrovert)
        #   TF bar → tf_prob * 100  (left=Thinking,  right=Feeling)
        ie_bar_pct = round(ie_prob * 100)
        tf_bar_pct = round(tf_prob * 100)

        # Closest MBTI guess (N/S and P/J fixed to dataset majority)
        first   = "E" if ie_type == "Extroverted" else "I"
        third   = "F" if tf_type == "Feeling"     else "T"
        guessed = first + "N" + third + "J"
        g_name  = MBTI_TYPES.get(guessed, ("—", "", ""))[0]
        g_desc  = MBTI_TYPES.get(guessed, ("", "", ""))[2]

        st.session_state.update({
            "ran":        True,
            "ie_type":    ie_type,
            "tf_type":    tf_type,
            "ie_conf":    ie_conf,
            "tf_conf":    tf_conf,
            "ie_bar_pct": ie_bar_pct,
            "tf_bar_pct": tf_bar_pct,
            "guessed":    guessed,
            "g_name":     g_name,
            "g_desc":     g_desc,
        })
    else:
        st.warning("Please share a few sentences above to begin.")

# Result panel (reads from session_state so it persists across reruns)
with right:
    if st.session_state.get("ran"):
        s = st.session_state
        st.markdown(f"""
        <div class="result-block">
            <div class="result-label">Social Orientation</div>
            <div class="result-value">{s['ie_type']}</div>
            <div class="result-desc">{IE_DESCS[s['ie_type']]}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{s['ie_conf']}%"></div>
            </div>
            <div class="conf-lbl">{s['ie_conf']}% confidence</div>
        </div>

        <div class="result-block">
            <div class="result-label">Cognitive Style</div>
            <div class="result-value">{s['tf_type']}</div>
            <div class="result-desc">{TF_DESCS[s['tf_type']]}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{s['tf_conf']}%"></div>
            </div>
            <div class="conf-lbl">{s['tf_conf']}% confidence</div>
        </div>

        <div class="result-block">
            <div class="result-label">Closest Archetype</div>
            <div class="result-value" style="font-size:1.5rem;">
                {s['guessed']} · {s['g_name']}
            </div>
            <div class="result-desc">{s['g_desc']}</div>
            <div class="chip-row">
                <span class="chip">{s['ie_type']}</span>
                <span class="chip">Intuitive</span>
                <span class="chip">{s['tf_type']}</span>
                <span class="chip">Judging</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="placeholder">'
            "Share a few sentences above and your cognitive portrait will emerge here…"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PERSONALITY DIMENSION SPECTRA
#
# I/E bar  → directly from model: ie_prob * 100
#             left edge = 100% Introvert, right edge = 100% Extrovert
#
# T/F bar  → directly from model: tf_prob * 100
#             left edge = 100% Thinking,  right edge = 100% Feeling
#
# N/S bar  → reference baseline (dataset is ~72% Intuitive; no model trained)
# P/J bar  → reference baseline (dataset is ~54% Perceiving; no model trained)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">Personality Dimensions</div>
    <div class="sec-sub">Spectrum positions derived from your text</div>
</div>
""", unsafe_allow_html=True)

ie_bar = st.session_state.get("ie_bar_pct", 50)
tf_bar = st.session_state.get("tf_bar_pct", 50)
ran    = st.session_state.get("ran", False)

ie_note = (
    f"{ie_bar}% toward {'Extraversion' if ie_bar > 50 else 'Introversion'}"
    if ran else "Run analysis to update"
)
tf_note = (
    f"{tf_bar}% toward {'Feeling' if tf_bar > 50 else 'Thinking'}"
    if ran else "Run analysis to update"
)

st.markdown(f"""
<div class="spec-wrap">
    <div class="spec-labels">
        <span>Introversion</span>
        <span>Model-predicted <span class="spec-badge badge-live">live</span></span>
        <span>Extraversion</span>
    </div>
    <div class="spec-track"><div class="spec-fill" style="width:{ie_bar}%"></div></div>
    <div class="spec-note">{ie_note}</div>
</div>

<div class="spec-wrap">
    <div class="spec-labels">
        <span>Intuition</span>
        <span>Dataset baseline <span class="spec-badge badge-ref">ref</span></span>
        <span>Sensing</span>
    </div>
    <div class="spec-track"><div class="spec-fill" style="width:28%"></div></div>
    <div class="spec-note">Dataset skews ~72% Intuitive — reference only</div>
</div>

<div class="spec-wrap">
    <div class="spec-labels">
        <span>Thinking</span>
        <span>Model-predicted <span class="spec-badge badge-live">live</span></span>
        <span>Feeling</span>
    </div>
    <div class="spec-track"><div class="spec-fill" style="width:{tf_bar}%"></div></div>
    <div class="spec-note">{tf_note}</div>
</div>

<div class="spec-wrap">
    <div class="spec-labels">
        <span>Perceiving</span>
        <span>Dataset baseline <span class="spec-badge badge-ref">ref</span></span>
        <span>Judging</span>
    </div>
    <div class="spec-track"><div class="spec-fill" style="width:46%"></div></div>
    <div class="spec-note">Dataset ~54% Perceiving — reference only</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DIMENSION INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">Personality at a Glance</div>
    <div class="sec-sub">What each dimension reveals</div>
</div>
""", unsafe_allow_html=True)

insight_data = [
    ("🌙", "Energy source",
     "Introverts recharge through solitude and reflection; Extroverts draw energy from connection and social engagement."),
    ("🌿", "Information style",
     "Sensors trust concrete, lived experience. Intuitives are pattern-seekers drawn to meaning beneath the surface."),
    ("⚖️", "Decision making",
     "Thinkers weigh logic and fairness. Feelers prioritise harmony and the emotional weight a choice carries."),
    ("🌊", "Lifestyle preference",
     "Judgers prefer structure and closure. Perceivers stay open, adaptable, and comfortable with ambiguity."),
]
cols = st.columns(4)
for col, (icon, title, text) in zip(cols, insight_data):
    with col:
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-icon">{icon}</div>'
            f'<div class="insight-title">{title}</div>'
            f'<div class="insight-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — THE 16 ARCHETYPES
# Predicted type is highlighted after a run.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">The 16 Archetypes</div>
    <div class="sec-sub">Your result lights up after analysis</div>
</div>
""", unsafe_allow_html=True)

guessed = st.session_state.get("guessed", None)
cols    = st.columns(4)
for i, (code, (name, emoji, _)) in enumerate(MBTI_TYPES.items()):
    hl = " hl" if code == guessed else ""
    with cols[i % 4]:
        st.markdown(
            f'<div class="mbti-tile{hl}">'
            f'<div class="mbti-emoji">{emoji}</div>'
            f'<div class="mbti-code">{code}</div>'
            f'<div class="mbti-name">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — AFFIRMATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">A Note for You</div>
    <div class="sec-sub">Gentle reminders</div>
</div>
""", unsafe_allow_html=True)

quote = AFFIRMATIONS[st.session_state["aff_idx"]]
st.markdown(
    f'<div class="affirmation-wrap">'
    f'<div class="affirmation-quote">"{quote}"</div>'
    f'<div class="affirmation-sub">Each type is a beginning, not a conclusion</div>'
    f'</div>',
    unsafe_allow_html=True,
)

_, btn_col, _ = st.columns([3, 1, 3])
with btn_col:
    if st.button("Next quote →"):
        st.session_state["aff_idx"] = (st.session_state["aff_idx"] + 1) % len(AFFIRMATIONS)
        st.rerun()

st.markdown('<div style="height:80px"></div>', unsafe_allow_html=True)
