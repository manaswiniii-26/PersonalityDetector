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

/* RADIO BUTTONS - FIX FOR WHITE TEXT */
div[data-testid="stRadio"] label p {
    color: #3b0764 !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] label {
    color: #7e22ce !important;
}

/* TEXTAREA */
textarea { border-radius:14px !important; border:2px solid #f9a8d4 !important;
    background:rgba(255,255,255,.95) !important; color:#3b0764 !important;
    font-size:.97rem !important; font-family:'Nunito',sans-serif !important; transition:border .2s !important; }
textarea:focus { border-color:#e879f9 !important; box-shadow:0 0 0 3px rgba(232,121,249,.18) !important; }

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

/* AFFIRMATION WRAP */
.affirmation-wrap { 
    background:rgba(255,255,255,.5); 
    border-radius:22px; padding:2rem; 
    text-align:center; border:1px solid rgba(244,114,182,.2);
}
.affirmation-quote { font-size:1.2rem; font-weight:700; color:#3b0764; font-style:italic; margin-bottom:1rem; }
.affirmation-sub { font-size:0.8rem; color:#9d174d; text-transform:uppercase; letter-spacing:1px; }

/* SUMMARY BANNER */
.summary { background:linear-gradient(135deg,#f472b6,#a855f7); border-radius:20px;
    padding:1.6rem 2rem; text-align:center; margin:1rem 0;
    box-shadow:0 6px 28px rgba(244,114,182,.3); }
.summary p { color:#fff; font-size:1.08rem; line-height:1.7; margin:0; font-weight:600; }

/* PROGRESS */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#f472b6,#c084fc) !important; border-radius:99px !important; }

.divider { border:none; border-top:1.5px solid rgba(244,114,182,.25); margin:1.6rem 0; }
.footer  { text-align:center; font-size:.78rem; color:#c084fc; padding:1rem 0 .5rem; }
</style>

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


# --- SECTION 6: AFFIRMATIONS ---
st.markdown("""
<div class="sec-wrap">
    <div class="sec-title">A Note for You</div>
    <div class="sec-sub">Gentle reminders</div>
</div>
""", unsafe_allow_html=True)

# Display the Quote Box
quote = AFFIRMATIONS[st.session_state["aff_idx"]]
st.markdown(
    f'<div class="affirmation-wrap">'
    f'<div class="affirmation-quote">"{quote}"</div>'
    f'<div class="affirmation-sub">Each type is a beginning, not a conclusion</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Increased spacing for a cleaner "South Mumbai gallery" aesthetic
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

# Center the button with balanced columns
_, btn_col, _ = st.columns([2, 1.5, 2]) 
with btn_col:
    if st.button("Next quote →"):
        st.session_state["aff_idx"] = (st.session_state["aff_idx"] + 1) % len(AFFIRMATIONS)
        st.rerun()

# Extra padding at the very bottom so the button doesn't hit the edge of the screen
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
