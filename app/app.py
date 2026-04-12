import streamlit as st
import sys
import importlib.util
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Root setup
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────
# Streamlit config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLoad",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# Global CSS (Safe + Stable)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base app */
html, body, [data-testid="stApp"] {
    background:
        radial-gradient(circle at top right, rgba(108,99,255,0.08), transparent 25%),
        radial-gradient(circle at bottom left, rgba(61,214,140,0.05), transparent 25%),
        #0a0a0f;
    color: #e8e8f0;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu,
footer,
header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}

section[data-testid="stSidebar"] {
    display: none !important;
}

/* Main layout container */
.block-container {
    max-width: 1280px !important;
    margin: auto !important;
    padding: 1.5rem 2rem 3rem 2rem !important;
}

/* Navbar */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 28px;
    margin: 12px auto 28px auto;
    max-width: 1280px;
    background: rgba(17,17,24,0.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.nav-logo {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -.02em;
}

.nav-logo span {
    color: #6c63ff;
}

.nav-status {
    font-size: 12px;
    color: #777790;
    font-family: 'JetBrains Mono', monospace;
}

/* Typography */
p, li, span {
    color: #9999b3;
    font-size: 14px;
    line-height: 1.6;
}

h1 {
    color: #ffffff;
    font-weight: 700;
    font-size: 24px;
}

h2 {
    color: #ffffff;
    font-weight: 600;
    font-size: 18px;
}

/* Panels */
.panel {
    background: rgba(17,17,24,0.82);
    border: 1px solid #1e1e30;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    margin-bottom: 24px;
}

/* Metric cards */
.metric {
    flex: 1;
    background: linear-gradient(145deg,#111118,#151520);
    border: 1px solid #1f1f30;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    transition: all .2s ease;
}

.metric:hover {
    transform: translateY(-2px);
}

.metric-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 11px;
    color: #666680;
    margin-top: 6px;
}

/* State badges */
.state-low {
    color: #3dd68c;
    background: rgba(61,214,140,.08);
    border: 1px solid rgba(61,214,140,.2);
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

.state-high {
    color: #ff5757;
    background: rgba(255,87,87,.08);
    border: 1px solid rgba(255,87,87,.2);
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

.state-idle {
    color: #777790;
    background: rgba(85,85,112,.08);
    border: 1px solid rgba(85,85,112,.2);
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    display: inline-block;
}

/* Alerts */
.alert-critical,
.alert-warning,
.alert-info {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
}

.alert-critical {
    background: rgba(255,87,87,.05);
    border: 1px solid rgba(255,87,87,.25);
    border-left: 3px solid #ff5757;
}

.alert-warning {
    background: rgba(255,179,71,.05);
    border: 1px solid rgba(255,179,71,.20);
    border-left: 3px solid #ffb347;
}

.alert-info {
    background: rgba(108,99,255,.05);
    border: 1px solid rgba(108,99,255,.20);
    border-left: 3px solid #6c63ff;
}

.alert-title {
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 4px;
}

.alert-msg {
    font-size: 12px;
    color: #bbbbd0;
}

.alert-suggest {
    font-size: 11px;
    color: #6c63ff;
    margin-top: 6px;
    font-style: italic;
}

/* Buttons */
.stButton > button {
    background: #6c63ff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: #7c73ff !important;
}

.stButton > button:disabled {
    background: #222233 !important;
    color: #555570 !important;
}

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #111118 !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border: none !important;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    background: #111118 !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 12px !important;
    color: #777790 !important;
    padding: 12px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background: #1a1a2e !important;
    color: #ffffff !important;
    border: 1px solid #6c63ff !important;
}

/* Labels */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 12px;
}

.divider {
    border: none;
    border-top: 1px solid #1a1a2e;
    margin: 24px 0;
}

.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #6c63ff;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0a0f;
}
::-webkit-scrollbar-thumb {
    background: #1e1e30;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Navbar
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">Neuro<span>Load</span></div>
    <div class="nav-status">
        EEG Cognitive Load Monitor • COG-BCI Dataset • XGBoost + SHAP
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Safe dynamic page loader
# ─────────────────────────────────────────────────────────────
def load_page(path: Path):
    if not path.exists():
        st.error(f"Missing page file: {path.name}")
        st.stop()

    spec = importlib.util.spec_from_file_location("_page", str(path))
    if spec is None or spec.loader is None:
        st.error(f"Could not load page: {path.name}")
        st.stop()

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────────────────────
# Load Pages
# ─────────────────────────────────────────────────────────────
PD = ROOT / "app" / "pages"

dashboard = load_page(PD / "dashboard.py")
session_review = load_page(PD / "session_review.py")
research_mode = load_page(PD / "research_mode.py")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Monitor",
    "Session Review",
    "Research"
])

# ─────────────────────────────────────────────────────────────
# Render pages safely
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    try:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        dashboard.render()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Dashboard error: {e}")

with tabs[1]:
    try:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        session_review.render()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Session Review error: {e}")

with tabs[2]:
    try:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        research_mode.render()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Research Mode error: {e}")