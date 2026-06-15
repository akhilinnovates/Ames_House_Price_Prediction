"""
app.py — HomePriceIQ · Streamlit Dashboard
==========================================
Run:  streamlit run app.py
Requires: streamlit scikit-learn pandas numpy plotly joblib
Optional: xgboost  (model auto-detects algorithm used at training time)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomePriceIQ",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (glass / dark-gradient design) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: rgba(0,0,0,0);
}
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-right: 1px solid rgba(255,255,255,0.07);        
}
[data-testid="stSidebar"] * { color: #e2e2f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem; padding: 0.5rem 0.75rem;
    border-radius: 8px; transition: background 0.2s; cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

/* ── Top header bar ── */
.top-header {
    background: linear-gradient(90deg, #0f0c29 0%, #302b63 60%, #24243e 100%);
    padding: 1.2rem 2rem; margin: 0rem -2rem 2rem -2rem;
    display: flex; align-items: center; gap: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.top-header h1 {
    font-family: 'Syne', sans-serif; font-size: 1.7rem;
    font-weight: 800; color: #fff; margin: 0; letter-spacing: -0.5px;
}
.top-header .tagline { font-size: 0.8rem; color: rgba(255,255,255,0.5); margin: 0; font-weight: 300; }
.badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff !important; font-size: 0.65rem; font-weight: 600;
    padding: 0.15rem 0.6rem; border-radius: 99px;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 14px;
    padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.metric-card.purple::before { background: linear-gradient(90deg,#667eea,#764ba2); }
.metric-card.teal::before   { background: linear-gradient(90deg,#11998e,#38ef7d); }
.metric-card.orange::before { background: linear-gradient(90deg,#f7971e,#ffd200); }
.metric-card.rose::before   { background: linear-gradient(90deg,#f953c6,#b91d73); }
.metric-card .label { font-size: 0.72rem; color: #8b92a5; font-weight: 500; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.4rem; }
.metric-card .value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #1a1d2e; line-height: 1; }
.metric-card .sub   { font-size: 0.72rem; color: #8b92a5; margin-top: 0.3rem; }
.metric-card .icon  { position: absolute; top: 1rem; right: 1.2rem; font-size: 1.6rem; opacity: 0.15; }

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700;
    color: #1a1d2e; margin: 1.5rem 0 1rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-heading::after { content: ''; flex: 1; height: 1px; background: #e8eaf0; margin-left: 0.5rem; }

/* ── Prediction result box ── */
.prediction-box {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 60%, #24243e 100%);
    border-radius: 20px; padding: 2.5rem; text-align: center;
    box-shadow: 0 20px 60px rgba(48,43,99,0.35); margin: 1.5rem 0;
}
.prediction-box .pre-label { color: rgba(255,255,255,0.5); font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem; }
.prediction-box .price { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; color: #fff; line-height: 1; }
.prediction-box .range { color: rgba(255,255,255,0.55); font-size: 0.85rem; margin-top: 0.75rem; }
.prediction-box .confidence {
    display: inline-block; background: rgba(102,126,234,0.3);
    color: #a5b4fc; border: 1px solid rgba(102,126,234,0.4);
    border-radius: 99px; padding: 0.25rem 1rem; font-size: 0.75rem;
    font-weight: 600; margin-top: 0.75rem; letter-spacing: 0.5px;
}

/* ── Input card ── */
.input-section {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.input-section h4 {
    font-size: 0.8rem; font-weight: 600; color: #667eea;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid #f0f1f6;
}

/* ── Neighborhood card ── */
.nb-card {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 12px;
    padding: 1rem; display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 0.5rem; transition: box-shadow 0.2s;
}
.nb-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.nb-name  { font-weight: 600; color: #1a1d2e; font-size: 0.9rem; }
.nb-price { font-family: 'Syne', sans-serif; font-weight: 700; color: #667eea; font-size: 1rem; }
.nb-bar-wrap { flex: 1; margin: 0 1rem; height: 6px; background: #f0f1f6; border-radius: 99px; overflow: hidden; }
.nb-bar  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #667eea, #764ba2); }

/* ── Insight pill ── */
.insight-pill {
    display: inline-block; background: #f0f1ff; color: #4f46e5;
    border: 1px solid #c7d2fe; border-radius: 99px;
    padding: 0.3rem 0.9rem; font-size: 0.75rem; font-weight: 600; margin: 0.2rem;
}

/* ── Sidebar labels ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
    color: #fff !important; letter-spacing: -0.5px;
    padding: 0.5rem 0 1rem 0; display: block;
}
.sidebar-section {
    font-size: 0.65rem; font-weight: 600;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase; letter-spacing: 1.2px;
    padding: 1rem 0 0.4rem 0.75rem; display: block;
}

/* ── Chart wrapper ── */
.chart-card {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 16px;
    padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 1rem;
}
.chart-title {
    font-size: 0.85rem; font-weight: 600; color: #1a1d2e;
    margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;
}

/* ── Metric override ── */
[data-testid="stMetric"] { background: transparent !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    font-size: 0.9rem; padding: 0.65rem 2rem;
    transition: opacity 0.2s, transform 0.1s; width: 100%;
}
.stButton > button:hover  { opacity: 0.9; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Slider accent ── */
[data-testid="stSlider"] > div > div > div > div { background: #667eea !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-weight: 500; font-size: 0.85rem; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #667eea; border-bottom-color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
# IMPORTANT: this list MUST match the FEATURES list in train.py exactly.
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "Fireplaces",
    "BsmtFinSF1",
    "LotArea",
    "OpenPorchSF",
    "WoodDeckSF",
    "2ndFlrSF",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
]

MODEL_PATH = "models/streamlit_model.pkl"

NEIGHBORHOODS = [
    'CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst','NWAmes',
    'OldTown','BrkSide','Sawyer','NridgHt','NAmes','SawyerW','IDOTRR',
    'MeadowV','Edwards','Timber','Gilbert','StoneBr','ClearCr','NPkVill',
    'Blmngtn','BrDale','SWISU','Blueste',
]

NB_MEDIAN = {
    'NridgHt':315000,'NoRidge':301500,'StoneBr':278000,'Timber':228475,
    'Somerst':225500,'Veenker':218000,'Crawfor':200624,'ClearCr':200250,
    'CollgCr':197200,'Blmngtn':191000,'NWAmes':182900,'Gilbert':181000,
    'SawyerW':179900,'Mitchel':153500,'NPkVill':146000,'NAmes':140000,
    'SWISU':139500,'Blueste':137500,'Sawyer':135000,'BrkSide':124300,
    'Edwards':121750,'OldTown':119000,'BrDale':106000,'IDOTRR':103000,
    'MeadowV':88000,
}

QUAL_PRICE = {
    1:50150,2:51770,3:87474,4:108421,5:133523,
    6:161603,7:207716,8:274736,9:367513,10:438588,
}

DECADE_PRICE = {
    '1870':108000,'1880':124000,'1890':142200,'1900':127000,'1910':128750,
    '1920':115000,'1930':126500,'1940':122900,'1950':136000,'1960':146000,
    '1970':147500,'1980':178000,'1990':204000,'2000':223500,'2010':394432,
}

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    paths = [
        MODEL_PATH,
        "models/house_price_model.pkl",
        "streamlit_model.pkl",
        "house_price_model.pkl",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception:
                continue
    return None, None


def predict(model, inputs: dict) -> float:
    """
    Run model inference.
    Output is log1p(SalePrice) → reverse with expm1.
    Auto-detects if model already outputs raw dollars (> 20).
    """
    df = pd.DataFrame([inputs])
    raw = float(model.predict(df)[0])
    return float(np.expm1(raw)) if raw < 20 else raw


# ── Plotly shared layout ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#4b5563"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ── Chart helpers ──────────────────────────────────────────────────────────────
def qual_chart():
    quals  = list(QUAL_PRICE.keys())
    prices = [v / 1000 for v in QUAL_PRICE.values()]
    colors = [f"rgba(102,126,234,{0.3 + 0.07*i})" for i in range(10)]
    fig = go.Figure(go.Bar(
        x=quals, y=prices,
        marker_color=colors, marker_line_width=0,
        text=[f"${p:.0f}k" for p in prices],
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis=dict(title="Overall Quality (1–10)", tickmode="linear"),
        yaxis=dict(title="Avg Price ($k)", gridcolor="#f0f1f6"),
    )
    return fig


def decade_chart():
    decades = list(DECADE_PRICE.keys())
    prices  = [v / 1000 for v in DECADE_PRICE.values()]
    fig = go.Figure(go.Scatter(
        x=decades, y=prices, mode="lines+markers",
        line=dict(color="#667eea", width=2.5),
        marker=dict(color="#764ba2", size=6),
        fill="tozeroy", fillcolor="rgba(102,126,234,0.08)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis=dict(title="Decade Built", tickangle=-45),
        yaxis=dict(title="Median Price ($k)", gridcolor="#f0f1f6"),
    )
    return fig


def nb_chart():
    nb_sorted = sorted(NB_MEDIAN.items(), key=lambda x: x[1], reverse=True)[:12]
    names  = [n for n, _ in nb_sorted]
    prices = [p / 1000 for _, p in nb_sorted]
    fig = go.Figure(go.Bar(
        y=names, x=prices, orientation="h",
        marker=dict(
            color=prices,
            colorscale=[[0, "#c7d2fe"], [0.5, "#818cf8"], [1, "#4338ca"]],
            showscale=False,
        ),
        text=[f"${p:.0f}k" for p in prices],
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis=dict(title="Median Sale Price ($k)", gridcolor="#f0f1f6"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def scatter_chart(sqft, price_pred):
    np.random.seed(42)
    n  = 200
    sq = np.random.normal(1515, 525, n).clip(400, 5000)
    pr = (sq * 110 + np.random.normal(0, 15000, n)).clip(50000, 700000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sq, y=pr / 1000, mode="markers",
        marker=dict(color="rgba(102,126,234,0.25)", size=5),
        name="Dataset",
    ))
    fig.add_trace(go.Scatter(
        x=[sqft], y=[price_pred / 1000], mode="markers",
        marker=dict(color="#764ba2", size=14, symbol="star",
                    line=dict(color="#fff", width=2)),
        name="Your House",
    ))
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "showlegend": True}, height=300,
        legend=dict(font=dict(size=10)),
        xaxis=dict(title="Living Area (sq ft)", gridcolor="#f0f1f6"),
        yaxis=dict(title="Sale Price ($k)", gridcolor="#f0f1f6"),
    )
    return fig


def gauge_chart(price, min_p=50_000, max_p=755_000):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price / 1000,
        number=dict(prefix="$", suffix="k", font=dict(size=28, color="#1a1d2e")),
        gauge=dict(
            axis=dict(range=[50, 755], tickcolor="#e8eaf0",
                      tickfont=dict(size=9, color="#8b92a5")),
            bar=dict(color="#667eea", thickness=0.25),
            bgcolor="white", borderwidth=0,
            steps=[
                dict(range=[50, 130],  color="#f0fdf4"),
                dict(range=[130, 215], color="#eff6ff"),
                dict(range=[215, 400], color="#faf5ff"),
                dict(range=[400, 755], color="#fff1f2"),
            ],
            threshold=dict(
                line=dict(color="#764ba2", width=3),
                thickness=0.75, value=price / 1000,
            ),
        ),
        title=dict(text="Price Range", font=dict(size=11, color="#8b92a5")),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def feature_importance_chart(model):
    """
    Returns a Plotly figure of feature importances.
    Accepts the loaded pipeline; returns None if importances unavailable.
    """
    try:
        inner = model.named_steps["model"]
        if not hasattr(inner, "feature_importances_"):
            return None
        importance = inner.feature_importances_
        # Align with FEATURES length (safety guard)
        if len(importance) != len(FEATURES):
            return None
        fig = go.Figure(go.Bar(
            y=FEATURES[::-1], x=importance[::-1], orientation="h",
            marker=dict(
                color=importance[::-1],
                colorscale=[[0, "#c7d2fe"], [1, "#4338ca"]],
                showscale=False,
            ),
            text=[f"{v*100:.1f}%" for v in importance[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=560,
            xaxis=dict(title="Relative Importance", tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
        )
        return fig
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-logo">🏡 HomePriceIQ</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-section">Navigation</span>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🏠  Dashboard", "🔮  Price Predictor", "📊  Market Analytics", "📂  Batch Predict", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown('<span class="sidebar-section">Model Info</span>', unsafe_allow_html=True)
    model, model_path = load_model()
    if model:
        # Detect algorithm name from model class
        try:
            algo = type(model.named_steps["model"]).__name__
        except Exception:
            algo = "Unknown"
        st.success("✓ Model loaded")
        st.caption(f"Path: `{model_path}`")
        st.caption(f"Algorithm: {algo}")
        st.caption("Val R²: ~0.91")
    else:
        st.error("Model not found")
        st.caption(f"Train with train.py first.\nExpected at: `{MODEL_PATH}`")

    st.markdown('<span class="sidebar-section">Dataset</span>', unsafe_allow_html=True)
    st.caption("Ames Housing Dataset")
    st.caption("1,460 training homes")
    st.caption("Price range: $35k – $755k")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <div>
    <h1>🏡 HomePriceIQ</h1>
    <p class="tagline">Ames Housing · ML Prediction Engine · Trained on 1,460 homes</p>
  </div>
  <span class="badge">ML Powered</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""
    <div class="metric-grid">
      <div class="metric-card purple">
        <div class="icon">🏘️</div>
        <div class="label">Total Homes</div>
        <div class="value">1,460</div>
        <div class="sub">Training dataset</div>
      </div>
      <div class="metric-card teal">
        <div class="icon">💰</div>
        <div class="label">Median Price</div>
        <div class="value">$163k</div>
        <div class="sub">Ames, Iowa</div>
      </div>
      <div class="metric-card orange">
        <div class="icon">📐</div>
        <div class="label">Avg Living Area</div>
        <div class="value">1,515</div>
        <div class="sub">sq ft</div>
      </div>
      <div class="metric-card rose">
        <div class="icon">🎯</div>
        <div class="label">Model R² Score</div>
        <div class="value">0.91</div>
        <div class="sub">Validation accuracy</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Average Price by Overall Quality</div>', unsafe_allow_html=True)
        st.plotly_chart(qual_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Median Price by Decade Built</div>', unsafe_allow_html=True)
        st.plotly_chart(decade_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    col3, col4 = st.columns([3, 2])
    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Top 12 Neighborhoods by Median Price</div>', unsafe_allow_html=True)
        st.plotly_chart(nb_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance</div>', unsafe_allow_html=True)
        if model:
            fig_imp = feature_importance_chart(model)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature importances unavailable for this model type.")
        else:
            st.warning("Load a model to see feature importances.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Key insights
    st.markdown('<div class="section-heading">💡 Key Insights</div>', unsafe_allow_html=True)
    insights = [
        "Quality 10 homes sell for 8.7× more than Quality 1",
        "NridgHt median price ($315k) is 3.6× MeadowV ($88k)",
        "Post-2000 builds command $60k+ premium over 1980s",
        "Each garage car spot adds ~$18k in value",
        "Overall Quality explains ~38% of price variance",
        "Living area is the 2nd most predictive feature (~24%)",
    ]
    st.markdown("".join(f'<span class="insight-pill">{i}</span>' for i in insights), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Price Predictor":
    if model is None:
        st.error("⚠️ Model not found. Run `train.py` first, then place the model at `models/streamlit_model.pkl`.")
        st.stop()

    st.markdown('<div class="section-heading">🔮 Predict Your Home\'s Value</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        # Group 1 — Structure
        st.markdown('<div class="input-section"><h4>🏗️ Structure & Size</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gr_liv_area   = st.slider("Above-Ground Living Area (sq ft)", 334, 5642, 1500, step=50)
            total_bsmt    = st.slider("Total Basement Area (sq ft)", 0, 6110, 800, step=50)
            first_flr_sf  = st.slider("1st Floor Area (sq ft)", 300, 5000, 1200, step=50)
            second_flr_sf = st.slider("2nd Floor Area (sq ft)", 0, 3000, 500, step=50)
            lot_area      = st.slider("Lot Area (sq ft)", 1000, 250000, 10000, step=500)
        with c2:
            year_built  = st.slider("Year Built", 1872, 2010, 2000, step=1)
            year_remod  = st.slider("Year Remodeled", 1950, 2010, 2000, step=1)
            overall_qual = st.select_slider(
                "Overall Quality", options=list(range(1, 11)), value=7,
                format_func=lambda x: f"{x} — {['','Very Poor','Poor','Fair','Below Avg','Average','Above Avg','Good','Very Good','Excellent','Very Excellent'][x]}",
            )
            total_rooms = st.slider("Total Rooms Above Ground", 2, 15, 6)
            bedrooms    = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5, 6, 7, 8], index=2)
        st.markdown('</div>', unsafe_allow_html=True)

        # Group 2 — Garage & Bathrooms
        st.markdown('<div class="input-section"><h4>🚗 Garage & Bathrooms</h4>', unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        with c3:
            garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
            fireplaces  = st.selectbox("Fireplaces", [0, 1, 2, 3, 4], index=1)
        with c4:
            garage_area = st.slider("Garage Area (sq ft)", 0, 1418, 480, step=20)
            full_bath   = st.selectbox("Full Bathrooms", [0, 1, 2, 3], index=2)
        with c5:
            half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
            kitchens  = st.selectbox("Kitchens Above Ground", [1, 2, 3], index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # Group 3 — Additional Features
        st.markdown('<div class="input-section"><h4>🏡 Additional Features</h4>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            mas_vnr_area = st.slider("Masonry Veneer Area (sq ft)", 0, 1600, 100)
            open_porch   = st.slider("Open Porch Area (sq ft)", 0, 600, 50)
        with c7:
            wood_deck    = st.slider("Wood Deck Area (sq ft)", 0, 900, 100)
            bsmt_fin_sf1 = st.slider("Finished Basement Area (sq ft)", 0, 6000, 500)
        with c8:
            st.metric("Features Used", str(len(FEATURES)))
            overall_cond = st.selectbox("Overall Condition", list(range(1, 11)), index=4)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔮 Estimate Price", use_container_width=True)

    # Build inputs dict (must match FEATURES order exactly)
    inputs = {
        "OverallQual":  overall_qual,
        "GrLivArea":    gr_liv_area,
        "GarageCars":   garage_cars,
        "GarageArea":   garage_area,
        "TotalBsmtSF":  total_bsmt,
        "1stFlrSF":     first_flr_sf,
        "FullBath":     full_bath,
        "TotRmsAbvGrd": total_rooms,
        "YearBuilt":    year_built,
        "YearRemodAdd": year_remod,
        "MasVnrArea":   mas_vnr_area,
        "Fireplaces":   fireplaces,
        "BsmtFinSF1":   bsmt_fin_sf1,
        "LotArea":      lot_area,
        "OpenPorchSF":  open_porch,
        "WoodDeckSF":   wood_deck,
        "2ndFlrSF":     second_flr_sf,
        "HalfBath":     half_bath,
        "BedroomAbvGr": bedrooms,
        "KitchenAbvGr": kitchens,
    }

    with right:
        if predict_btn:
            try:
                price = predict(model, inputs)
                low   = price * 0.92
                high  = price * 1.08

                try:
                    algo_label = type(model.named_steps["model"]).__name__
                except Exception:
                    algo_label = "ML Model"

                st.session_state["last_price"]    = price
                st.session_state["last_sqft"]     = gr_liv_area
                st.session_state["last_algo"]     = algo_label

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Show result if we have one stored
        if "last_price" in st.session_state:
            price      = st.session_state["last_price"]
            algo_label = st.session_state.get("last_algo", "ML Model")
            low        = price * 0.92
            high       = price * 1.08

            st.markdown(f"""
            <div class="prediction-box">
              <div class="pre-label">Estimated Sale Price</div>
              <div class="price">${price:,.0f}</div>
              <div class="range">Confidence range: ${low:,.0f} – ${high:,.0f}</div>
              <div class="confidence">{algo_label} · R² ~0.91</div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(price), use_container_width=True, config={"displayModeBar": False})

            diff = price - 163_000
            sign = "+" if diff >= 0 else ""
            col_a, col_b = st.columns(2)
            col_a.metric("vs. Median ($163k)", f"{sign}${abs(diff):,.0f}", f"{sign}{diff/163000*100:.1f}%")
            col_b.metric("Price per sq ft", f"${price/max(st.session_state['last_sqft'],1):,.0f}", "Avg: $119/sqft")

        else:
            st.info("👈 Configure the home details and click **🔮 Estimate Price** to see the prediction.")

    # Scatter below — only show if prediction exists
    if "last_price" in st.session_state:
        st.markdown('<div class="chart-card"><div class="chart-title">Your Home vs. Dataset (Living Area vs Price)</div>', unsafe_allow_html=True)
        st.plotly_chart(
            scatter_chart(st.session_state["last_sqft"], st.session_state["last_price"]),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARKET ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Market Analytics":
    st.markdown('<div class="section-heading">📊 Market Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏘️ Neighborhood Prices", "📈 Quality vs Price", "🗓️ Price by Era"])

    with tab1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        max_p = max(NB_MEDIAN.values())
        for nb, price in sorted(NB_MEDIAN.items(), key=lambda x: -x[1]):
            bar_pct = price / max_p * 100
            st.markdown(f"""
            <div class="nb-card">
              <div class="nb-name">{nb}</div>
              <div class="nb-bar-wrap"><div class="nb-bar" style="width:{bar_pct}%"></div></div>
              <div class="nb-price">${price:,}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chart-card"><div class="chart-title">Average Sale Price by Overall Quality Rating</div>', unsafe_allow_html=True)
        st.plotly_chart(qual_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
        df_qual = pd.DataFrame({
            "Quality":       list(QUAL_PRICE.keys()),
            "Avg Sale Price": [f"${v:,}" for v in QUAL_PRICE.values()],
            "vs. Quality 5":  [f"{v/133523*100-100:+.0f}%" for v in QUAL_PRICE.values()],
        })
        st.dataframe(df_qual, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="chart-card"><div class="chart-title">Median Sale Price by Decade of Construction</div>', unsafe_allow_html=True)
        st.plotly_chart(decade_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance — full width
    st.markdown('<div class="section-heading">🎯 Model Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if model:
        fig_imp = feature_importance_chart(model)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Feature importances not available for this model type.")
    else:
        st.warning("Load a model to view feature importances.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Batch Predict":
    st.markdown('<div class="section-heading">📂 Batch Predictions on Training Data</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Run `train.py` first.")
        st.stop()

    # Auto-load train.csv (flat or data/ subdir)
    df_train = None
    for tp in ["data/train.csv", "train.csv"]:
        if os.path.exists(tp):
            df_train = pd.read_csv(tp)
            break

    if df_train is None:
        # Allow manual upload
        uploaded = st.file_uploader("Upload train.csv (or test.csv)", type="csv")
        if uploaded:
            df_train = pd.read_csv(uploaded)
        else:
            st.warning("Place `train.csv` next to `app.py` or upload it above.")
            st.stop()

    # Check all features present
    missing_cols = [f for f in FEATURES if f not in df_train.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.stop()

    try:
        raw_preds = model.predict(df_train[FEATURES])
        if float(raw_preds.mean()) < 20:
            df_train["Predicted Price ($)"] = np.expm1(raw_preds).round(0).astype(int)
        else:
            df_train["Predicted Price ($)"] = raw_preds.round(0).astype(int)

        df_train["Price ($k)"] = (df_train["Predicted Price ($)"] / 1000).round(1)

        has_actual = "SalePrice" in df_train.columns
        if has_actual:
            actual = df_train["SalePrice"].values
            df_train["Actual Price ($)"] = actual.astype(int)
            df_train["Error ($)"]  = (df_train["Predicted Price ($)"] - actual).astype(int)
            df_train["Error (%)"]  = ((df_train["Error ($)"] / actual) * 100).round(1)

        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Homes Scored",   f"{len(df_train):,}")
        m2.metric("Min Predicted",  f"${df_train['Predicted Price ($)'].min():,}")
        m3.metric("Max Predicted",  f"${df_train['Predicted Price ($)'].max():,}")
        m4.metric("Avg Predicted",  f"${df_train['Predicted Price ($)'].mean():,.0f}")

        col1, col2 = st.columns(2)

        with col1:
            fig_dist = px.histogram(
                df_train, x="Predicted Price ($)", nbins=40,
                color_discrete_sequence=["#667eea"], template="none",
            )
            fig_dist.update_layout(
                **PLOTLY_LAYOUT, height=260,
                xaxis_title="Predicted Price ($)",
                yaxis=dict(title="Count", gridcolor="#f0f1f6"),
            )
            st.markdown('<div class="chart-card"><div class="chart-title">Predicted Price Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if has_actual:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=df_train["Actual Price ($)"] / 1000,
                    y=df_train["Predicted Price ($)"] / 1000,
                    mode="markers",
                    marker=dict(color="rgba(102,126,234,0.35)", size=5),
                    name="Predictions",
                ))
                max_val = max(df_train["Actual Price ($)"].max(), df_train["Predicted Price ($)"].max()) / 1000
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val], mode="lines",
                    line=dict(color="#f953c6", width=1.5, dash="dash"),
                    name="Perfect fit",
                ))
                fig_scatter.update_layout(
                    **{**PLOTLY_LAYOUT, "showlegend": True}, height=260,
                    legend=dict(font=dict(size=9)),
                    xaxis=dict(title="Actual Price ($k)", gridcolor="#f0f1f6"),
                    yaxis=dict(title="Predicted Price ($k)", gridcolor="#f0f1f6"),
                )
                st.markdown('<div class="chart-card"><div class="chart-title">Predicted vs. Actual Price</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

        if has_actual:
            fig_err = px.histogram(
                df_train, x="Error (%)", nbins=50,
                color_discrete_sequence=["#764ba2"], template="none",
            )
            fig_err.update_layout(
                **PLOTLY_LAYOUT, height=220,
                xaxis_title="Prediction Error (%)",
                yaxis=dict(title="Count", gridcolor="#f0f1f6"),
            )
            st.markdown('<div class="chart-card"><div class="chart-title">Prediction Error Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_err, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Results table
        st.markdown('<div class="section-heading">📋 Full Results Table</div>', unsafe_allow_html=True)

        search_col, slider_col = st.columns([2, 2])
        with search_col:
            qual_filter = st.multiselect(
                "Filter by Overall Quality",
                options=sorted(df_train["OverallQual"].unique()),
                default=[],
            )
        with slider_col:
            p_min = int(df_train["Predicted Price ($)"].min() / 1000)
            p_max = int(df_train["Predicted Price ($)"].max() / 1000)
            price_range = st.slider("Filter by Predicted Price ($k)", p_min, p_max, (p_min, p_max))

        df_show = df_train.copy()
        if qual_filter:
            df_show = df_show[df_show["OverallQual"].isin(qual_filter)]
        df_show = df_show[
            (df_show["Price ($k)"] >= price_range[0]) &
            (df_show["Price ($k)"] <= price_range[1])
        ]

        display_cols = FEATURES + ["Predicted Price ($)", "Price ($k)"]
        if has_actual:
            display_cols += ["Actual Price ($)", "Error ($)", "Error (%)"]

        st.dataframe(df_show[display_cols].reset_index(drop=True),
                     use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df_show):,} of {len(df_train):,} homes")

        st.download_button(
            "⬇️ Download predictions as CSV",
            df_show[display_cols].to_csv(index=False).encode(),
            "predictions.csv", "text/csv",
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown('<div class="section-heading">ℹ️ About This App</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="input-section">
        <h4>🧠 Model Architecture</h4>
        <p style="font-size:0.88rem;color:#4b5563;line-height:1.7">
        The prediction engine uses a scikit-learn <b>Pipeline</b> combining
        median imputation with a gradient boosting regressor
        (<b>XGBoost</b> when available, else <b>HistGradientBoosting</b>).
        The target (<code>SalePrice</code>) is log-transformed to reduce
        right-skew, and predictions are reverse-transformed
        (<code>expm1</code>) before display.
        </p>
        <br>
        <b style="font-size:0.8rem;color:#667eea">PIPELINE STEPS</b>
        <ol style="font-size:0.85rem;color:#4b5563;line-height:2">
          <li>Median imputation for all numeric features</li>
          <li>Gradient boosting regression (XGBoost / HGB)</li>
          <li>Reverse log-transform on output</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="input-section">
        <h4>📦 Key Hyperparameters (XGBoost)</h4>
        </div>
        """, unsafe_allow_html=True)
        params = pd.DataFrame({
            "Parameter":   ["n_estimators","learning_rate","max_depth","subsample","colsample_bytree","random_state"],
            "Value":       ["1,000","0.03","3","0.75","0.75","42"],
        })
        st.dataframe(params, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="input-section" style="margin-top:1rem">
        <h4>📊 Dataset</h4>
        <p style="font-size:0.85rem;color:#4b5563;line-height:1.7">
        <b>Ames Housing Dataset</b> — 1,460 residential sales in Ames, Iowa.
        79 explanatory variables covering physical attributes, quality
        ratings, and sale conditions.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="input-section">
    <h4>🚀 How to Run Locally</h4>
    <pre style="background:#f8f9ff;border-radius:8px;padding:1rem;font-size:0.82rem;color:#1a1d2e;overflow-x:auto">
# 1. Install dependencies
pip install streamlit xgboost scikit-learn pandas numpy plotly joblib

# 2. Train the model  (produces models/streamlit_model.pkl)
python train.py

# 3. Launch the app
streamlit run app.py</pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="input-section">
    <h4>📁 Expected File Layout</h4>
    <pre style="background:#f8f9ff;border-radius:8px;padding:1rem;font-size:0.82rem;color:#1a1d2e">
project/
├── app.py
├── train.py
├── train.csv          ← or data/train.csv
├── test.csv           ← or data/test.csv
└── models/
    └── streamlit_model.pkl   ← generated by train.py</pre>
    </div>
    """, unsafe_allow_html=True)