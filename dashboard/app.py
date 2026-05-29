import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="KHI AIR | Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["RandomForest"] = joblib.load("random_forest.pkl")
    except:
        models["RandomForest"] = None

    try:
        models["Ridge"] = joblib.load("ridge.pkl")
    except:
        models["Ridge"] = None
    return models

models = load_models()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');


/* Force the Main Content Screen Background (Light, airy, fresh gradient) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFF5F6 0%, #EBF4FF 60%, #DEEEFF 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Force the Sidebar Background (Sophisticated Dark Slate) */
[data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
    background-color: #1A2232 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Fix Sidebar text colors to display beautifully over dark background */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}

/* Soften the expanders inside the dark sidebar */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

/* --- DYNAMIC INTERACTIVE BUTTON HOVER EFFECT --- */
div[data-testid="stSidebar"] button {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #A0AEC0 !important; /* Darker grey when not hovering */
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    font-weight: 600 !important;
}

div[data-testid="stSidebar"] button:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important; /* Changes to pure white on mouse hover */
    border-color: rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
    transform: translateY(-1px);
}

/* --- DASHBOARD UI ELEMENTS --- */

/* Clean Floating Cards on Light Main Screen */
.weather-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.7);
    box-shadow: 0 10px 30px rgba(165, 200, 235, 0.2);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.weather-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(165, 200, 235, 0.35);
    background: rgba(255, 255, 255, 0.95);
}

.nav-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    letter-spacing: -1px;
    color: #111622;
    margin-bottom: 15px;
    margin-top: -15px;
}

.nav-logo-accent {
    background: linear-gradient(90deg, #FF6F91, #FF9EBB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

            /* ---------------- REMOVE STREAMLIT TOP NAVBAR ---------------- */

header[data-testid="stHeader"] {
    display: none !important;
}

/* Removes extra top spacing after hiding header */

[data-testid="stAppViewContainer"] {
    margin-top: -3.5rem;
}

/* Optional: remove hamburger + deploy button area */

[data-testid="collapsedControl"] {
    display: none !important;
}
            
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #111622;
    letter-spacing: -0.5px;
    margin-bottom: 16px;
}

.stat-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #5A6A85;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.stat-value {
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1;
    color: #111622;
    margin-bottom: 12px;
}

.stat-status {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-good { background: #E6FFFA; color: #319795; }
.status-mod { background: #FEFCBF; color: #B7791F; }
.status-unhealthy { background: #FEEBC8; color: #DD6B20; }
.status-hazard { background: #FED7D7; color: #E53E3E; }


</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR (DARK SLATE)
# ---------------------------
with st.sidebar:
    st.markdown("<div style='padding: 10px 0;'><h2 style='font-weight:700; margin:0;'>🌤️ KHI AIR</h2></div>", unsafe_allow_html=True)
    
    with st.expander("📅 View Options", expanded=True):
        selected_day = st.selectbox(
            "Select Range",
            ["Today", "Next 3 Days", "Historical (7 Days)"]
        )
        pollutant_focus = st.multiselect(
            "Focus Pollutants",
            ["PM2.5", "PM10", "NO2", "O3", "SO2"],
            default=["PM2.5", "PM10"]
        )

    with st.expander("🤖 Intelligence Engine", expanded=True):
        selected_model = st.selectbox(
            "Prediction Model",
            ["RandomForest", "Ridge"]
        )

    with st.expander("📂 Dataset Management", expanded=False):
        uploaded_file = st.file_uploader("Upload Air Quality CSV", type=["csv"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Update Metrics", use_container_width=True):
        st.rerun()

# ---------------------------
# DATA PROCESSING
# ---------------------------
def get_data(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        now = datetime.now(timezone.utc)
        times = [now + timedelta(hours=i) for i in range(24)]
        aqi_vals = np.random.randint(140, 260, 24)
        df = pd.DataFrame({"time": times, "aqi": aqi_vals})
    return df

df = get_data(uploaded_file)

def predict(model_name, data):
    model = models.get(model_name)
    if model is None:
        return np.random.randint(120, 250, len(data))
    try:
        X = data[["aqi"]].values
        return model.predict(X)
    except:
        return np.random.randint(120, 250, len(data))

df["prediction"] = predict(selected_model, df)

# ---------------------------
# MAIN DASHBOARD HEADER
# ---------------------------
st.markdown('<div class="nav-logo">Karachi <span class="nav-logo-accent">Air</span></div>', unsafe_allow_html=True)

# ---------------------------
# AQI STATUS SETUP
# ---------------------------
def get_aqi_badge(val):
    if val < 50:
        return '<span class="stat-status status-good">Good 😊</span>'
    elif val < 100:
        return '<span class="stat-status status-mod">Moderate 😐</span>'
    elif val < 150:
        return '<span class="stat-status status-unhealthy">Unhealthy 😷</span>'
    else:
        return '<span class="stat-status status-hazard">Hazardous ☠️</span>'

# ---------------------------
# HERO METRIC SECTION
# ---------------------------
col1, col2, col3 = st.columns([1.1, 1.1, 2], gap="large")
current_aqi = df["aqi"].iloc[0]

with col1:
    st.markdown(f"""
    <div class="weather-card">
        <div class="stat-label">Air Quality Index</div>
        <div class="stat-value" style="background: linear-gradient(135deg, #FF6F91, #D84B6F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{current_aqi}</div>
        {get_aqi_badge(current_aqi)}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="weather-card">
        <div class="stat-label">Temperature & Comfort</div>
        <div class="stat-value" style="background: linear-gradient(135deg, #4A90E2, #2172D2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">32°C</div>
        <div style="color: #4A5568; font-size:0.9rem; font-weight:500;">💧 Humidity: <span style="font-weight:700;">65%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["aqi"],
        name="Actual AQI",
        fill="tozeroy",
        line=dict(color="#FF6F91", width=4, shape='spline'),
        fillcolor="rgba(255,111,145,0.05)"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["prediction"],
        name="Predicted Forecast",
        line=dict(color="#4A90E2", width=2.5, dash="dot", shape='spline')
    ))
    fig.update_layout(
        height=155,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False)
    )

    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.markdown('<div class="stat-label" style="margin-bottom:0px;">24H Trend Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# DETAILED ANALYSIS SECTION
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Pollutant Breakdown</div>', unsafe_allow_html=True)

    poll_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2"],
        "Level": [95, 130, 45, 60, 15]
    }).sort_values(by="Level", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=poll_df["Level"],
        y=poll_df["Pollutant"],
        orientation="h",
        marker=dict(
            color=poll_df["Level"],
            colorscale=[[0, '#90CAF9'], [1, '#4A90E2']],
            line=dict(width=0),
            cornerradius=10
        ),
        hovertemplate='Level: %{x}<extra></extra>'
    ))
    fig2.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Concentration (µg/m³)", tickfont=dict(color="#718096")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#718096"))
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="weather-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔮 Model Insights & Weights</div>', unsafe_allow_html=True)
    
    metrics = [
        ("Traffic Dynamics", 80),
        ("Industrial Emissions", 60),
        ("Meteorological Boundary Layers", 40)
    ]
    
    for label, val in metrics:
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight:600; color: #4A5568; margin-bottom: 4px;">
                <span>{label}</span>
                <span>{val}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(val)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 50px; padding: 20px 0; color: #A0AEC0; font-size: 0.85rem; font-weight: 500;">
    Designed cleanly by Javariya Sohail • Running Live Analytics
</div>
""", unsafe_allow_html=True)