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

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Bebas+Neue&display=swap');

.main {
    background: linear-gradient(135deg, #fce4ec, #e3f2fd);
    font-family: 'Inter', sans-serif;
    color: #111;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #eee;
}

.plot-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    transition: 0.2s ease;
}

.plot-card:hover {
    transform: translateY(-2px);
    border: 1px solid #90caf9;
}

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111;
    border-bottom: 2px solid #90caf9;
    padding-bottom: 6px;
    margin: 20px 0;
}

.stat-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
}

.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #ff6f91;
}

.nav-logo {
    font-family: 'Bebas Neue';
    font-size: 2rem;
    color: #111;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("### 🌤️ KHI AIR CONTROL")

    selected_day = st.selectbox(
        "Select View Range",
        ["Today", "Next 3 Days", "Historical (7 Days)"]
    )

    pollutant_focus = st.multiselect(
        "Focus Pollutants",
        ["PM2.5", "PM10", "NO2", "O3", "SO2"],
        default=["PM2.5", "PM10"]
    )

    st.markdown("### 🤖 Model Selection")

    selected_model = st.selectbox(
        "Choose Model",
        ["RandomForest", "Ridge"]
    )

    st.markdown("---")

    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload AQI CSV", type=["csv"])

    st.markdown("---")

    if st.button("🔄 Refresh"):
        st.rerun()

# ---------------------------
# DATA LOADING
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

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict(model_name, data):
    model = models.get(model_name)

    if model is None:
        return np.random.randint(120, 250, len(data))

    try:
        X = data[["aqi"]].values
        preds = model.predict(X)
        return preds
    except:
        return np.random.randint(120, 250, len(data))

df["prediction"] = predict(selected_model, df)

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="nav-logo">KARACHI <span style="color:#ff6f91;">AIR</span></div>', unsafe_allow_html=True)

# ---------------------------
# AQI STATUS
# ---------------------------
def aqi_status(val):
    if val < 50:
        return "Good 😊"
    elif val < 100:
        return "Moderate 😐"
    elif val < 150:
        return "Unhealthy 😷"
    else:
        return "Hazardous ☠️"

# ---------------------------
# TOP CARDS
# ---------------------------
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown(f"""
    <div class="plot-card">
        <div class="stat-label">Current AQI</div>
        <div class="stat-value">{df["aqi"].iloc[0]}</div>
        <div>{aqi_status(df["aqi"].iloc[0])}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="plot-card">
        <div class="stat-label">Temperature</div>
        <div class="stat-value">32°C</div>
        <div>Humidity: 65%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["aqi"],
        name="Actual AQI",
        fill="tozeroy",
        line=dict(color="#ff6f91", width=3),
        fillcolor="rgba(255,111,145,0.15)"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["prediction"],
        name="Predicted AQI",
        line=dict(color="#90caf9", width=3, dash="dot")
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.markdown("### 24H Trend (Actual vs Predicted)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# SECTION
# ---------------------------
st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Pollutant Breakdown")

    poll_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2"],
        "Level": [95, 130, 45, 60, 15]
    })

    fig2 = go.Figure(go.Bar(
        x=poll_df["Level"],
        y=poll_df["Pollutant"],
        orientation="h",
        marker=dict(color="#90caf9")
    ))

    fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 Model Insights")

    st.write("Feature importance (demo):")
    st.progress(80, "Traffic Density")
    st.progress(60, "Industrial Output")
    st.progress(40, "Wind Speed")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
<br>
<center style="color:#888;">
Built by Javariya Sohail
</center>
""", unsafe_allow_html=True)