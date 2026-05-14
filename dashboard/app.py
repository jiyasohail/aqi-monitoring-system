"""
dashboard/app.py — Streamlit AQI Dashboard for Karachi
Run: streamlit run dashboard/app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

  .main { background: #0a0e1a; }
  .block-container { padding: 2rem 3rem; max-width: 1400px; }

  .metric-card {
    background: linear-gradient(135deg, #0f1529 0%, #141c35 100%);
    border: 1px solid #1e2d5a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
  }
  .metric-value { font-size: 2.8rem; font-weight: 800; line-height: 1; }
  .metric-label { font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7db3; margin-top: 0.4rem; }

  .aqi-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .section-header {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a5a8a;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1a2340;
    padding-bottom: 0.5rem;
  }

  .alert-box {
    background: rgba(255, 60, 60, 0.1);
    border: 1px solid rgba(255, 60, 60, 0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
  }

  .stMetric { background: transparent !important; }
  div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                           "#00e676", "#00c853"),
    (51,  100, "Moderate",                        "#ffee58", "#f9a825"),
    (101, 150, "Unhealthy for Sensitive Groups",  "#ffa726", "#e65100"),
    (151, 200, "Unhealthy",                       "#ef5350", "#b71c1c"),
    (201, 300, "Very Unhealthy",                  "#ce93d8", "#6a1b9a"),
    (301, 500, "Hazardous",                       "#b71c1c", "#4a0000"),
]

def aqi_info(val: float) -> dict:
    for lo, hi, label, c1, c2 in AQI_CATEGORIES:
        if lo <= val <= hi:
            return {"label": label, "color": c1, "dark": c2}
    return {"label": "Hazardous", "color": "#b71c1c", "dark": "#4a0000"}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_live_data():
    """Try to fetch live data from OpenWeather; fall back to synthetic."""
    api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY", "")
    if api_key:
        try:
            from pipelines.fetcher  import fetch_current, fetch_forecast, fetch_weather_current
            from pipelines.features import compute_aqi
            df_cur  = fetch_current(api_key)
            df_fc   = fetch_forecast(api_key)
            weather = fetch_weather_current(api_key)
            df_cur["aqi"] = compute_aqi(df_cur)
            df_fc["aqi"]  = compute_aqi(df_fc)
            return df_cur, df_fc, weather, False
        except Exception as e:
            st.warning(f"Live fetch failed ({e}), using synthetic data.")
    return _synthetic_data()


def _synthetic_data():
    """Generate synthetic data for demo when API key is not set."""
    now   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times = [now - timedelta(hours=i) for i in range(48, 0, -1)]
    np.random.seed(42)
    base  = 160 + np.cumsum(np.random.randn(48) * 4)
    base  = np.clip(base, 80, 280)

    df_cur = pd.DataFrame([{
        "time": now, "aqi": float(base[-1]),
        "pm2_5": 95.0, "pm10": 130.0, "co": 800.0,
        "no2": 45.0, "o3": 60.0, "so2": 15.0,
    }])

    fc_times = [now + timedelta(hours=i) for i in range(1, 73)]
    fc_aqi   = base[-1] + np.cumsum(np.random.randn(72) * 3)
    fc_aqi   = np.clip(fc_aqi, 60, 320)
    df_fc = pd.DataFrame({"time": fc_times, "aqi": fc_aqi})
    df_fc["pm2_5"] = df_fc["aqi"] * 0.6 + np.random.randn(72) * 5
    df_fc["pm10"]  = df_fc["aqi"] * 0.8 + np.random.randn(72) * 8

    weather = {"temp": 33.2, "humidity": 72, "pressure": 1008, "wind_speed": 3.1, "visibility": 5000}
    return df_cur, df_fc, weather, True


@st.cache_data(ttl=3600)
def get_predictions():
    """Load pre-computed model predictions if available."""
    pred_csv = "data/predictions.csv"
    if os.path.exists(pred_csv):
        return pd.read_csv(pred_csv, parse_dates=["time"])
    return None


@st.cache_data(ttl=86400)
def get_shap_importance():
    shap_csv = "models/shap_importance.csv"
    if os.path.exists(shap_csv):
        return pd.read_csv(shap_csv)
    return None


@st.cache_data(ttl=86400)
def get_historical_metrics():
    metrics_csv = "models/metrics.csv"
    if os.path.exists(metrics_csv):
        return pd.read_csv(metrics_csv)
    return None


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,0.8)",
    font=dict(family="Syne, sans-serif", color="#8899cc"),
    xaxis=dict(gridcolor="#1a2340", linecolor="#1e2d5a"),
    yaxis=dict(gridcolor="#1a2340", linecolor="#1e2d5a"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def aqi_color_scale():
    return [
        [0.0,  "#00e676"], [0.1,  "#00e676"],
        [0.2,  "#ffee58"], [0.3,  "#ffa726"],
        [0.4,  "#ef5350"], [0.6,  "#ce93d8"],
        [1.0,  "#b71c1c"],
    ]


# ── MAIN APP ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
      <h1 style="font-size: 2.5rem; font-weight: 800; color: #e8eeff; margin: 0; letter-spacing: -0.02em;">
        🌫️ Karachi Air Quality
      </h1>
      <p style="color: #4a5a8a; font-size: 0.9rem; margin-top: 0.3rem; font-family: 'Space Mono', monospace;">
        72-HOUR FORECAST  ·  REAL-TIME MONITORING  ·  ML-POWERED
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df_cur, df_fc, weather, is_synthetic = get_live_data()
    df_preds = get_predictions()

    current_aqi = float(df_cur["aqi"].iloc[-1]) if not df_cur.empty else 155.0
    info = aqi_info(current_aqi)

    if is_synthetic:
        st.info("⚡ Demo mode — set `OPENWEATHER_API_KEY` in secrets for live data.", icon="💡")

    # ── Row 1: Current conditions ──────────────────────────────────────────────
    st.markdown('<p class="section-header">Current Conditions — Karachi</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color: {info['color']};">{int(current_aqi)}</div>
          <div class="metric-label">AQI</div>
          <div style="margin-top:0.5rem;">
            <span class="aqi-badge" style="background:{info['color']}22; color:{info['color']};">
              {info['label']}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    metrics_map = [
        ("🌡️", f"{weather.get('temp', '--')}°C",           "Temperature"),
        ("💧", f"{weather.get('humidity', '--')}%",         "Humidity"),
        ("💨", f"{weather.get('wind_speed', '--')} m/s",    "Wind Speed"),
        ("🔵", f"{int(weather.get('pressure', 0))} hPa",   "Pressure"),
    ]
    for col, (icon, val, label) in zip([c2, c3, c4, c5], metrics_map):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.8rem;">{icon}</div>
              <div class="metric-value" style="color:#a0b4e8; font-size:1.8rem;">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Hazard alert ──────────────────────────────────────────────────────────
    if current_aqi > 150:
        msg = "⚠️  Air quality is currently **Unhealthy**. Limit prolonged outdoor exposure." if current_aqi <= 200 else "🚨  **Hazardous** air quality detected. Avoid outdoor activities."
        st.markdown(f'<div class="alert-box">{msg}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Row 2: 72h Forecast Chart ──────────────────────────────────────────────
    st.markdown('<p class="section-header">72-Hour AQI Forecast</p>', unsafe_allow_html=True)

    # Prefer model predictions, fallback to OpenWeather forecast
    plot_df = df_preds if df_preds is not None else df_fc.rename(columns={"aqi": "aqi_predicted"})
    if "aqi_predicted" not in plot_df.columns and "aqi" in plot_df.columns:
        plot_df = plot_df.rename(columns={"aqi": "aqi_predicted"})

    if not plot_df.empty:
        # Color bands
        fig = go.Figure()

        # Background AQI zones
        for lo, hi, label, c1_, c2_ in AQI_CATEGORIES:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=c1_, opacity=0.05, line_width=0)

        # Forecast line
        colors = [aqi_info(v)["color"] for v in plot_df["aqi_predicted"]]
        fig.add_trace(go.Scatter(
            x=plot_df["time"],
            y=plot_df["aqi_predicted"],
            mode="lines+markers",
            line=dict(color="#4d7fff", width=2.5),
            marker=dict(size=5, color=colors, line=dict(width=1, color="rgba(0,0,0,0.5)")),
            fill="tozeroy",
            fillcolor="rgba(77, 127, 255, 0.08)",
            name="Predicted AQI",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>",
        ))

        # Current marker
        fig.add_vline(x=datetime.now(timezone.utc), line_dash="dash", line_color="#ffffff33", line_width=1)

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            showlegend=False,
            yaxis_title="AQI",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Pollutants + Day breakdown ─────────────────────────────────────
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<p class="section-header">Pollutant Breakdown</p>', unsafe_allow_html=True)
        pollutants = {}
        row = df_cur.iloc[-1] if not df_cur.empty else {}
        for p in ["pm2_5", "pm10", "co", "no2", "o3", "so2"]:
            if p in row:
                pollutants[p.upper().replace("_", ".")] = float(row[p])

        if pollutants:
            fig2 = go.Figure(go.Bar(
                x=list(pollutants.keys()),
                y=list(pollutants.values()),
                marker=dict(
                    color=list(pollutants.values()),
                    colorscale=[[0, "#00e676"], [0.3, "#ffee58"], [0.6, "#ef5350"], [1.0, "#b71c1c"]],
                    showscale=False,
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}" for v in pollutants.values()],
                textposition="outside",
                textfont=dict(family="Space Mono", size=11),
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False, yaxis_title="μg/m³")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<p class="section-header">3-Day Daily Average</p>', unsafe_allow_html=True)
        if not plot_df.empty:
            plot_df["date"] = pd.to_datetime(plot_df["time"]).dt.date
            daily = plot_df.groupby("date")["aqi_predicted"].mean().reset_index()
            daily = daily.head(3)
            daily["info"] = daily["aqi_predicted"].apply(aqi_info)
            for _, row in daily.iterrows():
                inf = row["info"]
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:space-between;
                            background: #0f1529; border:1px solid #1e2d5a; border-radius:10px;
                            padding: 0.8rem 1.2rem; margin-bottom: 0.6rem;">
                  <div style="font-size:0.9rem; color:#a0b4e8;">{row['date'].strftime('%A, %b %d')}</div>
                  <div style="display:flex; align-items:center; gap:0.8rem;">
                    <span style="font-family:'Space Mono'; font-size:1.3rem; color:{inf['color']}; font-weight:700;">{int(row['aqi_predicted'])}</span>
                    <span class="aqi-badge" style="background:{inf['color']}22; color:{inf['color']}; font-size:0.7rem;">{inf['label']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── Row 4: Model performance + SHAP ───────────────────────────────────────
    st.markdown('<p class="section-header">Model Analytics</p>', unsafe_allow_html=True)
    ml1, ml2 = st.columns(2)

    with ml1:
        metrics = get_historical_metrics()
        if metrics is not None:
            st.markdown("**Model Comparison**")
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=metrics["model"], y=metrics["rmse"],
                name="RMSE", marker_color="#4d7fff",
            ))
            fig3.add_trace(go.Bar(
                x=metrics["model"], y=metrics["mae"],
                name="MAE", marker_color="#ff6b6b",
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, height=250, barmode="group",
                               legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run training pipeline to see model comparison metrics.")

    with ml2:
        shap_df = get_shap_importance()
        if shap_df is not None:
            st.markdown("**Top Feature Importances (SHAP)**")
            top10 = shap_df.head(10)
            fig4 = go.Figure(go.Bar(
                x=top10["shap_mean_abs"],
                y=top10["feature"],
                orientation="h",
                marker=dict(
                    color=top10["shap_mean_abs"],
                    colorscale=[[0, "#1a2340"], [1, "#4d7fff"]],
                    showscale=False,
                ),
            ))
            fig4.update_layout(**PLOTLY_LAYOUT, height=250, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Run training pipeline with SHAP to see feature importances.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color:#2a3555; font-size:0.7rem; font-family:'Space Mono'; margin-top:3rem;">
      KARACHI AQI FORECAST  ·  POWERED BY OPENWEATHER + HOPSWORKS + SKLEARN
      <br>Data refreshes every hour via GitHub Actions
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
