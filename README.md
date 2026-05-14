# 🌫️ Karachi AQI Forecast

End-to-end serverless ML pipeline for predicting Air Quality Index (AQI) in Karachi for the next 72 hours.

**Live dashboard · Automated retraining · Feature Store · SHAP explainability**

---

## Architecture

```
OpenWeather API
      │
      ▼
 feature_pipeline.py  ◄──── GitHub Actions (hourly)
      │  (fetch + engineer features)
      ▼
 Hopsworks Feature Store  (or local CSV in dev mode)
      │
      ▼
 train_pipeline.py  ◄──── GitHub Actions (daily 02:00 UTC)
      │  (RF · Ridge · XGBoost · TF MLP)
      ▼
 Hopsworks Model Registry  →  predict.py  →  predictions.csv
                                                   │
                                                   ▼
                                          Streamlit Dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data API | OpenWeather Air Pollution API |
| Feature Store | Hopsworks |
| ML Models | Random Forest, Ridge, XGBoost, TensorFlow MLP |
| Explainability | SHAP |
| Automation | GitHub Actions |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.12 |

---

## Project Structure

```
aqi_karachi/
├── pipelines/
│   ├── fetcher.py           # OpenWeather API helpers
│   ├── features.py          # Feature engineering + EPA AQI formula
│   ├── feature_pipeline.py  # Hourly ingestion pipeline
│   ├── backfill.py          # Historical data backfill
│   ├── train_pipeline.py    # Multi-model training + SHAP
│   └── predict.py           # 72h recursive forecasting
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── .github/workflows/
│   ├── hourly_feature_pipeline.yml
│   └── train_daily.yml
├── data/                    # Local CSV fallbacks (gitignored in prod)
├── models/                  # Saved models + metrics + SHAP
└── requirements.txt
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/aqi_karachi.git
cd aqi_karachi
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file (never commit this):

```env
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=aqi_karachi
```

Get a free OpenWeather key at https://openweathermap.org/api

Get a free Hopsworks account at https://app.hopsworks.ai

### 3. Backfill historical data

```bash
python -m pipelines.backfill --days 365
```

### 4. Train models

```bash
python -m pipelines.train_pipeline
```

### 5. Run dashboard

```bash
streamlit run dashboard/app.py
```

---

## GitHub Actions CI/CD

Add these secrets in **Settings → Secrets → Actions**:

| Secret | Description |
|---|---|
| `OPENWEATHER_API_KEY` | OpenWeather API key |
| `HOPSWORKS_API_KEY` | Hopsworks API key |
| `HOPSWORKS_PROJECT` | Hopsworks project name |

Workflows run automatically:
- **Hourly**: `hourly_feature_pipeline.yml` — fetches new data, engineers features
- **Daily 02:00 UTC**: `train_daily.yml` — retrains models, saves predictions

---

## Models & Features

**10 raw features** from OpenWeather: `pm2_5, pm10, co, no, no2, o3, so2, nh3` + weather (temp, humidity, wind, pressure)

**~80 engineered features**:
- Time cyclical: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`
- Lag features: 1h, 3h, 6h, 12h, 24h, 48h, 72h
- Rolling stats: mean/std/max over 6h, 12h, 24h, 48h windows
- AQI change rate

**Target**: `aqi_next_24h` — EPA-standard AQI 24 hours ahead

**AQI** is computed using **EPA linear interpolation** between PM2.5 and PM10 breakpoints.

---

## AQI Scale

| AQI | Category | Health Impact |
|---|---|---|
| 0–50 | Good | No risk |
| 51–100 | Moderate | Unusually sensitive people should consider limiting prolonged exertion |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive groups should limit prolonged outdoor exertion |
| 151–200 | Unhealthy | Everyone may begin to experience health effects |
| 201–300 | Very Unhealthy | Health alert — everyone may experience serious effects |
| 301+ | Hazardous | Health emergency |

---

## License

MIT
