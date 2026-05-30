# 🌍 Karachi AQI Forecasting System

### End-to-End MLOps Pipeline for Real-Time Air Quality Forecasting

An automated, cloud-native MLOps solution that predicts **Karachi's Air Quality Index (AQI) up to 72 hours ahead** using machine learning, feature engineering, continuous retraining, and real-time monitoring.

The system continuously ingests environmental data, retrains models on fresh observations, and serves forecasts through an interactive dashboard.

🚀 **Live Dashboard:** https://huggingface.co/spaces/jsohail/KarachiAQIForecast_10Pearls

---

## 📌 Project Overview

Air quality directly impacts public health, transportation planning, and environmental decision-making. This project was developed to provide near real-time AQI forecasting for Karachi through a fully automated MLOps pipeline.

The system:

* Collects live pollution data from OpenWeather APIs
* Engineers advanced temporal and statistical features
* Stores features in a centralized Feature Store
* Retrains multiple machine learning models daily
* Tracks model performance automatically
* Deploys forecasts to a live interactive dashboard

---

## 🏗️ System Architecture

```text
                 OpenWeather API
                        │
                        ▼
          Hourly Feature Pipeline
             (GitHub Actions)
                        │
                        ▼
           Hopsworks Feature Store
                        │
                        ▼
            Daily Training Pipeline
             (GitHub Actions)
                        │
                        ▼
      Random Forest | Ridge | XGBoost
                        │
                        ▼
         Hopsworks Model Registry
                        │
                        ▼
     Gradio Dashboard (Hugging Face)
```

---

## ⚡ Key Features

### 📊 Real-Time Data Pipeline

* Automated hourly data ingestion
* Live pollutant monitoring
* Historical data backfilling
* Centralized feature storage using Hopsworks

### 🧠 Advanced Feature Engineering

* EPA-standard AQI calculation
* Cyclical time encoding (hour, day, month)
* Multi-step lag features
* Rolling window statistics
* AQI trend and change-rate calculations

### 🤖 Machine Learning Models

The system trains and evaluates multiple forecasting models:

| Model            | Purpose                        |
| ---------------- | ------------------------------ |
| Random Forest    | Non-linear ensemble learning   |
| Ridge Regression | Linear baseline model          |
| XGBoost          | Gradient boosting optimization |

Each model is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

The best-performing model is automatically registered and deployed.

### 📈 Model Explainability

* SHAP feature importance analysis
* Model interpretability reporting
* Automated feature ranking generation

### 🔄 Automated MLOps Workflow

#### Hourly Pipeline

* Fetches latest AQI data
* Performs feature engineering
* Updates Feature Store

#### Daily Pipeline

* Retrains all models
* Evaluates performance
* Generates SHAP insights
* Pushes best model to Model Registry

---

## 🛠️ Technology Stack

| Category             | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python 3.11                   |
| Data Source          | OpenWeather Air Pollution API |
| Feature Store        | Hopsworks                     |
| Model Registry       | Hopsworks                     |
| Machine Learning     | Scikit-Learn, XGBoost         |
| Explainability       | SHAP                          |
| Automation           | GitHub Actions                |
| Dashboard            | Gradio                        |
| Visualization        | Plotly                        |

---

## 📂 Project Structure

```text
aqi-monitoring-system/
│
├── pipelines/
│   ├── fetcher.py
│   ├── features.py
│   ├── feature_pipeline.py
│   ├── backfill.py
│   ├── train_pipeline.py
│   └── predict.py
│
├── dashboard/
│   └── app.py
│
├── app.py
│
├── .github/
│   └── workflows/
│       ├── hourly_feature_pipeline.yml
│       └── train_daily.yml
│
├── data/
│   └── features_backfill.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── ridge.pkl
│   ├── xgboost.pkl
│   ├── metrics.csv
│   └── shap_importance.csv
│
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jiyasohail/aqi-monitoring-system.git
cd aqi-monitoring-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
OPENWEATHER_API_KEY=your_api_key
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT=your_project_name
```

### 4. Backfill Historical Data

```bash
python -m pipelines.backfill --days 365
```

### 5. Train Models

```bash
python -m pipelines.train_pipeline
```

### 6. Launch Dashboard

```bash
python app.py
```

---

## 🔐 GitHub Actions Secrets

The following repository secrets are required for automated execution:

| Secret              | Description            |
| ------------------- | ---------------------- |
| OPENWEATHER_API_KEY | OpenWeather API Key    |
| HOPSWORKS_API_KEY   | Hopsworks API Key      |
| HOPSWORKS_PROJECT   | Hopsworks Project Name |

---

## 📊 Dashboard Highlights

The dashboard provides:

* Current AQI monitoring
* 72-hour AQI forecasting
* Pollutant concentration breakdown
* Model performance insights
* Interactive visual analytics
* Real-time environmental monitoring

---

## 🎯 Future Improvements

* Deep Learning forecasting models (LSTM / GRU)
* Multi-city AQI monitoring
* Weather-integrated forecasting
* Alert and notification system
* Docker & Kubernetes deployment
* Model drift monitoring

---

## 👩‍💻 Author

**Javariya Sohail**

Computer Science Student | Data Analytics Enthusiast | MLOps Practitioner

Developed during the **10Pearls Internship Program (2026)** as a real-world machine learning and MLOps project focused on environmental intelligence and predictive analytics.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
