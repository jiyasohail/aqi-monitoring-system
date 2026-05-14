"""
predict.py — AQI prediction and 72-hour recursive forecasting
Loads a trained model and generates rolling 3-day forecasts.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pipelines.features import add_time_features, compute_aqi

MODEL_DIR = "models"

# AQI category thresholds (EPA)
AQI_CATEGORIES = [
    (0,   50,  "Good",              "#00E400"),
    (51,  100, "Moderate",          "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy",         "#FF0000"),
    (201, 300, "Very Unhealthy",    "#8F3F97"),
    (301, 500, "Hazardous",         "#7E0023"),
]


def aqi_category(aqi: float) -> dict:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"label": label, "color": color}
    return {"label": "Hazardous", "color": "#7E0023"}


def load_model(name: str = "random_forest"):
    """Load a pickled sklearn pipeline and its feature columns."""
    pkl = f"{MODEL_DIR}/{name}.pkl"
    feat_json = f"{MODEL_DIR}/{name}_feature_cols.json"
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Model not found: {pkl}. Run train_pipeline.py first.")
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    with open(feat_json) as f:
        feat_cols = json.load(f)
    return model, feat_cols


def prepare_input_row(df_history: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Given a history DataFrame, engineer features for the LAST row
    so we can make a one-step-ahead prediction.
    """
    from pipelines.features import (
        add_lag_features, add_rolling_features, add_change_rate,
        add_time_features
    )
    df = df_history.copy().sort_values("time").reset_index(drop=True)
    pollutant_cols = ["pm2_5", "pm10", "co", "no2", "o3", "so2", "aqi"]
    df = add_time_features(df)
    df = add_lag_features(df, pollutant_cols, [1, 3, 6, 12, 24, 48, 72])
    df = add_rolling_features(df, pollutant_cols, [6, 12, 24, 48])
    df = add_change_rate(df, "aqi")
    last = df.iloc[[-1]]
    # Align to expected columns, fill missing with 0
    X = last.reindex(columns=feat_cols, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    return X


def forecast_72h(
    df_history: pd.DataFrame,
    model_name: str = "random_forest",
    horizon: int = 72,
) -> pd.DataFrame:
    """
    Recursively forecast AQI for next `horizon` hours.
    Returns a DataFrame with columns [time, aqi_predicted, category, color].
    """
    model, feat_cols = load_model(model_name)
    df = df_history.copy()

    if "aqi" not in df.columns:
        df["aqi"] = compute_aqi(df)

    last_time = df["time"].max()
    predictions = []

    for step in range(1, horizon + 1):
        pred_time = last_time + timedelta(hours=step)
        X = prepare_input_row(df, feat_cols)
        aqi_pred = float(model.predict(X)[0])
        aqi_pred = max(0.0, aqi_pred)  # AQI can't be negative
        cat = aqi_category(aqi_pred)

        predictions.append({
            "time":          pred_time,
            "aqi_predicted": round(aqi_pred, 1),
            "category":      cat["label"],
            "color":         cat["color"],
        })

        # Append synthetic row for next iteration (carry last pollutants forward)
        new_row = df.iloc[[-1]].copy()
        new_row["time"] = pred_time
        new_row["aqi"]  = aqi_pred
        df = pd.concat([df, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def load_forecast_from_csv(path: str = "data/predictions.csv") -> pd.DataFrame | None:
    """Load pre-computed predictions from CSV (used in Streamlit when model isn't loaded)."""
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["time"])
    return None


def run_and_save(model_name: str = "random_forest", horizon: int = 72):
    """Generate forecast and save to CSV. Called by CI/CD predict workflow."""
    from pipelines.fetcher import fetch_historical, OW_API_KEY
    from pipelines.features import build_features

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set.")

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=4)  # need enough history for 72h lags

    df_raw  = fetch_historical(api_key, start, end)
    df_feat = build_features(df_raw)

    df_pred = forecast_72h(df_feat, model_name=model_name, horizon=horizon)
    os.makedirs("data", exist_ok=True)
    df_pred.to_csv("data/predictions.csv", index=False)
    print(f"✅  Saved {len(df_pred)} predictions to data/predictions.csv")
    return df_pred


if __name__ == "__main__":
    preds = run_and_save()
    print(preds.head())
