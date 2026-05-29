"""
train_pipeline.py - Daily training pipeline
Loads features from Hopsworks (or local CSV), trains multiple ML models,
evaluates them, and saves the best to Hopsworks Model Registry.

Run: python -m pipelines.train_pipeline
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_karachi")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_DIR = "models"


def load_features():
    csv = "data/features_backfill.csv"
    if os.path.exists(csv):
        print(f"Loading features from {csv}")
        return pd.read_csv(csv, parse_dates=["time"])
    if HOPSWORKS_API_KEY:
        return _load_from_hopsworks()
    raise FileNotFoundError("No feature data found. Run backfill.py first.")


def _load_from_hopsworks():
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    print(f"Loaded {len(df)} rows from Hopsworks.")
    return df


def get_xy(df):
    target = "aqi_next_24h"
    drop_cols = {"time", target, "ow_aqi_scale", "ingested_at"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[target].fillna(0)
    return X, y, list(X.columns)


def time_split(df, test_ratio=0.2):
    df = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:<25}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def build_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ])


def build_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def build_xgb():
    try:
        from xgboost import XGBRegressor
        return Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0)),
        ])
    except ImportError:
        print("XGBoost not installed, skipping.")
        return None


def run():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Training Pipeline Start")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_features()
    train_df, test_df = time_split(df)
    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    X_train, y_train, feat_cols = get_xy(train_df)
    X_test, y_test, _ = get_xy(test_df)

    results = []

    rf = build_rf()
    rf.fit(X_train, y_train)
    results.append(evaluate("RandomForest", y_test, rf.predict(X_test)))
    _save_sklearn(rf, "random_forest", feat_cols)

    ridge = build_ridge()
    ridge.fit(X_train, y_train)
    results.append(evaluate("Ridge", y_test, ridge.predict(X_test)))
    _save_sklearn(ridge, "ridge", feat_cols)

    xgb = build_xgb()
    if xgb is not None:
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", y_test, xgb.predict(X_test)))
        _save_sklearn(xgb, "xgboost", feat_cols)

    _compute_shap(rf, X_train, feat_cols)

    metrics_df = pd.DataFrame(results).sort_values("rmse")
    metrics_df.to_csv(f"{MODEL_DIR}/metrics.csv", index=False)

    best_row = metrics_df.iloc[0]
    print(f"Best model: {best_row['model']}  (RMSE={best_row['rmse']:.2f})")

    if HOPSWORKS_API_KEY:
        model_name = best_row["model"].lower().replace(" ", "_")
        clean_metrics = {
            "rmse": float(best_row["rmse"]),
            "mae": float(best_row["mae"]),
            "r2": float(best_row["r2"]),
        }
        _push_to_hopsworks(model_name, clean_metrics)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Training Pipeline Done")


def _save_sklearn(model, name, feat_cols):
    _save_pickle(model, name)
    with open(f"{MODEL_DIR}/{name}_feature_cols.json", "w") as f:
        json.dump(feat_cols, f)


def _save_pickle(obj, name):
    path = f"{MODEL_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved {path}")


def _compute_shap(model, X_train, feat_cols):
    try:
        import shap
        print("Computing SHAP values...")
        inner = model.named_steps.get("rf") or model.named_steps.get("ridge")
        X_tr_scaled = model.named_steps["scaler"].transform(X_train)
        X_sample = X_tr_scaled[:500]
        if hasattr(inner, "feature_importances_"):
            explainer = shap.TreeExplainer(inner)
        else:
            explainer = shap.LinearExplainer(inner, X_sample)
        shap_values = explainer.shap_values(X_sample)
        importance = pd.DataFrame({
            "feature": feat_cols,
            "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False)
        importance.to_csv(f"{MODEL_DIR}/shap_importance.csv", index=False)
        print(f"  Top-5 features: {importance['feature'].head(5).tolist()}")
    except ImportError:
        print("SHAP not installed, skipping.")
    except Exception as e:
        print(f"SHAP failed: {e}")


def _push_to_hopsworks(model_name, metrics):
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    hw_model = mr.sklearn.create_model(
        name=f"aqi_{model_name}",
        metrics=metrics,
        description=f"AQI 24h forecast model: {model_name}",
    )
    hw_model.save(MODEL_DIR)
    print(f"Pushed model '{model_name}' to Hopsworks Model Registry.")


if __name__ == "__main__":
    run()