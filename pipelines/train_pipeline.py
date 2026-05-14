"""
train_pipeline.py — Daily training pipeline
Loads features from Hopsworks (or local CSV), trains multiple ML models,
evaluates them, and saves the best to Hopsworks Model Registry.

Models trained:
  - Random Forest (sklearn)
  - Ridge Regression (sklearn)
  - XGBoost
  - TensorFlow MLP (simple deep learning model)

Run: python -m pipelines.train_pipeline
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble       import RandomForestRegressor
from sklearn.linear_model   import Ridge
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline       import Pipeline

warnings.filterwarnings("ignore")

OW_API_KEY        = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_karachi")
FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_DIR = "models"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load features from Hopsworks or fallback to local CSV."""
    if HOPSWORKS_API_KEY:
        return _load_from_hopsworks()
    csv = "data/features_backfill.csv"
    if os.path.exists(csv):
        print(f"📂  Loading features from {csv}")
        df = pd.read_csv(csv, parse_dates=["time"])
        return df
    raise FileNotFoundError("No feature data found. Run backfill.py first.")


def _load_from_hopsworks() -> pd.DataFrame:
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df  = fg.read()
    print(f"✅  Loaded {len(df)} rows from Hopsworks.")
    return df


def get_xy(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target y."""
    target   = "aqi_next_24h"
    drop_cols = {"time", target, "ow_aqi_scale", "ingested_at"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[target].fillna(0)
    return X, y, list(X.columns)


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Chronological train/test split."""
    df = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<25}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ── Model builders ─────────────────────────────────────────────────────────────

def build_rf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ])


def build_ridge() -> Pipeline:
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
        print("⚠️   XGBoost not installed, skipping.")
        return None


def build_tf_mlp(input_dim: int):
    """Simple TensorFlow MLP for AQI regression."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
        return model
    except ImportError:
        print("⚠️   TensorFlow not installed, skipping deep learning model.")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print(f"[{datetime.now(timezone.utc).isoformat()}] ── Training Pipeline Start ──")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_features()
    train_df, test_df = time_split(df)
    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    X_train, y_train, feat_cols = get_xy(train_df)
    X_test,  y_test,  _         = get_xy(test_df)

    results = []

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf = build_rf()
    rf.fit(X_train, y_train)
    results.append(evaluate("RandomForest", y_test, rf.predict(X_test)))
    _save_sklearn(rf, "random_forest", feat_cols)

    # ── Ridge Regression ───────────────────────────────────────────────────────
    ridge = build_ridge()
    ridge.fit(X_train, y_train)
    results.append(evaluate("Ridge", y_test, ridge.predict(X_test)))
    _save_sklearn(ridge, "ridge", feat_cols)

    # ── XGBoost ────────────────────────────────────────────────────────────────
    xgb = build_xgb()
    if xgb is not None:
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", y_test, xgb.predict(X_test)))
        _save_sklearn(xgb, "xgboost", feat_cols)

    # ── TensorFlow MLP ─────────────────────────────────────────────────────────
    tf_model = build_tf_mlp(X_train.shape[1])
    if tf_model is not None:
        from sklearn.preprocessing import StandardScaler as SS
        sc = SS()
        Xtr_sc = sc.fit_transform(X_train)
        Xte_sc = sc.transform(X_test)
        tf_model.fit(Xtr_sc, y_train, epochs=30, batch_size=64, validation_split=0.1, verbose=0)
        tf_preds = tf_model.predict(Xte_sc, verbose=0).flatten()
        results.append(evaluate("TF_MLP", y_test, tf_preds))
        tf_model.save(f"{MODEL_DIR}/tf_mlp.keras")
        _save_pickle(sc, "tf_scaler")
        with open(f"{MODEL_DIR}/tf_feature_cols.json", "w") as f:
            json.dump(feat_cols, f)

    # ── SHAP feature importance (on best sklearn model) ────────────────────────
    _compute_shap(rf, X_train, feat_cols)

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(results).sort_values("rmse")
    metrics_df.to_csv(f"{MODEL_DIR}/metrics.csv", index=False)
    print(f"\n🏆  Best model: {metrics_df.iloc[0]['model']}  (RMSE={metrics_df.iloc[0]['rmse']:.2f})")

    # ── Push best sklearn model to Hopsworks ───────────────────────────────────
    if HOPSWORKS_API_KEY:
        best = metrics_df.iloc[0]["model"].lower().replace(" ", "_")
        _push_to_hopsworks(best, metrics_df.iloc[0].to_dict())

    print(f"[{datetime.now(timezone.utc).isoformat()}] ── Training Pipeline Done ──")


def _save_sklearn(model, name: str, feat_cols: list):
    _save_pickle(model, name)
    with open(f"{MODEL_DIR}/{name}_feature_cols.json", "w") as f:
        json.dump(feat_cols, f)


def _save_pickle(obj, name: str):
    path = f"{MODEL_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  💾  Saved {path}")


def _compute_shap(model, X_train: pd.DataFrame, feat_cols: list):
    try:
        import shap
        print("\n📊  Computing SHAP values …")
        # Use the inner estimator for SHAP (skip pipeline scaler)
        inner = model.named_steps.get("rf") or model.named_steps.get("xgb") or model.named_steps.get("ridge")
        X_tr_scaled = model.named_steps["scaler"].transform(X_train) if "scaler" in model.named_steps else X_train.values
        X_sample = X_tr_scaled[:500]  # sample for speed
        explainer = shap.TreeExplainer(inner) if hasattr(inner, "feature_importances_") else shap.LinearExplainer(inner, X_sample)
        shap_values = explainer.shap_values(X_sample)
        importance = pd.DataFrame({
            "feature":    feat_cols,
            "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False)
        importance.to_csv(f"{MODEL_DIR}/shap_importance.csv", index=False)
        print(f"  Top-5 features: {importance['feature'].head(5).tolist()}")
    except ImportError:
        print("⚠️   SHAP not installed, skipping feature importance.")
    except Exception as e:
        print(f"⚠️   SHAP failed: {e}")


def _push_to_hopsworks(model_name: str, metrics: dict):
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    model_dir = MODEL_DIR
    hw_model = mr.sklearn.create_model(
        name=f"aqi_{model_name}",
        metrics=metrics,
        description=f"AQI 24h forecast model: {model_name}",
    )
    hw_model.save(model_dir)
    print(f"✅  Pushed model '{model_name}' to Hopsworks Model Registry.")


if __name__ == "__main__":
    run()
