"""
features.py — Feature engineering for AQI forecasting
Computes time features, lag features, rolling stats, AQI target via EPA formula.
"""

import numpy as np
import pandas as pd

# ── EPA AQI breakpoints ────────────────────────────────────────────────────────
# Each entry: (C_low, C_high, I_low, I_high)
PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0,    54,    0,   50),
    (55,   154,  51,  100),
    (155,  254, 101,  150),
    (255,  354, 151,  200),
    (355,  424, 201,  300),
    (425,  504, 301,  400),
    (505,  604, 401,  500),
]


def _epa_aqi(concentration: float, breakpoints: list) -> float:
    """EPA linear interpolation between concentration breakpoints."""
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    return 500.0  # above highest breakpoint


def compute_aqi(df: pd.DataFrame) -> pd.Series:
    """
    Compute standard AQI from PM2.5 and PM10 columns.
    Returns the max sub-index as the overall AQI.
    """
    aqi_pm25 = df["pm2_5"].apply(lambda x: _epa_aqi(x, PM25_BREAKPOINTS) if pd.notna(x) else np.nan)
    aqi_pm10 = df["pm10"].apply(lambda x: _epa_aqi(x, PM10_BREAKPOINTS)  if pd.notna(x) else np.nan)
    return pd.concat([aqi_pm25, aqi_pm10], axis=1).max(axis=1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and categorical time features from 'time' column."""
    df = df.copy()
    t = df["time"]
    df["hour"]        = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["month"]       = t.dt.month
    df["day_of_year"] = t.dt.dayofyear
    # Cyclical encoding
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]       / 12)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    """Add lag features for given columns and lag steps (hours)."""
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list, windows: list) -> pd.DataFrame:
    """Add rolling mean, std, max for given columns and window sizes."""
    df = df.copy()
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}h_mean"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}h_std"]  = df[col].rolling(w, min_periods=1).std()
            df[f"{col}_roll{w}h_max"]  = df[col].rolling(w, min_periods=1).max()
    return df


def add_change_rate(df: pd.DataFrame, col: str = "aqi") -> pd.DataFrame:
    """Add 1-step change rate for a column."""
    df = df.copy()
    df[f"{col}_change_rate"] = df[col].diff()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: raw DataFrame with columns [time, pm2_5, pm10, co, no2, o3, so2, nh3, ...]
    Output: DataFrame with all engineered features + 'aqi' target column.
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().bfill()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Compute AQI target
    df["aqi"] = compute_aqi(df)

    # Time features
    df = add_time_features(df)

    # Pollutant lag features (1h, 3h, 6h, 12h, 24h, 48h, 72h)
    pollutant_cols = ["pm2_5", "pm10", "co", "no2", "o3", "so2", "aqi"]
    lag_hours      = [1, 3, 6, 12, 24, 48, 72]
    df = add_lag_features(df, pollutant_cols, lag_hours)

    # Rolling stats (6h, 12h, 24h, 48h)
    df = add_rolling_features(df, pollutant_cols, [6, 12, 24, 48])

    # AQI change rate
    df = add_change_rate(df, "aqi")

    # Target: next 24h AQI (shift -24)
    df["aqi_next_24h"] = df["aqi"].shift(-24)

    # Drop rows with NaN targets
    df = df.dropna(subset=["aqi_next_24h"]).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns (exclude 'time' and target columns)."""
    exclude = {"time", "aqi_next_24h", "ow_aqi_scale"}
    return [c for c in df.columns if c not in exclude]
