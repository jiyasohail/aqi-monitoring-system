"""
fetcher.py — OpenWeather API helpers for AQI Karachi project
Fetches current, historical, and forecast air quality + weather data.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

# ── API endpoints ──────────────────────────────────────────────────────────────
OW_BASE        = "https://api.openweathermap.org/data/2.5"
OW_POLLUTION   = f"{OW_BASE}/air_pollution"
OW_HIST        = f"{OW_BASE}/air_pollution/history"
OW_FORECAST    = f"{OW_BASE}/air_pollution/forecast"
OW_WEATHER     = f"{OW_BASE}/weather"

# Karachi coords
LAT = 24.8607
LON = 67.0011


def _ts_to_utc(ts: int) -> datetime:
    """Convert Unix epoch (int) → timezone-aware UTC datetime."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def _parse_pollution_list(items: list) -> pd.DataFrame:
    """Parse a list of OpenWeather air_pollution items into a DataFrame."""
    records = []
    for item in items:
        comps = item.get("components", {})
        records.append({
            "time":         _ts_to_utc(item["dt"]),
            "pm2_5":        comps.get("pm2_5"),
            "pm10":         comps.get("pm10"),
            "co":           comps.get("co"),
            "no":           comps.get("no", 0.0),
            "no2":          comps.get("no2"),
            "o3":           comps.get("o3"),
            "so2":          comps.get("so2"),
            "nh3":          comps.get("nh3", 0.0),
            "ow_aqi_scale": item.get("main", {}).get("aqi"),   # 1-5 OW scale
        })
    return pd.DataFrame(records)


def fetch_current(api_key: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    """Return single-row DataFrame with current air quality readings."""
    resp = requests.get(OW_POLLUTION, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("list", [])
    if not items:
        return pd.DataFrame()
    return _parse_pollution_list(items[:1])


def fetch_historical(
    api_key: str,
    start_date: datetime,
    end_date: datetime,
    lat: float = LAT,
    lon: float = LON,
) -> pd.DataFrame:
    """Return hourly historical air quality DataFrame between start_date and end_date."""
    start_ts = int(start_date.timestamp())
    end_ts   = int(end_date.timestamp())
    resp = requests.get(
        OW_HIST,
        params={"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("list", [])
    if not items:
        print("⚠️  No historical records returned.")
        return pd.DataFrame()
    print(f"✅  Fetched {len(items)} historical records.")
    return _parse_pollution_list(items)


def fetch_forecast(api_key: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    """Return ~96-hour air quality forecast DataFrame."""
    resp = requests.get(OW_FORECAST, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("list", [])
    if not items:
        return pd.DataFrame()
    print(f"✅  Fetched {len(items)} forecast records.")
    return _parse_pollution_list(items)


def fetch_weather_current(api_key: str, lat: float = LAT, lon: float = LON) -> dict:
    """Return dict with current weather metrics (temperature, humidity, wind, pressure)."""
    resp = requests.get(
        OW_WEATHER,
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()
    return {
        "temp":       d["main"]["temp"],
        "humidity":   d["main"]["humidity"],
        "pressure":   d["main"]["pressure"],
        "wind_speed": d["wind"]["speed"],
        "wind_deg":   d["wind"].get("deg", 0),
        "visibility": d.get("visibility", 10000),
        "weather_main": d["weather"][0]["main"],
    }
