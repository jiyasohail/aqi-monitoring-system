import os
import sys
from datetime import datetime, timezone

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def run():
    # ── Environment Variables ─────────────────────────────
    OW_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
    HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_karachi")

    print(f"DEBUG: OW_API_KEY = {'SET ✅' if OW_API_KEY else 'NOT SET ❌'}")
    print(f"DEBUG: HOPSWORKS_API_KEY = {'SET ✅' if HOPSWORKS_API_KEY else 'NOT SET ❌'}")

    if not OW_API_KEY:
        print("❌ OPENWEATHER_API_KEY is not set.")
        sys.exit(1)

    # ── Imports ───────────────────────────────────────────
    from pipelines.fetcher import fetch_current, fetch_weather_current
    from pipelines.features import compute_aqi, add_time_features, add_change_rate
    import pandas as pd

    # ── Fetch Data ────────────────────────────────────────
    print("🌍 Fetching current air quality...")
    df = fetch_current(OW_API_KEY)

    print("🌤 Fetching weather data...")
    weather = fetch_weather_current(OW_API_KEY)

    # merge weather into dataframe
    for k, v in weather.items():
        df[k] = v

    # ── Feature Engineering ───────────────────────────────
    df["aqi"] = compute_aqi(df)
    df = add_time_features(df)
    df = add_change_rate(df, "aqi")

    df["ingested_at"] = datetime.now(timezone.utc).isoformat()

    # ── Safety Check ──────────────────────────────────────
    if "time" not in df.columns:
        print("❌ Missing required column: 'time'")
        sys.exit(1)

    # ── Save Local Backup ────────────────────────────────
    os.makedirs("data", exist_ok=True)
    out = "data/features_live.csv"

    header = not os.path.exists(out)
    df.to_csv(out, mode="a", index=False, header=header)

    print(f"💾 Saved {len(df)} row(s) locally to {out}")

    # ── Upload to Hopsworks Feature Store ────────────────
    if HOPSWORKS_API_KEY:
        print("📤 Uploading to Hopsworks Feature Store...")

        import hopsworks

        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )

        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            description="AQI monitoring features for Karachi",
            primary_key=["time"],
            event_time="time",
            online_enabled=True
        )

    df["pm10"] = df["pm10"].astype(float)
    df["no"]   = df["no"].astype(float)
    df["pm2_5"] = df["pm2_5"].astype(float)
    fg.insert(df, write_options={"wait_for_job": True})

    print("✅ Upload complete: Feature Group updated in Hopsworks")

    print("🚀 Feature pipeline finished successfully.")


if __name__ == "__main__":
    run()