import os
import sys
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

def run():
    OW_API_KEY        = os.environ.get("OPENWEATHER_API_KEY")
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

    print(f"DEBUG: OW_API_KEY = {'SET ✅' if OW_API_KEY else 'NOT SET ❌'}")

    if not OW_API_KEY:
        print("❌  OPENWEATHER_API_KEY is not set.")
        sys.exit(1)

    from pipelines.fetcher  import fetch_current, fetch_weather_current
    from pipelines.features import compute_aqi, add_time_features, add_change_rate
    import pandas as pd

    print("Fetching current air quality...")
    df = fetch_current(OW_API_KEY)
    weather = fetch_weather_current(OW_API_KEY)

    for k, v in weather.items():
        df[k] = v

    df["aqi"] = compute_aqi(df)
    df = add_time_features(df)
    df = add_change_rate(df, "aqi")
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs("data", exist_ok=True)
    out = "data/features_live.csv"
    header = not os.path.exists(out)
    df.to_csv(out, mode="a", index=False, header=header)
    print(f"✅  Saved {len(df)} row(s) to {out}")

if __name__ == "__main__":
    run()