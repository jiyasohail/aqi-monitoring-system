import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

# ── Load .env file FIRST before anything else ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import pandas as pd
from pipelines.fetcher  import fetch_historical
from pipelines.features import build_features

# ── Config — read AFTER load_dotenv ───────────────────────────────────────────
OW_API_KEY        = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("AQI_Karachi", "aqi_karachi")

FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1
CHUNK_DAYS = 25


def run(days: int = 365, start_date=None):
    # ── Debug: print what we found ─────────────────────────────────────────────
    print(f"DEBUG: OW_API_KEY = {'SET ✅' if OW_API_KEY else 'NOT SET ❌'}")
    print(f"DEBUG: HOPSWORKS  = {'SET ✅' if HOPSWORKS_API_KEY else 'not set (will use CSV)'}")

    if not OW_API_KEY:
        print("\n❌  OPENWEATHER_API_KEY is not set.")
        print("    Run this in your terminal first:")
        print('    $env:OPENWEATHER_API_KEY="your_key_here"')
        sys.exit(1)

    end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = start_date or (end_dt - timedelta(days=days))
    print(f"Backfill: {start_dt.date()} → {end_dt.date()} (~{(end_dt - start_dt).days} days)")

    chunks = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end_dt)
        print(f"  Fetching {cursor.date()} → {chunk_end.date()} …")
        df_chunk = fetch_historical(OW_API_KEY, cursor, chunk_end)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        cursor = chunk_end

    if not chunks:
        print("❌  No data retrieved.")
        return

    df_raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset="time").sort_values("time")
    print(f"📊  Raw rows: {len(df_raw)}")

    df_feat = build_features(df_raw)
    print(f"✅  Feature rows: {len(df_feat)}  |  columns: {len(df_feat.columns)}")

    os.makedirs("data", exist_ok=True)
    out_csv = "data/features_backfill.csv"
    df_feat.to_csv(out_csv, index=False)
    print(f"💾  Saved to {out_csv}")

    if HOPSWORKS_API_KEY:
        _push_hopsworks(df_feat)
    else:
        print("ℹ️   Hopsworks not configured — data saved locally to CSV only.")


def _push_hopsworks(df: pd.DataFrame):
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["time"],
        event_time="time",
        description="Hourly AQI features for Karachi (backfill)",
    )
    df_hw = df.copy()
    if df_hw["time"].dt.tz is not None:
        df_hw["time"] = df_hw["time"].dt.tz_convert(None)
    fg.insert(df_hw, write_options={"wait_for_job": False})
    print(f"✅  Pushed {len(df_hw)} rows to Hopsworks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument("--days",  type=int, default=365)
    parser.add_argument("--start", type=str, default=None)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    run(days=args.days, start_date=start)