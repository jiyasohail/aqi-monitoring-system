import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
from pipelines.fetcher import fetch_historical
from pipelines.features import build_features

OW_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_karachi")

CHUNK_DAYS = 25


def run(days=365, start_date=None):
    print(f"API KEY = {'FOUND ✅' if OW_API_KEY else 'MISSING ❌'}")

    if not OW_API_KEY:
        print("❌ Add OPENWEATHER_API_KEY to your .env file and try again.")
        sys.exit(1)

    end_dt = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0
    )

    start_dt = start_date or (end_dt - timedelta(days=days))

    print(f"Fetching {start_dt.date()} → {end_dt.date()}")

    chunks = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end_dt)

        print(f"  {cursor.date()} → {chunk_end.date()} ...")

        df_chunk = fetch_historical(
            OW_API_KEY,
            cursor,
            chunk_end
        )

        if not df_chunk.empty:
            chunks.append(df_chunk)

        cursor = chunk_end

    if not chunks:
        print("❌ No data retrieved.")
        return

    df_raw = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset="time")
        .sort_values("time")
    )

    df_feat = build_features(df_raw)

    # Convert int64 columns to float for Hopsworks compatibility
    for col in df_feat.select_dtypes(include=["int64"]).columns:
        df_feat[col] = df_feat[col].astype(float)

    os.makedirs("data", exist_ok=True)

    output_file = "data/features_backfill.csv"
    df_feat.to_csv(output_file, index=False)

    print(f"✅ Done! Saved {len(df_feat)} rows to {output_file}")

    # Upload to Hopsworks
    if HOPSWORKS_API_KEY:
        _push_hopsworks(df_feat)
    else:
        print("ℹ️ Hopsworks not configured — saved locally only.")


def _push_hopsworks(df: pd.DataFrame):
    import hopsworks

    print("📤 Uploading to Hopsworks...")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["time"],
        event_time="time",
        description="Hourly AQI features for Karachi"
    )

    df_hw = df.copy()

    # Remove timezone if present
    if pd.api.types.is_datetime64tz_dtype(df_hw["time"]):
        df_hw["time"] = df_hw["time"].dt.tz_convert(None)

    # Convert int64 columns to float
    for col in df_hw.select_dtypes(include=["int64"]).columns:
        df_hw[col] = df_hw[col].astype(float)

    fg.insert(
        df_hw,
        write_options={"wait_for_job": True}
    )

    print(
        f"✅ Pushed {len(df_hw)} rows to "
        f"Hopsworks Feature Group 'aqi_features'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--days",
        type=int,
        default=365
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None
    )

    args = parser.parse_args()

    start = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        if args.start
        else None
    )

    run(
        days=args.days,
        start_date=start
    )