import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
from pipelines.fetcher import fetch_historical
from pipelines.features import build_features

OW_API_KEY        = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_karachi")
CHUNK_DAYS = 25

def run(days=365, start_date=None):
    print(f"API KEY = {'FOUND ✅' if OW_API_KEY else 'MISSING ❌'}")
    if not OW_API_KEY:
        print("Add OPENWEATHER_API_KEY to your .env file and try again.")
        sys.exit(1)

    end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = start_date or (end_dt - timedelta(days=days))
    print(f"Fetching {start_dt.date()} → {end_dt.date()}")

    chunks, cursor = [], start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end_dt)
        print(f"  {cursor.date()} → {chunk_end.date()} …")
        df_chunk = fetch_historical(OW_API_KEY, cursor, chunk_end)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        cursor = chunk_end

    if not chunks:
        print("❌ No data retrieved.")
        return

    df_raw  = pd.concat(chunks, ignore_index=True).drop_duplicates(subset="time").sort_values("time")
    df_feat = build_features(df_raw)
    os.makedirs("data", exist_ok=True)
    df_feat.to_csv("data/features_backfill.csv", index=False)
    print(f"✅ Done! Saved {len(df_feat)} rows to data/features_backfill.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=365)
    parser.add_argument("--start", type=str, default=None)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    run(days=args.days, start_date=start)