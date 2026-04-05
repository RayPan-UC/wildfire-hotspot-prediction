"""
collect/hotspots.py
-------------------
Download VIIRS active fire hotspot data from NASA FIRMS.

Output:
    <project>/data_raw/firms/hotspots_raw.csv

Credentials:
    Set FIRMS_API_KEY environment variable, or pass api_key explicitly.
    Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from wildfire_hotspot_prediction.study import Study

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def collect_hotspots(study: Study, api_key: str = None) -> Path:
    """Download VIIRS active fire hotspots from NASA FIRMS.

    Queries the FIRMS API for the study bounding box and date range,
    selecting the best available dataset (SP preferred over NRT).
    Skips download if the output file already exists.

    Args:
        study:   Study instance defining the AOI and time range.
        api_key: FIRMS MAP_KEY. Defaults to the FIRMS_API_KEY environment
                 variable. Get a free key at:
                 https://firms.modaps.eosdis.nasa.gov/api/map_key/

    Returns:
        Path to the downloaded raw CSV file.
    """
    out_path = study.firms_raw_dir / "hotspots_raw.csv"
    if out_path.exists():
        print(f"[firms] already exists, skipping → {out_path}")
        return out_path

    key = api_key or os.environ.get("FIRMS_API_KEY")
    if not key:
        raise EnvironmentError(
            "FIRMS_API_KEY not set. Pass api_key= or set the environment variable. "
            "Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    lon_min, lat_min, lon_max, lat_max = study.bbox
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    start_day = date.fromisoformat(study.start_date)
    end_day   = date.fromisoformat(study.end_date)

    source = _pick_firms_dataset(key, study.start_date, study.end_date)
    total  = (end_day - start_day).days + 1
    print(f"[firms] {study.start_date} → {study.end_date} ({total} days)  source={source}")
    print(f"[firms] bbox → {bbox_str}")

    frames  = []
    current = start_day
    while current <= end_day:
        df = _fetch_day(key, source, bbox_str, current)
        if not df.empty:
            frames.append(df)
        current += timedelta(days=1)

    study.firms_raw_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        result = pd.concat(frames).drop_duplicates()
        result.to_csv(out_path, index=False)
        print(f"[firms] {len(result):,} records saved → {out_path}")
    else:
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"[firms] 0 records saved → {out_path}")

    return out_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_day(api_key: str, source: str, bbox_str: str, day: date) -> pd.DataFrame:
    url = f"{FIRMS_BASE}/{api_key}/{source}/{bbox_str}/1/{day.strftime('%Y-%m-%d')}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"[firms] warning: failed to fetch {day}: {e}")
        return pd.DataFrame()


def _pick_firms_dataset(api_key: str, time_start: str, time_end: str) -> str:
    """Query FIRMS data_availability and return the best dataset for the period.

    Priority (SP preferred over NRT, VIIRS preferred over MODIS):
      VIIRS_NOAA21_SP → VIIRS_NOAA20_SP → VIIRS_SNPP_SP → MODIS_SP
      → VIIRS_NOAA21_NRT → VIIRS_NOAA20_NRT → VIIRS_SNPP_NRT → MODIS_NRT
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{api_key}/all"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[firms] warning: could not query data_availability ({e}), defaulting to VIIRS_SNPP_NRT")
        return "VIIRS_SNPP_NRT"

    df["min_date"] = pd.to_datetime(df["min_date"]).dt.tz_localize(None)
    df["max_date"] = pd.to_datetime(df["max_date"]).dt.tz_localize(None)
    t_start = pd.Timestamp(time_start).tz_localize(None)
    t_end   = pd.Timestamp(time_end).tz_localize(None)

    available = df[(df["min_date"] <= t_start) & (df["max_date"] >= t_end)]
    if available.empty:
        raise ValueError(f"No FIRMS dataset covers {time_start} to {time_end}")

    priority = [
        "VIIRS_NOAA21_SP", "VIIRS_NOAA20_SP", "VIIRS_SNPP_SP", "MODIS_SP",
        "VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT",
    ]
    ids = set(available["data_id"])
    for candidate in priority:
        if candidate in ids:
            return candidate
    return available.iloc[0]["data_id"]
