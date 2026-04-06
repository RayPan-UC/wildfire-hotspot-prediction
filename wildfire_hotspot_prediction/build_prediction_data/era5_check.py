"""
build_prediction_data/era5_check.py
-------------------------------------
Ensure ERA5 weather data covers the full study window + buffer.

Called once at pipeline startup (not per-request). Downloads the missing
time range via CDS API if the processed era5.parquet does not extend far
enough. Reuses the ERA5 download + preprocess logic from collect/environment.py
and preprocess/environment.py.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from wildfire_hotspot_prediction.study import Study


def ensure_era5_coverage(study: Study, extra_days: int = 3) -> None:
    """Ensure era5.parquet covers study.start_date → study.end_date + extra_days.

    If the processed parquet already covers the required window, returns
    immediately. Otherwise downloads the missing period via CDS API and
    appends it to the parquet.

    Args:
        study:      Study instance (provides bbox, dates, weather_dir).
        extra_days: Buffer days beyond study.end_date. Defaults to 3.
    """
    era5_path  = study.weather_dir / "era5.parquet"
    need_until = pd.Timestamp(study.end_date) + timedelta(days=extra_days)

    if era5_path.exists():
        era5 = pd.read_parquet(era5_path, columns=["valid_time"])
        era5["valid_time"] = pd.to_datetime(era5["valid_time"])
        if era5["valid_time"].max() >= need_until:
            return

    print(f"[era5_check] ERA5 coverage insufficient — downloading to {need_until.date()} ...")
    _download_and_append(study, need_until)
    print("[era5_check] ERA5 download complete")


# ── Internal ──────────────────────────────────────────────────────────────────

def _download_and_append(study: Study, need_until: pd.Timestamp) -> None:
    """Download ERA5 for the missing window and merge into era5.parquet."""
    from wildfire_hotspot_prediction.collect.environment import download_era5
    from wildfire_hotspot_prediction.preprocess.environment import _preprocess_era5

    era5_path = study.weather_dir / "era5.parquet"

    # Determine download start: day after current max, or study start
    if era5_path.exists():
        existing = pd.read_parquet(era5_path, columns=["valid_time"])
        existing["valid_time"] = pd.to_datetime(existing["valid_time"])
        download_start = existing["valid_time"].max().normalize() + timedelta(days=1)
    else:
        download_start = pd.Timestamp(study.start_date)

    # Download raw ERA5 .nc for missing window into a temp study
    # (reuse collect logic which writes to study.weather_raw_dir / era5.nc)
    download_era5(
        study,
        start_date=download_start.strftime("%Y-%m-%d"),
        end_date=need_until.strftime("%Y-%m-%d"),
    )

    # Preprocess the newly downloaded .nc → parquet (overwrites era5.parquet)
    # If existing data is present, merge old + new
    if era5_path.exists():
        old_df = pd.read_parquet(era5_path)
        _preprocess_era5(study)               # overwrites era5_path with new window
        new_df = pd.read_parquet(era5_path)
        merged = (
            pd.concat([old_df, new_df])
            .drop_duplicates(subset=["valid_time", "grid_id"])
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        merged.to_parquet(era5_path, index=False)
    else:
        _preprocess_era5(study)
