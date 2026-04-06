"""
build_prediction_data/era5_check.py
-------------------------------------
Ensure ERA5 weather data covers the full study window + buffer.

If coverage is insufficient, re-downloads ERA5 for the full study window
(study.start_date → study.end_date + extra_days) and regenerates era5.parquet.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from wildfire_hotspot_prediction.study import Study


def ensure_era5_coverage(study: Study, extra_days: int = 3) -> None:
    """Ensure era5.parquet covers study.start_date → study.end_date + extra_days.

    If coverage is insufficient, deletes the existing raw ERA5 file and
    re-downloads for the full window (including the buffer), then regenerates
    the processed parquet.

    Args:
        study:      Study instance.
        extra_days: Buffer days beyond study.end_date. Defaults to 3.
    """
    from wildfire_hotspot_prediction.collect.environment import collect_environment
    from wildfire_hotspot_prediction.preprocess.environment import preprocess_environment

    era5_path  = study.weather_dir / "era5.parquet"
    need_until = pd.Timestamp(study.end_date) + timedelta(days=extra_days)

    if era5_path.exists():
        era5 = pd.read_parquet(era5_path, columns=["valid_time"])
        era5["valid_time"] = pd.to_datetime(era5["valid_time"])
        if era5["valid_time"].max() >= need_until:
            print(f"[era5_check] ERA5 coverage OK (through {era5['valid_time'].max().date()})")
            return

    print(f"[era5_check] ERA5 coverage insufficient — re-downloading {study.start_date} → {need_until.date()} ...")

    # Extend study end_date to include buffer, delete cached raw files to force re-download
    extended_study = Study(
        name        = study.name,
        bbox        = study.bbox,
        start_date  = study.start_date,
        end_date    = need_until.strftime("%Y-%m-%d"),
        project_dir = study.project_dir,
    )

    # Remove existing raw + processed ERA5 files to force full re-download + re-preprocess
    raw_nc = study.weather_raw_dir / "era5.nc"
    if raw_nc.exists():
        raw_nc.unlink()
    for part in study.weather_raw_dir.glob("era5_*.nc"):
        part.unlink()
    processed = study.weather_dir / "era5.parquet"
    if processed.exists():
        processed.unlink()

    collect_environment(extended_study, sources=["era5"])
    preprocess_environment(extended_study, sources=["era5"])

    print(f"[era5_check] ERA5 re-download complete (through {need_until.date()})")
