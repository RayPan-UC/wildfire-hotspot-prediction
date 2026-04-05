"""
preprocess/clouds.py
--------------------
Match downloaded cloud mask granules to T2 time steps
and store per-timestep cloudy pixel arrays.

Reads:
    <project>/data_raw/clouds/CLDMSK_*.npy

Produces:
    <project>/data_processed/clouds/<YYYY-MM-DDTHHMM>.parquet
        cols: x_proj [float32], y_proj [float32]  (cloudy pixel centres in EPSG:3978)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData

_FNAME_RE    = re.compile(r"CLDMSK_(\d{4})(\d{3})_(\d{2})(\d{2})\.npy")
_MAX_GAP_MIN = 10   # max minutes between T2 and granule to be considered a match


def preprocess_clouds(
    study:        Study,
    hotspot_data: HotspotData,
) -> Path:
    """Match cached cloud mask granules to T2 time steps and save per-step parquets.

    For each T2 time step, finds the nearest granule .npy file within
    _MAX_GAP_MIN minutes, then writes its cloudy pixel coordinates as parquet.
    Cloud masks were already clipped to the study AOI during collection.

    Args:
        study:        Study instance.
        hotspot_data: Output of preprocess_hotspots; provides T2 timestamps.

    Returns:
        Path to the processed clouds directory.
    """
    study.clouds_dir.mkdir(parents=True, exist_ok=True)

    # ── Build granule-time → npy-path index ──────────────────────────────────
    granule_map: dict[pd.Timestamp, Path] = {}
    for p in study.clouds_raw_dir.glob("CLDMSK_*.npy"):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        year, doy, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm)
        granule_map[pd.Timestamp(dt)] = p

    if not granule_map:
        print("[preprocess] clouds — no .npy files found, skipping")
        return study.clouds_dir

    granule_ts     = sorted(granule_map)
    max_gap        = pd.Timedelta(minutes=_MAX_GAP_MIN)
    saved, skipped = 0, 0

    for t2 in hotspot_data.overpass_times:
        out_path = study.clouds_dir / (t2.strftime("%Y-%m-%dT%H%M") + ".parquet")
        if out_path.exists():
            continue

        closest = min(granule_ts, key=lambda t: abs(t - t2))
        if abs(closest - t2) > max_gap:
            skipped += 1
            continue

        xy = np.load(granule_map[closest])
        if len(xy) == 0:
            skipped += 1
            continue

        pd.DataFrame(xy, columns=["x_proj", "y_proj"]).to_parquet(out_path, index=False)
        saved += 1

    print(f"[preprocess] clouds → {saved} parquets saved, {skipped} time steps skipped")
    return study.clouds_dir
