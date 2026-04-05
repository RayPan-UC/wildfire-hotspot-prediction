"""
preprocess/grid.py
------------------
Define the study's 500 m spatial grid and align all static
environmental layers onto it.

Reads:
    <project>/data_processed/terrain/dtm.tif
    <project>/data_processed/terrain/slope.tif
    <project>/data_processed/terrain/aspect.tif
    <project>/data_processed/landcover/fuel_type.tif

Produces:
    <project>/data_processed/grid_static.parquet
        cols: grid_id, x_proj, y_proj, dtm, slope, aspect, fuel_type
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.utils.geo import snap_grid_ids
from wildfire_hotspot_prediction.utils.raster import RasterSampler

_GRID_RES = 500.0   # metres
_PROJ_CRS = "EPSG:3978"


def build_grid(study: Study) -> Path:
    """Define study grid cells and align static features to the 500 m grid.

    Projects the study bbox to EPSG:3978, creates a regular 500 m grid,
    then samples terrain (DTM/slope/aspect) and fuel type at each cell centre.

    Args:
        study: Study instance.

    Returns:
        Path to data_processed/grid_static.parquet.
    """
    out_path = study.project_dir / "data_processed" / "grid_static.parquet"
    if out_path.exists():
        print(f"[preprocess] grid_static already exists, skipping → {out_path}")
        return out_path

    # ── Project bbox to EPSG:3978 ─────────────────────────────────────────────
    tr = Transformer.from_crs("EPSG:4326", _PROJ_CRS, always_xy=True)
    lon_min, lat_min, lon_max, lat_max = study.bbox
    x_min, y_min = tr.transform(lon_min, lat_min)
    x_max, y_max = tr.transform(lon_max, lat_max)

    # Snap bbox to grid
    x0 = np.floor(x_min / _GRID_RES) * _GRID_RES
    y0 = np.floor(y_min / _GRID_RES) * _GRID_RES
    x1 = np.ceil(x_max  / _GRID_RES) * _GRID_RES
    y1 = np.ceil(y_max  / _GRID_RES) * _GRID_RES

    xs = np.arange(x0, x1 + _GRID_RES, _GRID_RES)
    ys = np.arange(y0, y1 + _GRID_RES, _GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    x_flat = xx.ravel().astype(np.float32)
    y_flat = yy.ravel().astype(np.float32)
    xy     = np.column_stack([x_flat, y_flat])

    grid_ids = snap_grid_ids(xy, _GRID_RES)

    df = pd.DataFrame({
        "grid_id": grid_ids,
        "x_proj":  x_flat,
        "y_proj":  y_flat,
    })

    # ── Sample static raster layers ───────────────────────────────────────────
    for name in ("dtm", "slope", "aspect"):
        tif = study.terrain_dir / f"{name}.tif"
        if tif.exists():
            df[name] = RasterSampler(tif).sample(xy)
        else:
            print(f"[preprocess] grid — {name}.tif not found, column will be NaN")
            df[name] = np.nan

    fuel_tif = study.landcover_dir / "fuel_type.tif"
    if fuel_tif.exists():
        df["fuel_type"] = RasterSampler(fuel_tif).sample(xy).astype(np.int16)
    else:
        print("[preprocess] grid — fuel_type.tif not found, column will be NaN")
        df["fuel_type"] = pd.NA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[preprocess] grid → {len(df):,} cells ({len(xs)}×{len(ys)}) → {out_path}")
    return out_path
