"""
preprocess/environment.py
-------------------------
Standardise and reproject raw environmental datasets.

  - ERA5-Land: parse .nc, compute wind speed/direction, convert units,
               build (time × grid) numpy arrays, save as .parquet
  - Terrain:   clip and reproject DEM/slope/aspect rasters to study CRS
  - Landcover: clip and reproject fuel type raster to study CRS

Reads:
    <project>/data_raw/weather/era5.nc
    <project>/data_raw/terrain/dtm.tif, slope.tif, aspect.tif
    <project>/data_raw/landcover/fuel_type.tif

Produces:
    <project>/data_processed/weather/era5.parquet
    <project>/data_processed/terrain/dtm.tif, slope.tif, aspect.tif
    <project>/data_processed/landcover/fuel_type.tif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import rasterio.warp

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.utils.geo import snap_grid_ids

_GRID_RES = 500.0
_PROJ_CRS = "EPSG:3978"


def preprocess_environment(
    study:   Study,
    sources: list[str] = None,
) -> Path:
    """Standardise and reproject raw environmental datasets.

    Args:
        study:   Study instance.
        sources: Datasets to process. Any subset of ["era5", "terrain", "landcover"].
                 Defaults to all three.

    Returns:
        Path to the data_processed directory.
    """
    if sources is None:
        sources = ["era5", "terrain", "landcover"]

    if "era5" in sources:
        _preprocess_era5(study)
    if "terrain" in sources:
        _preprocess_terrain(study)
    if "landcover" in sources:
        _preprocess_landcover(study)

    return study.project_dir / "data_processed"


def _preprocess_era5(study: Study) -> Path:
    """Parse ERA5 .nc, compute wind speed/direction from u/v components,
    convert units, and save as columnar parquet for fast time-series lookup.

    Returns:
        Path to data_processed/weather/era5.parquet.
    """
    import xarray as xr

    out_path = study.weather_dir / "era5.parquet"
    if out_path.exists():
        print(f"[preprocess] era5 already exists, skipping → {out_path}")
        return out_path

    nc_path = study.weather_raw_dir / "era5.nc"
    print(f"[preprocess] era5 → reading {nc_path}")

    ds = xr.open_dataset(nc_path, engine="netcdf4")

    # ── Flatten to long-form DataFrame ───────────────────────────────────────
    df = ds.to_dataframe().reset_index()
    ds.close()

    # Rename standard ERA5-Land variable names
    rename = {
        "u10":  "u10",   # 10m u-component of wind [m/s]
        "v10":  "v10",   # 10m v-component of wind [m/s]
        "t2m":  "t2m",   # 2m temperature [K]
        "d2m":  "d2m",   # 2m dewpoint temperature [K]
        "tp":   "tp",    # total precipitation [m]
        "valid_time": "valid_time",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Unit conversions ──────────────────────────────────────────────────────
    if "t2m" in df.columns:
        df["temp_c"] = df["t2m"] - 273.15           # K → °C
    if "d2m" in df.columns:
        df["dewpoint_c"] = df["d2m"] - 273.15       # K → °C
    if "tp" in df.columns:
        df["precip_mm"] = df["tp"] * 1000           # m → mm

    # ── Wind speed and direction ──────────────────────────────────────────────
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)       # m/s
        df["wind_dir"]   = (270 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360

    # ── Relative humidity from dewpoint ──────────────────────────────────────
    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        df["rh"] = 100 * np.exp(
            17.625 * df["dewpoint_c"] / (243.04 + df["dewpoint_c"]) -
            17.625 * df["temp_c"]     / (243.04 + df["temp_c"])
        )
        df["rh"] = df["rh"].clip(0, 100)

    # ── Snap ERA5 lat/lon to 500 m grid_id ───────────────────────────────────
    # Project ERA5 coords to EPSG:3978, snap to nearest 500 m cell centre.
    # Multiple 500 m cells will share the same ERA5 grid_id (nearest-neighbour).
    unique_pts = df[["latitude", "longitude"]].drop_duplicates()
    from pyproj import Transformer as _T
    _tr = _T.from_crs("EPSG:4326", _PROJ_CRS, always_xy=True)
    xs, ys = _tr.transform(
        unique_pts["longitude"].tolist(),
        unique_pts["latitude"].tolist(),
    )
    xy = np.column_stack([xs, ys])
    unique_pts = unique_pts.copy()
    unique_pts["grid_id"] = snap_grid_ids(xy, _GRID_RES)
    df = df.merge(unique_pts, on=["latitude", "longitude"], how="left")

    # ── Keep relevant columns ─────────────────────────────────────────────────
    keep = ["valid_time", "grid_id", "latitude", "longitude",
            "temp_c", "dewpoint_c", "rh", "precip_mm",
            "wind_speed", "wind_dir", "u10", "v10"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["valid_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    study.weather_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[preprocess] era5 → {len(df):,} rows saved → {out_path}")
    return out_path


def _preprocess_terrain(study: Study) -> Path:
    """Reproject DTM/slope/aspect rasters from EPSG:3979 to study CRS (EPSG:3978).

    Returns:
        Path to data_processed/terrain/.
    """
    study.terrain_dir.mkdir(parents=True, exist_ok=True)
    dst_crs = CRS.from_epsg(3978)

    for name in ("dtm", "slope", "aspect"):
        src_path = study.terrain_raw_dir / f"{name}.tif"
        dst_path = study.terrain_dir    / f"{name}.tif"

        if dst_path.exists():
            print(f"[preprocess] terrain/{name} already exists, skipping")
            continue
        if not src_path.exists():
            print(f"[preprocess] terrain/{name} not found, skipping")
            continue

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({"crs": dst_crs, "transform": transform,
                         "width": width, "height": height})

            with rasterio.open(dst_path, "w", **meta) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source      = rasterio.band(src, band),
                        destination = rasterio.band(dst, band),
                        src_crs     = src.crs,
                        dst_crs     = dst_crs,
                        resampling  = Resampling.bilinear,
                    )
        print(f"[preprocess] terrain/{name} → {dst_path}")

    return study.terrain_dir


def _preprocess_landcover(study: Study) -> Path:
    """Reproject fuel type raster to study CRS (EPSG:3978).

    Returns:
        Path to data_processed/landcover/.
    """
    study.landcover_dir.mkdir(parents=True, exist_ok=True)
    src_path = study.landcover_raw_dir / "fuel_type.tif"
    dst_path = study.landcover_dir     / "fuel_type.tif"
    dst_crs  = CRS.from_epsg(3978)

    if dst_path.exists():
        print(f"[preprocess] landcover already exists, skipping → {dst_path}")
        return study.landcover_dir
    if not src_path.exists():
        print(f"[preprocess] landcover not found → {src_path}")
        return study.landcover_dir

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": transform,
                     "width": width, "height": height})

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, band),
                    destination = rasterio.band(dst, band),
                    src_crs     = src.crs,
                    dst_crs     = dst_crs,
                    resampling  = Resampling.nearest,  # categorical data
                )

    print(f"[preprocess] landcover → {dst_path}")
    return study.landcover_dir
