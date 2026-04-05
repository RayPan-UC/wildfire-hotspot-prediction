"""
training/features.py
--------------------
Feature extraction from precomputed parquets.

All features are derived via grid_id joins — no raster sampling at training time.
ERA5 (~9 km) is matched to 500 m receptor cells via nearest-neighbour lookup
using a cKDTree built on ERA5 grid coordinates.

Functions
---------
build_era5_index(era5)
    Build spatial index mapping any (x, y) to nearest ERA5 grid_id.

join_static(b_grid_ids, grid_static)
    dtm, slope, aspect, fuel_type at receptor B cells.

join_weather(b_grid_ids, t1, era5, era5_index)
    temp_c, rh, wind_speed, wind_dir at receptor B cells (nearest ERA5 hour).

join_fwi(b_grid_ids, t1, ffmc_daily, isi_hourly, ros_hourly, era5_index)
    ffmc, isi, ros at receptor B cells.

fire_geometry_features(t1, fire_state)
    Scalar fire boundary descriptors: fire_age_h, perimeter_m,
    compactness, growth_rate_km2h, frp_per_area_km2, new_area_km2.

dist_to_fire_front(b_xy, new_polys)
    Distance from each B cell to the nearest point on the fire front [m].

Note: path_features (A→B spread-path features) lives in sampling_path.py.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from wildfire_hotspot_prediction.training.fire_state import FireState


# ── Feature cache (precomputed once before the pair loop) ─────────────────────

@dataclass
class FeatureCache:
    """Precomputed indexes to avoid repeated set_index / unique scans per pair."""
    gs_idx:       pd.DataFrame           # grid_static indexed by grid_id
    gs_defaults:  dict                   # median fill values for out-of-grid cells
    era5_times:   list                   # sorted unique ERA5 valid_times
    era5_by_time: dict                   # Timestamp → era5 slice indexed by grid_id
    ffmc_dates:   list                   # sorted unique dates
    ffmc_by_date: dict                   # date → ffmc slice indexed by grid_id
    isi_times:    list                   # sorted unique ISI valid_times
    isi_by_time:  dict
    ros_times:    list
    ros_by_time:  dict


def build_feature_cache(
    grid_static: pd.DataFrame,
    era5:        pd.DataFrame,
    ffmc_daily:  pd.DataFrame,
    isi_hourly:  pd.DataFrame,
    ros_hourly:  pd.DataFrame,
) -> FeatureCache:
    """Precompute all indexes needed for feature extraction.

    Call once before the pair loop; pass the returned cache to join_* functions.
    """
    gs_idx = grid_static.set_index("grid_id")[["dtm", "slope", "aspect", "fuel_type"]]
    gs_defaults = {
        "dtm":       float(grid_static["dtm"].median()),
        "slope":     float(grid_static["slope"].median()),
        "aspect":    float(grid_static["aspect"].median()),
        "fuel_type": int(grid_static["fuel_type"].mode().iloc[0]),
    }

    era5_times   = sorted(era5["valid_time"].unique().tolist())
    era5_by_time = {
        t: era5[era5["valid_time"] == t].set_index("grid_id")[["temp_c", "rh", "wind_speed", "wind_dir"]]
        for t in era5_times
    }

    ffmc_dates   = sorted(ffmc_daily["date"].unique().tolist())
    ffmc_by_date = {
        d: ffmc_daily[ffmc_daily["date"] == d].set_index("grid_id")[["ffmc"]]
        for d in ffmc_dates
    }

    isi_times   = sorted(isi_hourly["valid_time"].unique().tolist())
    isi_by_time = {
        t: isi_hourly[isi_hourly["valid_time"] == t].set_index("grid_id")[["isi"]]
        for t in isi_times
    }

    ros_times   = sorted(ros_hourly["valid_time"].unique().tolist())
    ros_by_time = {
        t: ros_hourly[ros_hourly["valid_time"] == t].set_index("grid_id")[["ros"]]
        for t in ros_times
    }

    return FeatureCache(
        gs_idx=gs_idx,         gs_defaults=gs_defaults,
        era5_times=era5_times, era5_by_time=era5_by_time,
        ffmc_dates=ffmc_dates, ffmc_by_date=ffmc_by_date,
        isi_times=isi_times,   isi_by_time=isi_by_time,
        ros_times=ros_times,   ros_by_time=ros_by_time,
    )


def _nearest_in(t, sorted_times: list, by_time: dict) -> pd.DataFrame:
    """Return the pre-indexed slice closest to timestamp t (bisect, O(log n))."""
    if not sorted_times:
        return pd.DataFrame()
    idx = bisect.bisect_left(sorted_times, t)
    if idx == 0:
        best = sorted_times[0]
    elif idx >= len(sorted_times):
        best = sorted_times[-1]
    else:
        before, after = sorted_times[idx - 1], sorted_times[idx]
        best = before if abs(t - before) <= abs(t - after) else after
    return by_time[best]


# ── ERA5 spatial index ─────────────────────────────────────────────────────────

def build_era5_index(era5: pd.DataFrame) -> tuple[cKDTree, np.ndarray]:
    """Build a cKDTree on unique ERA5 grid coordinates.

    Args:
        era5: ERA5 parquet DataFrame with grid_id, latitude, longitude columns.

    Returns:
        Tuple of (tree, grid_ids_array):
            tree:          cKDTree built on ERA5 (x_proj, y_proj) decoded from grid_id.
            grid_ids_array: 1-D string array of ERA5 grid_ids (same order as tree).
    """
    unique = era5[["grid_id"]].drop_duplicates().copy()
    coords = np.array([list(map(float, g.split("_")))
                       for g in unique["grid_id"]], dtype=np.float64)
    tree = cKDTree(coords)
    return tree, unique["grid_id"].values


def _nearest_era5_grid_ids(
    xy:            np.ndarray,
    era5_tree:     cKDTree,
    era5_grid_ids: np.ndarray,
) -> np.ndarray:
    """Return the nearest ERA5 grid_id for each (x, y) coordinate."""
    _, idx = era5_tree.query(xy, workers=1)
    return era5_grid_ids[idx]


def _nearest_era5_time(era5: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return the ERA5 slice at the valid_time closest to t."""
    times  = era5["valid_time"].unique()
    deltas = np.abs(pd.to_datetime(times) - t)
    best   = times[np.argmin(deltas)]
    return era5[era5["valid_time"] == best]


# ── Point features at receptor B ──────────────────────────────────────────────

def join_static(
    b_grid_ids:  np.ndarray,
    grid_static: pd.DataFrame = None,
    cache:       FeatureCache = None,
) -> pd.DataFrame:
    """Join static terrain/landcover features at receptor B grid cells."""
    if cache is not None:
        gs_idx    = cache.gs_idx
        defaults  = cache.gs_defaults
    else:
        gs_idx   = grid_static.set_index("grid_id")[["dtm", "slope", "aspect", "fuel_type"]]
        defaults = {
            "dtm":       float(grid_static["dtm"].median()),
            "slope":     float(grid_static["slope"].median()),
            "aspect":    float(grid_static["aspect"].median()),
            "fuel_type": int(grid_static["fuel_type"].mode().iloc[0]),
        }
    result = pd.DataFrame({"grid_id": b_grid_ids}).join(gs_idx, on="grid_id")
    result = result[["dtm", "slope", "aspect", "fuel_type"]].reset_index(drop=True)
    # B cells outside the preprocessed grid extent produce NaN — fill with study medians
    result["dtm"].fillna(defaults["dtm"], inplace=True)
    result["slope"].fillna(defaults["slope"], inplace=True)
    result["aspect"].fillna(defaults["aspect"], inplace=True)
    result["fuel_type"].fillna(defaults["fuel_type"], inplace=True)
    return result


def join_weather(
    b_grid_ids:  np.ndarray,
    t1:          pd.Timestamp,
    era5:        pd.DataFrame = None,
    era5_tree:   cKDTree = None,
    era5_gids:   np.ndarray = None,
    cache:       FeatureCache = None,
) -> pd.DataFrame:
    """Join ERA5 weather features at receptor B cells for the nearest hour to t1."""
    b_xy   = np.array([list(map(float, g.split("_"))) for g in b_grid_ids])
    e_gids = _nearest_era5_grid_ids(b_xy, era5_tree, era5_gids)

    era5_idx = _nearest_in(t1, cache.era5_times, cache.era5_by_time) if cache is not None \
               else _nearest_era5_time(era5, t1).set_index("grid_id")[["temp_c", "rh", "wind_speed", "wind_dir"]]

    result = pd.DataFrame({"grid_id": e_gids}).join(era5_idx, on="grid_id")
    return result[["temp_c", "rh", "wind_speed", "wind_dir"]].reset_index(drop=True)


def join_fwi(
    b_grid_ids:  np.ndarray,
    t1:          pd.Timestamp,
    ffmc_daily:  pd.DataFrame = None,
    isi_hourly:  pd.DataFrame = None,
    ros_hourly:  pd.DataFrame = None,
    era5_tree:   cKDTree = None,
    era5_gids:   np.ndarray = None,
    cache:       FeatureCache = None,
) -> pd.DataFrame:
    """Join FFMC/ISI/ROS at receptor B cells for t1."""
    b_xy   = np.array([list(map(float, g.split("_"))) for g in b_grid_ids])
    e_gids = _nearest_era5_grid_ids(b_xy, era5_tree, era5_gids)

    if cache is not None:
        ffmc_idx = _nearest_in(t1.date(), cache.ffmc_dates, cache.ffmc_by_date)
        isi_idx  = _nearest_in(t1, cache.isi_times, cache.isi_by_time)
        ros_idx  = _nearest_in(t1, cache.ros_times, cache.ros_by_time)
    else:
        ffmc_day = ffmc_daily[ffmc_daily["date"] == t1.date()]
        ffmc_idx = ffmc_day.set_index("grid_id")[["ffmc"]]
        isi_idx  = _nearest_era5_time(isi_hourly, t1).set_index("grid_id")[["isi"]]
        ros_idx  = _nearest_era5_time(ros_hourly, t1).set_index("grid_id")[["ros"]]

    df = pd.DataFrame({"grid_id": e_gids})
    df = df.join(ffmc_idx, on="grid_id")
    df = df.join(isi_idx,  on="grid_id")
    df = df.join(ros_idx,  on="grid_id")
    return df[["ffmc", "isi", "ros"]].reset_index(drop=True)


# ── Fire geometry features ────────────────────────────────────────────────────

def fire_geometry_features(
    t1:         pd.Timestamp,
    fire_state: FireState,
) -> dict:
    """Compute scalar fire boundary descriptors at T1.

    Args:
        t1:         Target overpass timestamp.
        fire_state: FireState from build_fire_state.

    Returns:
        Dict with scalar float values:
            fire_age_h, perimeter_m, compactness,
            growth_rate_km2h, frp_per_area_km2, new_area_km2.
    """
    boundary = fire_state.boundary_after.get(t1)
    new_poly = fire_state.step_new_polys.get(t1)
    meta     = fire_state.step_cluster_meta.get(t1, [])
    area_km2 = fire_state.boundary_area_km2.get(t1, 0.0)
    steps    = fire_state.steps

    # Fire age
    fire_age_h = 0.0
    if fire_state.first_detection is not None:
        fire_age_h = (t1 - fire_state.first_detection).total_seconds() / 3600.0

    # Perimeter and compactness
    if boundary is not None and not boundary.is_empty:
        perimeter_m  = boundary.length
        compactness  = (4 * np.pi * boundary.area / perimeter_m ** 2
                        if perimeter_m > 0 else 0.0)
    else:
        perimeter_m = 0.0
        compactness = 0.0

    # Area growth rate [km²/h]
    growth_rate_km2h = 0.0
    idx = steps.index(t1) if t1 in steps else -1
    if idx > 0:
        t_prev = steps[idx - 1]
        prev_area = fire_state.boundary_area_km2.get(t_prev, 0.0)
        dt_h = (t1 - t_prev).total_seconds() / 3600.0
        if dt_h > 0:
            growth_rate_km2h = (area_km2 - prev_area) / dt_h

    # FRP per area
    total_frp = sum(float(c["frp"].sum()) for c in meta if len(c["frp"]) > 0)
    frp_per_area_km2 = total_frp / area_km2 if area_km2 > 0 else 0.0

    # New area at this step
    new_area_km2 = new_poly.area / 1e6 if new_poly is not None else 0.0

    return {
        "fire_age_h":         float(fire_age_h),
        "perimeter_m":        float(perimeter_m),
        "compactness":        float(compactness),
        "growth_rate_km2h":   float(growth_rate_km2h),
        "frp_per_area_km2":   float(frp_per_area_km2),
        "new_area_km2":       float(new_area_km2),
    }


# ── Distance to fire front ────────────────────────────────────────────────────

def dist_to_fire_front(
    b_xy:      np.ndarray,
    new_polys: object,
) -> np.ndarray:
    """Distance from each receptor B cell to the nearest fire front point.

    The fire front is the boundary of step_new_polys[T1] — the newly
    detected cluster hulls at T1.

    Args:
        b_xy:      Receptor coordinates, shape (n, 2).
        new_polys: Shapely geometry (Polygon / MultiPolygon) or None.

    Returns:
        Float64 array of shape (n,). Returns np.inf if new_polys is None.
    """
    if new_polys is None or new_polys.is_empty:
        return np.full(len(b_xy), np.inf, dtype=np.float64)

    # Sample fire front boundary as dense point cloud → fast cKDTree lookup
    boundary = new_polys.boundary
    if boundary.is_empty:
        return np.full(len(b_xy), np.inf, dtype=np.float64)

    # Extract boundary coords (works for LineString, MultiLineString)
    if boundary.geom_type == "LineString":
        front_pts = np.array(boundary.coords)
    else:
        front_pts = np.vstack([
            np.array(line.coords) for line in boundary.geoms
        ])

    if len(front_pts) == 0:
        return np.full(len(b_xy), np.inf, dtype=np.float64)

    tree = cKDTree(front_pts)
    dists, _ = tree.query(b_xy, workers=1)
    return dists.astype(np.float64)
