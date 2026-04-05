"""
training/sampling_path.py
-------------------------
Path-based features along the A→B fire-spread vector for each receptor B cell.

Samples equidistant points along each source A → receptor B segment,
snaps them to the 500 m terrain / weather grid, and aggregates:
    grade                  net elevation rise / horizontal distance
    slope_mean / slope_std mean and std of terrain slope along path
    wind_speed_mean        mean wind speed along path (time-interpolated)
    wind_alignment_mean    mean wind alignment with spread direction [0, 1]
    wind_alignment_max     maximum wind alignment along path
    wind_align_product     product of wind alignment values along path

Temporal wind interpolation (ROS-based):
    Fire travels from A at the local ROS (m/min, from FBP/FWI at t1).
    Each sample point at distance d from A is reached at:
        t_arrive = t1 + d / ROS
    Capped at t2. ERA5 wind at that point is linearly interpolated between
    the ERA5 snapshots nearest to t1 and t2:
        wind(t) = wind_t1 * (1 - frac) + wind_t2 * frac
        frac = clip((t_arrive - t1) / (t2 - t1), 0, 1)
    Wind vectors are interpolated (not angles) to avoid 0/360 wraparound.

Wind convention:
    ERA5 wind_dir is "from" direction (meteorological).
    Converted to "to" direction for alignment with spread bearing.
    Alignment = (cos(spread_bearing − wind_to) + 1) / 2  ∈ [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wildfire_hotspot_prediction.utils.geo import snap_grid_ids
from wildfire_hotspot_prediction.training.features import (
    FeatureCache,
    _nearest_in,
    _nearest_era5_grid_ids,
    _nearest_era5_time,
)

_GRID_RES      = 500.0
_PATH_SPACING  = 500.0   # sample every 500 m along A→B
_PATH_MIN_SAMP = 3       # minimum number of path samples


def path_features(
    a_xy:        np.ndarray,
    b_xy:        np.ndarray,
    t1:          pd.Timestamp,
    t2:          pd.Timestamp,
    grid_static: pd.DataFrame = None,
    era5:        pd.DataFrame = None,
    era5_tree:   "cKDTree | None" = None,
    era5_gids:   np.ndarray = None,
    cache:       FeatureCache = None,
) -> dict[str, np.ndarray]:
    """Compute path-based features along each A→B spread vector.

    Samples points every _PATH_SPACING metres along each A→B segment,
    snaps them to the 500 m grid, and aggregates terrain and wind values.
    Wind is temporally interpolated using ROS-estimated arrival times.

    Args:
        a_xy:        Source coordinates, shape (n, 2).
        b_xy:        Receptor coordinates, shape (n, 2).
        t1:          Start timestamp (used for terrain + ROS lookup).
        t2:          End timestamp (used for wind interpolation endpoint).
        grid_static: grid_static.parquet DataFrame (required if cache is None).
        era5:        era5.parquet DataFrame (required if cache is None).
        era5_tree:   cKDTree from build_era5_index.
        era5_gids:   ERA5 grid_id array from build_era5_index.
        cache:       FeatureCache for fast lookup.

    Returns:
        Dict with float32 arrays of shape (n,):
            grade, slope_mean, slope_std,
            wind_speed_mean, wind_alignment_mean,
            wind_alignment_max, wind_align_product.
    """
    n_pairs = len(a_xy)
    diff    = b_xy - a_xy                                 # (n, 2)
    dists   = np.linalg.norm(diff, axis=1)                # (n,)

    # Spread unit vector (east, north)
    safe_d = np.maximum(dists, 1e-6)
    uv     = diff / safe_d[:, None]                       # (n, 2)

    # Number of path samples (shared across all pairs for vectorisation)
    n_samp = max(_PATH_MIN_SAMP, int(np.max(dists) / _PATH_SPACING) + 1)
    fracs  = np.linspace(0.0, 1.0, n_samp)               # (n_samp,)

    # Sample coordinates: (n, n_samp, 2)
    pts      = a_xy[:, None, :] + fracs[None, :, None] * diff[:, None, :]
    pts_flat = pts.reshape(-1, 2)

    # Snap to 500 m grid → grid_ids
    samp_gids = snap_grid_ids(pts_flat, _GRID_RES)        # (n * n_samp,)

    # ── Terrain lookup ────────────────────────────────────────────────────────
    gs_idx  = cache.gs_idx[["dtm", "slope"]] if cache is not None \
              else grid_static.set_index("grid_id")[["dtm", "slope"]]
    samp_df = pd.DataFrame({"grid_id": samp_gids}).join(gs_idx, on="grid_id")

    dtm_default   = cache.gs_defaults["dtm"]   if cache is not None else 0.0
    slope_default = cache.gs_defaults["slope"] if cache is not None else 0.0
    dtm_flat   = samp_df["dtm"].fillna(dtm_default).values.astype(np.float32)
    slope_flat = samp_df["slope"].fillna(slope_default).values.astype(np.float32)

    dtm_s   = dtm_flat.reshape(n_pairs, n_samp)           # (n, n_samp)
    slope_s = slope_flat.reshape(n_pairs, n_samp)

    grade      = ((dtm_s[:, -1] - dtm_s[:, 0]) / safe_d).astype(np.float32)
    slope_mean = np.nanmean(slope_s, axis=1).astype(np.float32)
    slope_std  = np.nanstd(slope_s,  axis=1).astype(np.float32)

    # ── ROS-based arrival time fraction per sample point ─────────────────────
    # ROS at source A (m/min, FBP) → convert to m/h for distance/time calc
    e_a_gids = _nearest_era5_grid_ids(a_xy, era5_tree, era5_gids)
    ros_idx  = _nearest_in(t1, cache.ros_times, cache.ros_by_time)
    ros_mph  = (ros_idx.reindex(e_a_gids)["ros"]
                .fillna(1.0).values.astype(np.float64) * 60.0)  # m/min → m/h

    # Distance of each sample point from A: (n, n_samp)
    sample_d = dists[:, None] * fracs[None, :]            # metres

    # Hours to reach each sample point from A
    arrive_h = sample_d / np.maximum(ros_mph[:, None], 1.0)

    # Normalise to fraction of t1→t2 window, clamped to [0, 1]
    delta_h   = max((t2 - t1).total_seconds() / 3600.0, 1e-6)
    time_frac = np.clip(arrive_h / delta_h, 0.0, 1.0).astype(np.float32)  # (n, n_samp)

    # ── Temporally interpolated wind lookup ───────────────────────────────────
    e_samp_gids = _nearest_era5_grid_ids(pts_flat, era5_tree, era5_gids)

    era5_t1_idx = _nearest_in(t1, cache.era5_times, cache.era5_by_time) \
                  [["wind_speed", "wind_dir"]] if cache is not None \
                  else _nearest_era5_time(era5, t1).set_index("grid_id")[["wind_speed", "wind_dir"]]
    era5_t2_idx = _nearest_in(t2, cache.era5_times, cache.era5_by_time) \
                  [["wind_speed", "wind_dir"]] if cache is not None \
                  else _nearest_era5_time(era5, t2).set_index("grid_id")[["wind_speed", "wind_dir"]]

    def _wind_at(idx) -> tuple[np.ndarray, np.ndarray]:
        wdf = pd.DataFrame({"grid_id": e_samp_gids}).join(idx, on="grid_id")
        ws  = wdf["wind_speed"].fillna(0.0).values.astype(np.float32).reshape(n_pairs, n_samp)
        wd  = wdf["wind_dir"].fillna(0.0).values.astype(np.float32).reshape(n_pairs, n_samp)
        return ws, wd

    ws_t1, wd_t1 = _wind_at(era5_t1_idx)
    ws_t2, wd_t2 = _wind_at(era5_t2_idx)

    f = time_frac  # (n, n_samp)

    # Interpolate wind speed (scalar)
    ws_s = ws_t1 * (1.0 - f) + ws_t2 * f

    # Interpolate wind direction as unit vectors ("from" → "to", then interpolate)
    wto_t1 = np.deg2rad((wd_t1 + 180.0) % 360.0)
    wto_t2 = np.deg2rad((wd_t2 + 180.0) % 360.0)
    we = np.sin(wto_t1) * (1.0 - f) + np.sin(wto_t2) * f   # east component
    wn = np.cos(wto_t1) * (1.0 - f) + np.cos(wto_t2) * f   # north component

    # Alignment of interpolated wind vector with spread direction, scaled [0, 1]
    align        = uv[:, 0:1] * we + uv[:, 1:2] * wn        # dot product (n, n_samp)
    align_scaled = ((align + 1.0) / 2.0).astype(np.float32)

    return {
        "grade":                grade,
        "slope_mean":           slope_mean,
        "slope_std":            slope_std,
        "wind_speed_mean":      np.nanmean(ws_s,         axis=1).astype(np.float32),
        "wind_alignment_mean":  np.nanmean(align_scaled, axis=1).astype(np.float32),
        "wind_alignment_max":   np.nanmax(align_scaled,  axis=1).astype(np.float32),
        "wind_align_product":   np.nanprod(align_scaled, axis=1).astype(np.float32),
    }
