"""
build_prediction_data/feature_builder.py
-----------------------------------------
Build inference features for an arbitrary (T1, T1+delta_t) pair without
requiring observed T2 hotspots.

Designed to be called from wildfire-decision-support (or any system)
to generate features for a user-specified future time offset:

    from wildfire_hotspot_prediction import build_prediction_features

    feature_df = build_prediction_features(study, t1="2016-05-03T08:54:00", delta_t_h=6.0)
    # Pass to WildfirePredictor.predict(feature_df)

The output DataFrame contains all columns expected by ``feature_cols.json``
plus ``b_x``, ``b_y``, ``b_grid_id`` metadata columns for GeoJSON rendering.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.training.fire_state import load_fire_state
from wildfire_hotspot_prediction.training.receptor_selector import build_receptor_selector
from wildfire_hotspot_prediction.training.sampling import sample_sources, sample_receptors
from wildfire_hotspot_prediction.training.sampling_path import path_features
from wildfire_hotspot_prediction.training.features import (
    build_era5_index,
    build_feature_cache,
    join_static,
    join_weather,
    join_fwi,
    fire_geometry_features,
    dist_to_fire_front,
    _nearest_era5_grid_ids,
)
from wildfire_hotspot_prediction.utils.geo import snap_grid_ids

log = logging.getLogger(__name__)

_GRID_RES_M = 500.0


def build_prediction_features(
    study:     Study,
    t1:        "pd.Timestamp | str",
    delta_t_h: float,
) -> pd.DataFrame:
    """Build inference features for (T1, T1+delta_t) without T2 observations.

    T1 is snapped to the nearest observed overpass in fire_state. T2 is
    computed synthetically as T1 + delta_t_h hours and used only for
    path-feature wind interpolation.

    Args:
        study:     Study instance.
        t1:        Requested T1 timestamp (snapped to nearest overpass).
        delta_t_h: Hours ahead to predict (e.g. 3.0, 6.0, 12.0).

    Returns:
        DataFrame with feature columns matching ``feature_cols.json`` plus
        ``b_x``, ``b_y``, ``b_grid_id``, ``t1``, ``t2`` metadata columns.
        Empty DataFrame if no valid receptor candidates exist.

    Raises:
        FileNotFoundError: If ``fire_state.pkl`` or processed parquets are missing.
    """
    t1 = pd.Timestamp(t1)
    proc_dir  = study.data_processed_dir
    train_dir = proc_dir / "training"

    # ── Load fire state ───────────────────────────────────────────────────────
    fire_state_path = train_dir / "fire_state.pkl"
    if not fire_state_path.exists():
        raise FileNotFoundError(
            f"{fire_state_path} not found — run build_training_data() first."
        )
    fire_state = load_fire_state(fire_state_path)

    # ── Snap t1 to nearest observed overpass ──────────────────────────────────
    steps = fire_state.steps
    if not steps:
        log.warning("[feature_builder] fire_state has no steps")
        return pd.DataFrame(), {}

    t1_actual = min(steps, key=lambda s: abs((s - t1).total_seconds()))
    log.info("[feature_builder] t1 snapped %s → %s", t1, t1_actual)

    # ── Synthetic T2 ──────────────────────────────────────────────────────────
    t2 = t1_actual + timedelta(hours=delta_t_h)

    # ── Receptor selector for t1_actual ──────────────────────────────────────
    selector = _load_selector(train_dir, t1_actual)
    if selector is None:
        selector = build_receptor_selector(t1_actual, fire_state)
    if selector is None or selector.is_empty:
        log.warning("[feature_builder] no receptor selector for %s", t1_actual)
        return pd.DataFrame(), {}

    # ── Load hotspots ─────────────────────────────────────────────────────────
    hs_df = pd.read_parquet(proc_dir / "firms" / "hotspots.parquet")
    hs_df["overpass_time"] = pd.to_datetime(hs_df["overpass_time"])

    # ── Source A hotspots at T1 ───────────────────────────────────────────────
    import geopandas as gpd
    from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData

    hs_gdf = gpd.GeoDataFrame(
        hs_df,
        geometry=gpd.points_from_xy(hs_df["x_proj"], hs_df["y_proj"]),
        crs="EPSG:3978",
    )
    hotspot_data = HotspotData(
        gdf=hs_gdf,
        overpass_times=sorted(hs_gdf["overpass_time"].unique().tolist()),
    )

    a_xy, a_frp = sample_sources(t1_actual, hotspot_data.gdf, fire_state)
    if len(a_xy) == 0:
        log.warning("[feature_builder] no source hotspots at %s", t1_actual)
        return pd.DataFrame(), {}

    # ── T1 hotspot XY for receptor exclusion ─────────────────────────────────
    t1_mask = hs_gdf["overpass_time"] == t1_actual
    t1_xy   = hs_gdf.loc[t1_mask, ["x_proj", "y_proj"]].values.astype(np.float64)

    # ── Receptor B cells (no T2 obs, no cloud filter) ────────────────────────
    boundary = fire_state.boundary_after.get(t1_actual)
    b_xy, _  = sample_receptors(
        receptor_selector = selector,
        fire_boundaries   = boundary,
        t1_hotspot_xy     = t1_xy,
        t2_hotspot_xy     = None,   # no T2 observations for inference
        cloud_tree        = None,   # no cloud filtering for inference
        grid_res_m        = _GRID_RES_M,
    )
    if len(b_xy) == 0:
        log.warning("[feature_builder] no receptor candidates at %s", t1_actual)
        return pd.DataFrame(), {}

    log.info("[feature_builder] %d receptor candidates  t1=%s  delta_t=%.1fh", len(b_xy), t1_actual, delta_t_h)

    # ── Nearest source A per receptor B ──────────────────────────────────────
    a_tree   = cKDTree(a_xy)
    _, a_idx = a_tree.query(b_xy, workers=1)
    a_xy_nn  = a_xy[a_idx]
    frp_A    = a_frp[a_idx].astype(np.float32)

    dist_ab  = np.linalg.norm(b_xy - a_xy_nn, axis=1).astype(np.float32)
    log_dist = np.log1p(dist_ab).astype(np.float32)

    # ── Cluster-level scalars ─────────────────────────────────────────────────
    main_meta = max(
        fire_state.step_cluster_meta.get(t1_actual, [{}]),
        key=lambda c: c.get("count", 0),
        default={},
    )
    cluster_frp_sum   = np.float32(float(main_meta.get("frp", np.array([0])).sum()))
    cluster_n_pts     = np.float32(float(main_meta.get("count", 0)))
    cluster_hull_area = np.float32(float(main_meta.get("hull_area", 0.0)))
    cluster_density   = np.float32(
        cluster_n_pts / cluster_hull_area if cluster_hull_area > 0 else 0.0
    )

    # ── Grid IDs for B cells ──────────────────────────────────────────────────
    b_grid_ids = snap_grid_ids(b_xy, _GRID_RES_M)

    # ── Feature cache ─────────────────────────────────────────────────────────
    era5        = pd.read_parquet(proc_dir / "weather" / "era5.parquet")
    ffmc_daily  = pd.read_parquet(proc_dir / "weather" / "ffmc_daily.parquet")
    isi_hourly  = pd.read_parquet(proc_dir / "weather" / "isi_hourly.parquet")
    ros_hourly  = pd.read_parquet(proc_dir / "weather" / "ros_hourly.parquet")
    grid_static = pd.read_parquet(proc_dir / "grid_static.parquet")

    if ffmc_daily["date"].dtype == object:
        ffmc_daily["date"] = pd.to_datetime(ffmc_daily["date"]).dt.date

    era5_tree, era5_gids = build_era5_index(era5)
    cache = build_feature_cache(grid_static, era5, ffmc_daily, isi_hourly, ros_hourly)

    # ── Point features ────────────────────────────────────────────────────────
    static_df  = join_static(b_grid_ids, cache=cache)
    weather_df = join_weather(b_grid_ids, t1_actual,
                              era5_tree=era5_tree, era5_gids=era5_gids, cache=cache)
    fwi_df     = join_fwi(b_grid_ids, t1_actual,
                          era5_tree=era5_tree, era5_gids=era5_gids, cache=cache)

    # ── Path features (A→B, wind interpolated to synthetic T2) ───────────────
    path_feats = path_features(a_xy_nn, b_xy, t1_actual, t2,
                               era5_tree=era5_tree, era5_gids=era5_gids, cache=cache)

    # ── Fire geometry features ────────────────────────────────────────────────
    geo_feats  = fire_geometry_features(t1_actual, fire_state)

    # ── Distance to fire front ────────────────────────────────────────────────
    new_polys  = fire_state.step_new_polys.get(t1_actual)
    dist_front = dist_to_fire_front(b_xy, new_polys).astype(np.float32)

    # ── Interaction feature ───────────────────────────────────────────────────
    frp_x_wind = (frp_A * path_feats["wind_alignment_mean"]).astype(np.float32)

    # ── Intermediates (fire context for report / spatial stage) ───────────────
    steps_sorted = sorted(steps)
    t1_idx = steps_sorted.index(t1_actual) if t1_actual in steps_sorted else -1
    t0            = steps_sorted[t1_idx - 1] if t1_idx > 0 else None
    actual_delta_h = float((t1_actual - t0).total_seconds() / 3600.0) if t0 else None

    burned_area_km2 = fire_state.boundary_area_km2.get(t1_actual, 0.0)
    n_hotspots_t1   = int(t1_mask.sum())
    frp_sum_t1      = float(hs_gdf.loc[t1_mask, "frp"].sum())

    # vector-mean wind over receptors (avoids 0/360 wraparound)
    ws_arr = weather_df["wind_speed"].values.astype(np.float64)
    wd_rad = np.deg2rad(weather_df["wind_dir"].values.astype(np.float64))
    u_mean = np.nanmean(-np.sin(wd_rad) * ws_arr)
    v_mean = np.nanmean(-np.cos(wd_rad) * ws_arr)

    weather_t1 = {
        "wind_speed_kmh": round(float(np.nanmean(ws_arr)) * 3.6, 1),
        "wind_dir":       round(float((np.degrees(np.arctan2(-u_mean, -v_mean)) + 360) % 360), 1),
        "temp_c":         round(float(np.nanmean(weather_df["temp_c"].values)), 1),
        "rh":             round(float(np.nanmean(weather_df["rh"].values)), 1),
    }

    ros_arr = fwi_df["ros"].values.astype(np.float64)
    fwi_t1 = {
        "ffmc":        round(float(np.nanmean(fwi_df["ffmc"].values)), 1),
        "isi":         round(float(np.nanmean(fwi_df["isi"].values)), 1),
        "ros_mean_mh": round(float(np.nanmean(ros_arr)) * 60.0, 1),  # m/min → m/h
        "ros_max_mh":  round(float(np.nanmax(ros_arr))  * 60.0, 1),
    }

    # wind forecast: t1 → t1+12h hourly at fire centroid
    wind_forecast = []
    boundary_t1 = fire_state.boundary_after.get(t1_actual)
    if boundary_t1 is not None and not boundary_t1.is_empty:
        centroid_xy  = np.array([[boundary_t1.centroid.x, boundary_t1.centroid.y]])
        centroid_gid = _nearest_era5_grid_ids(centroid_xy, era5_tree, era5_gids)[0]
        for h in range(13):
            t = t1_actual + timedelta(hours=h)
            snap = _nearest_in(t, cache.era5_times, cache.era5_by_time)
            if centroid_gid in snap.index:
                row = snap.loc[centroid_gid]
                wind_forecast.append({
                    "hour":      h,
                    "speed_kmh": round(float(row["wind_speed"]) * 3.6, 1),
                    "dir":       round(float(row["wind_dir"]), 1),
                })

    intermediates = {
        "t1":             t1_actual.isoformat(),
        "t0":             t0.isoformat() if t0 else None,
        "actual_delta_h": round(actual_delta_h, 2) if actual_delta_h is not None else None,
        "fire": {
            "burned_area_km2":  round(burned_area_km2, 2),
            "new_area_km2":     round(geo_feats["new_area_km2"], 3),
            "growth_rate_km2h": round(geo_feats["growth_rate_km2h"], 3),
            "perimeter_m":      round(geo_feats["perimeter_m"], 1),
            "n_hotspots":       n_hotspots_t1,
            "frp_sum":          round(frp_sum_t1, 1),
        },
        "weather_t1":    weather_t1,
        "fwi_t1":        fwi_t1,
        "wind_forecast": wind_forecast,
    }

    # ── Assemble — column names must match feature_cols.json exactly ─────────
    return pd.DataFrame({
        # metadata (not features — used by backend for GeoJSON)
        "b_grid_id": b_grid_ids,
        "b_x":       b_xy[:, 0].astype(np.float32),
        "b_y":       b_xy[:, 1].astype(np.float32),
        "t1":        t1_actual,
        "t2":        t2,
        # ── features ────────────────────────────────────────────────────────
        "dist":     dist_ab,
        "log_dist": log_dist,
        "frp_A":    frp_A,
        # static terrain
        "dtm":       static_df["dtm"].values.astype(np.float32),
        "slope":     static_df["slope"].values.astype(np.float32),
        "aspect":    static_df["aspect"].values.astype(np.float32),
        "fuel_type": static_df["fuel_type"].values.astype(np.int16),
        # weather at T1
        "temp_c":     weather_df["temp_c"].values.astype(np.float32),
        "rh":         weather_df["rh"].values.astype(np.float32),
        "wind_speed": weather_df["wind_speed"].values.astype(np.float32),
        "wind_dir":   weather_df["wind_dir"].values.astype(np.float32),
        # FWI at T1
        "ffmc": fwi_df["ffmc"].values.astype(np.float32),
        "isi":  fwi_df["isi"].values.astype(np.float32),
        "ros":  fwi_df["ros"].values.astype(np.float32),
        # path A→B
        "grade":               path_feats["grade"],
        "slope_mean":          path_feats["slope_mean"],
        "slope_std":           path_feats["slope_std"],
        "wind_speed_mean":     path_feats["wind_speed_mean"],
        "wind_alignment_mean": path_feats["wind_alignment_mean"],
        "wind_alignment_max":  path_feats["wind_alignment_max"],
        "wind_align_product":  path_feats["wind_align_product"],
        # interaction
        "frp_x_wind": frp_x_wind,
        # cluster scalars
        "cluster_frp_sum":   cluster_frp_sum,
        "cluster_n_pts":     cluster_n_pts,
        "cluster_hull_area": cluster_hull_area,
        "cluster_density":   cluster_density,
        # fire geometry scalars
        "fire_age_h":      np.float32(geo_feats["fire_age_h"]),
        "perimeter_m":     np.float32(geo_feats["perimeter_m"]),
        "compactness":     np.float32(geo_feats["compactness"]),
        "growth_rate_km2h": np.float32(geo_feats["growth_rate_km2h"]),
        "frp_per_area_km2": np.float32(geo_feats["frp_per_area_km2"]),
        "new_area_km2":    np.float32(geo_feats["new_area_km2"]),
        # distance to fire front
        "dist_to_fire_front": dist_front,
        # pair metadata feature
        "delta_t_h": np.float32(delta_t_h),
    }), intermediates


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_selector(train_dir: Path, t1: pd.Timestamp):
    """Try loading the pre-built selector for t1 from selectors.parquet."""
    path = train_dir / "selectors.parquet"
    if not path.exists():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(path)
        row = gdf[gdf["T1"] == t1]
        if row.empty:
            return None
        geom = row.iloc[0].geometry
        return geom if geom is not None and not geom.is_empty else None
    except Exception as exc:
        log.warning("[feature_builder] could not load selector: %s", exc)
        return None
