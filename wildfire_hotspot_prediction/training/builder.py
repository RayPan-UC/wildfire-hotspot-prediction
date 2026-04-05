"""
training/builder.py
-------------------
Main training data construction loop — staged pipeline.

Stages
------
1. pair_index   Build all valid (T1, T2) pairs.
                Saved → training/pair_index.parquet

2. fire_state   Accumulated fire boundaries at every timestep.
                Saved → training/fire_state.pkl

3. selectors    Receptor selector polygon per pair (wind-driven buffer).
                Saved → training/selectors.parquet  (GeoDataFrame, EPSG:3978)

4. sample+feat  Per-pair: sample receptor B cells, label them, join all
                features (static / weather / FWI / path / fire-geometry),
                then split into temporal k-folds.
                Saved → training/fold_<k>/train.parquet
                         training/fold_<k>/test.parquet

Each stage is skipped when its output already exists on disk and
``override_exist=False`` (default).  Set ``override_exist=True`` to
rebuild everything from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wildfire_hotspot_prediction.study               import Study
from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData
from wildfire_hotspot_prediction.training.pair_index        import build_pair_index
from wildfire_hotspot_prediction.training.fire_state        import (
    FireState,
    build_fire_state,
    save_fire_state,
    load_fire_state,
)
from wildfire_hotspot_prediction.training.receptor_selector import build_receptor_selector
from wildfire_hotspot_prediction.training.sampling          import sample_sources, sample_receptors
from wildfire_hotspot_prediction.training.sampling_path     import path_features
from wildfire_hotspot_prediction.training.features          import (
    build_era5_index,
    build_feature_cache,
    FeatureCache,
    join_static,
    join_weather,
    join_fwi,
    fire_geometry_features,
    dist_to_fire_front,
)
from wildfire_hotspot_prediction.utils.geo import snap_grid_ids

log = logging.getLogger(__name__)

_GRID_RES_M = 500.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_hotspot_data(proc_dir: Path) -> HotspotData:
    """Load hotspots.parquet → HotspotData."""
    hs_df = pd.read_parquet(proc_dir / "firms" / "hotspots.parquet")
    hs_df["overpass_time"] = pd.to_datetime(hs_df["overpass_time"])
    gdf = gpd.GeoDataFrame(
        hs_df,
        geometry=gpd.points_from_xy(hs_df["x_proj"], hs_df["y_proj"]),
        crs="EPSG:3978",
    )
    return HotspotData(
        gdf=gdf,
        overpass_times=sorted(gdf["overpass_time"].unique().tolist()),
    )


def _load_cloud_tree(clouds_dir: Path, t: pd.Timestamp) -> "cKDTree | None":
    """Load cloud parquet for timestamp t and return cKDTree, or None."""
    fname = t.strftime("%Y-%m-%dT%H%M") + ".parquet"
    path  = clouds_dir / fname
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "x_proj" not in df.columns:
        return None
    xy = df[["x_proj", "y_proj"]].values.astype(np.float64)
    return cKDTree(xy) if len(xy) > 0 else None


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _build_and_save_selectors(
    pair_index: pd.DataFrame,
    fire_state: FireState,
    path:       Path,
) -> dict[int, object]:
    """Build receptor selector per pair and save as GeoDataFrame parquet."""
    from tqdm import tqdm
    pair_ids, t1s, t2s, geoms = [], [], [], []
    for _, row in tqdm(pair_index.iterrows(), total=len(pair_index),
                       desc="selectors", unit="pair", ncols=80):
        sel = build_receptor_selector(row["T1"], fire_state)
        pair_ids.append(row["pair_id"])
        t1s.append(row["T1"])
        t2s.append(row["T2"])
        geoms.append(sel)   # None if no valid boundary

    gdf = gpd.GeoDataFrame(
        {"pair_id": pair_ids, "T1": t1s, "T2": t2s},
        geometry=geoms,
        crs="EPSG:3978",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)

    return {
        pid: geom
        for pid, geom in zip(pair_ids, geoms)
        if geom is not None
    }


def _load_selectors(path: Path) -> dict[int, object]:
    """Load selector GeoDataFrame → {pair_id: geometry}."""
    gdf = gpd.read_parquet(path)
    return {
        int(row["pair_id"]): row.geometry
        for _, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    }


# ── K-fold temporal split ─────────────────────────────────────────────────────

def _assign_folds(pair_index: pd.DataFrame, n_folds: int) -> pd.Series:
    """Assign each pair to fold 1..n_folds based on its T1 timestamp.

    The T1 time range is divided into n_folds equal-length segments.
    Pairs in the k-th segment are assigned fold k (1-indexed).

    Returns:
        Int Series aligned to pair_index with fold numbers 1..n_folds.
    """
    t_min = pair_index["T1"].min()
    t_max = pair_index["T1"].max()
    span  = (t_max - t_min).total_seconds()

    if span == 0:
        return pd.Series(1, index=pair_index.index)

    elapsed  = (pair_index["T1"] - t_min).dt.total_seconds()
    fold_idx = np.clip(
        (elapsed / span * n_folds).astype(int),
        0, n_folds - 1,
    )
    return pd.Series(fold_idx + 1, index=pair_index.index)


# ── Per-pair processing ───────────────────────────────────────────────────────

def _process_pair(
    row:          pd.Series,
    selector:     object,           # Shapely geometry
    hotspot_data: HotspotData,
    fire_state:   FireState,
    era5_tree:    cKDTree,
    era5_gids:    np.ndarray,
    clouds_dir:   Path,
    cache:        FeatureCache,
) -> pd.DataFrame | None:
    """Build one labeled DataFrame for a single (T1, T2) pair.

    Returns None if no source hotspots, no receptor candidates, or the
    selector was already None (handled by the caller).
    """
    t1        = row["T1"]
    t2        = row["T2"]
    pair_id   = row["pair_id"]
    delta_t_h = row["delta_t_h"]

    # ── Source A hotspots ─────────────────────────────────────────────────────
    a_xy, a_frp = sample_sources(t1, hotspot_data.gdf, fire_state)
    if len(a_xy) == 0:
        return None

    # ── Receptor B cells ──────────────────────────────────────────────────────
    gdf    = hotspot_data.gdf
    t1_xy  = gdf.loc[gdf["overpass_time"] == t1, ["x_proj", "y_proj"]].values.astype(np.float64)
    t2_df  = gdf.loc[gdf["overpass_time"] == t2, ["x_proj", "y_proj"]]
    t2_xy  = t2_df.values.astype(np.float64) if len(t2_df) > 0 else None

    cloud_tree = _load_cloud_tree(clouds_dir, t2)
    boundary   = fire_state.boundary_after.get(t1)

    b_xy, labels = sample_receptors(
        receptor_selector = selector,
        fire_boundaries   = boundary,
        t1_hotspot_xy     = t1_xy,
        t2_hotspot_xy     = t2_xy,
        cloud_tree        = cloud_tree,
        grid_res_m        = _GRID_RES_M,
    )
    if len(b_xy) == 0:
        return None

    # ── Nearest source A per receptor B ──────────────────────────────────────
    a_tree      = cKDTree(a_xy)
    _, a_idx    = a_tree.query(b_xy, workers=1)
    a_xy_nn     = a_xy[a_idx]              # (n, 2)
    frp_A       = a_frp[a_idx].astype(np.float32)   # FRP of nearest source

    # A→B Euclidean distance
    diff_ab     = b_xy - a_xy_nn
    dist_ab     = np.linalg.norm(diff_ab, axis=1).astype(np.float32)

    # Cluster-level features (main cluster at T1, scalar → broadcast)
    main_meta   = max(fire_state.step_cluster_meta.get(t1, [{}]),
                      key=lambda c: c.get("count", 0), default={})
    cluster_frp_sum  = np.float32(float(main_meta.get("frp", np.array([0])).sum()))
    cluster_n_pts    = np.float32(float(main_meta.get("count", 0)))
    cluster_hull_area = np.float32(float(main_meta.get("hull_area", 0.0)))
    cluster_density  = np.float32(
        cluster_n_pts / cluster_hull_area if cluster_hull_area > 0 else 0.0
    )

    # ── Grid IDs for B cells ──────────────────────────────────────────────────
    b_grid_ids = snap_grid_ids(b_xy, _GRID_RES_M)

    # ── Point features ────────────────────────────────────────────────────────
    static_df  = join_static(b_grid_ids, cache=cache)
    weather_df = join_weather(b_grid_ids, t1, era5_tree=era5_tree,
                               era5_gids=era5_gids, cache=cache)
    fwi_df     = join_fwi(b_grid_ids, t1, era5_tree=era5_tree,
                           era5_gids=era5_gids, cache=cache)

    # ── Path features (A→B) ───────────────────────────────────────────────────
    path_feats = path_features(a_xy_nn, b_xy, t1, t2, era5_tree=era5_tree,
                                era5_gids=era5_gids, cache=cache)

    # ── Fire geometry features (scalar → broadcast) ───────────────────────────
    geo_feats  = fire_geometry_features(t1, fire_state)

    # ── Distance to fire front ────────────────────────────────────────────────
    new_polys  = fire_state.step_new_polys.get(t1)
    dist_front = dist_to_fire_front(b_xy, new_polys).astype(np.float32)

    # ── Interaction feature ───────────────────────────────────────────────────
    frp_x_wind = (frp_A * path_feats["wind_alignment_mean"]).astype(np.float32)

    # ── Log transforms ────────────────────────────────────────────────────────
    log_dist = np.log1p(dist_ab).astype(np.float32)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    return pd.DataFrame({
        # identifiers
        "pair_id":   pair_id,
        "T1":        t1,
        "T2":        t2,
        "delta_t_h": np.float32(delta_t_h),
        "b_grid_id": b_grid_ids,
        "b_x":       b_xy[:, 0].astype(np.float32),
        "b_y":       b_xy[:, 1].astype(np.float32),
        "a_x":       a_xy_nn[:, 0].astype(np.float32),
        "a_y":       a_xy_nn[:, 1].astype(np.float32),
        # label
        "label":     labels,
        # distance A→B
        "dist":                 dist_ab,
        "log_dist":             log_dist,
        # source FRP
        "frp_A":                frp_A,
        # static
        "dtm":       static_df["dtm"].values.astype(np.float32),
        "slope":     static_df["slope"].values.astype(np.float32),
        "aspect":    static_df["aspect"].values.astype(np.float32),
        "fuel_type": static_df["fuel_type"].values.astype(np.int16),
        # weather
        "temp_c":     weather_df["temp_c"].values.astype(np.float32),
        "rh":         weather_df["rh"].values.astype(np.float32),
        "wind_speed": weather_df["wind_speed"].values.astype(np.float32),
        "wind_dir":   weather_df["wind_dir"].values.astype(np.float32),
        # FWI
        "ffmc": fwi_df["ffmc"].values.astype(np.float32),
        "isi":  fwi_df["isi"].values.astype(np.float32),
        "ros":  fwi_df["ros"].values.astype(np.float32),
        # path
        "grade":                path_feats["grade"],
        "slope_mean":           path_feats["slope_mean"],
        "slope_std":            path_feats["slope_std"],
        "wind_speed_mean":      path_feats["wind_speed_mean"],
        "wind_alignment_mean":  path_feats["wind_alignment_mean"],
        "wind_alignment_max":   path_feats["wind_alignment_max"],
        "wind_align_product":   path_feats["wind_align_product"],
        # interaction
        "frp_x_wind":           frp_x_wind,
        # cluster (scalar → broadcast)
        "cluster_frp_sum":   cluster_frp_sum,
        "cluster_n_pts":     cluster_n_pts,
        "cluster_hull_area": cluster_hull_area,
        "cluster_density":   cluster_density,
        # fire geometry (scalar → broadcast)
        "fire_age_h":        np.float32(geo_feats["fire_age_h"]),
        "perimeter_m":       np.float32(geo_feats["perimeter_m"]),
        "compactness":       np.float32(geo_feats["compactness"]),
        "growth_rate_km2h":  np.float32(geo_feats["growth_rate_km2h"]),
        "frp_per_area_km2":  np.float32(geo_feats["frp_per_area_km2"]),
        "new_area_km2":      np.float32(geo_feats["new_area_km2"]),
        # distance to fire front
        "dist_to_fire_front": dist_front,
    })


# ── Public API ────────────────────────────────────────────────────────────────

def build_training_data(
    study:          Study,
    n_folds:        int   = 3,
    max_steps:      int   = 1,
    grid_res_m:     float = 500.0,
    override_exist: bool  = False,
) -> None:
    """Build labeled training data with k-fold temporal splits.

    Runs four stages in sequence, each skipped when its output already
    exists on disk (unless override_exist=True):

    1. pair_index   → training/pair_index.parquet
    2. fire_state   → training/fire_state.pkl
    3. selectors    → training/selectors.parquet
    4. sample+feat  → training/fold_<k>/train|test.parquet

    Args:
        study:          Study instance.
        n_folds:        Number of temporal folds (1-indexed). Defaults to 3.
        max_steps:      Max overpass hops per pair. Defaults to 1.
        grid_res_m:     Grid cell size [m]. Defaults to 500.
        override_exist: If True, rebuild all stages even if outputs exist.
    """
    print("[build_training_data] starting training data build ...")
    proc_dir  = study.data_processed_dir
    train_dir = proc_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    pair_index_path = train_dir / "pair_index.parquet"
    fire_state_path = train_dir / "fire_state.pkl"
    selectors_path  = train_dir / "selectors.parquet"

    # ── Load hotspot data (always needed) ─────────────────────────────────────
    print("[build_training_data] loading hotspots ...")
    hotspot_data = _load_hotspot_data(proc_dir)

    # ── Stage 1: pair index ───────────────────────────────────────────────────
    if override_exist or not pair_index_path.exists():
        print("[build_training_data] stage 1: building pair index ...")
        pair_index = build_pair_index(hotspot_data, max_steps=max_steps)
        pair_index.to_parquet(pair_index_path, index=False)
        print(f"  {len(pair_index)} pairs  ->  pair_index.parquet")
    else:
        print("[build_training_data] stage 1: loading pair index ...")
        pair_index = pd.read_parquet(pair_index_path)
        print(f"  {len(pair_index)} pairs loaded")

    if pair_index.empty:
        print("[build_training_data] no valid pairs — stopping")
        return

    # ── Stage 2: fire state ───────────────────────────────────────────────────
    if override_exist or not fire_state_path.exists():
        print("[build_training_data] stage 2: building fire state ...")
        fire_state = build_fire_state(hotspot_data)
        save_fire_state(fire_state, fire_state_path)
        print(f"  {len(fire_state.steps)} active steps  ->  fire_state.pkl")
    else:
        print("[build_training_data] stage 2: loading fire state ...")
        fire_state = load_fire_state(fire_state_path)
        print(f"  {len(fire_state.steps)} active steps loaded")

    # ── Stage 3: receptor selectors ───────────────────────────────────────────
    if override_exist or not selectors_path.exists():
        print("[build_training_data] stage 3: building receptor selectors ...")
        sel_map = _build_and_save_selectors(pair_index, fire_state, selectors_path)
        print(f"  {len(sel_map)} selectors  ->  selectors.parquet")
    else:
        print("[build_training_data] stage 3: loading selectors ...")
        sel_map = _load_selectors(selectors_path)
        print(f"  {len(sel_map)} selectors loaded")

    # Load era5 for stage 4 feature extraction
    era5 = pd.read_parquet(proc_dir / "weather" / "era5.parquet")

    # ── Check final fold output ────────────────────────────────────────────────
    if not override_exist:
        all_folds_exist = all(
            (train_dir / f"fold_{k}" / split).exists()
            for k in range(1, n_folds + 1)
            for split in ("train.parquet", "test.parquet")
        )
        if all_folds_exist:
            print("[build_training_data] fold parquets exist, skipping stage 4 (override_exist=False)")
            return

    # ── Stage 4: sampling + features + fold split ─────────────────────────────
    print("[build_training_data] stage 4: sampling and feature extraction ...")

    ffmc_daily  = pd.read_parquet(proc_dir / "weather" / "ffmc_daily.parquet")
    isi_hourly  = pd.read_parquet(proc_dir / "weather" / "isi_hourly.parquet")
    ros_hourly  = pd.read_parquet(proc_dir / "weather" / "ros_hourly.parquet")
    grid_static = pd.read_parquet(proc_dir / "grid_static.parquet")
    clouds_dir  = proc_dir / "clouds"

    if ffmc_daily["date"].dtype == object:
        ffmc_daily["date"] = pd.to_datetime(ffmc_daily["date"]).dt.date

    era5_tree, era5_gids = build_era5_index(era5)
    cache = build_feature_cache(grid_static, era5, ffmc_daily, isi_hourly, ros_hourly)

    pairs_with_sel = pair_index[pair_index["pair_id"].isin(sel_map)]
    print(f"  {len(pairs_with_sel)} / {len(pair_index)} pairs have selectors")

    from tqdm import tqdm
    all_dfs:  list[pd.DataFrame] = []
    n_skipped = 0

    for _, row in tqdm(pairs_with_sel.iterrows(), total=len(pairs_with_sel),
                       desc="pairs", unit="pair", ncols=80):
        selector = sel_map.get(row["pair_id"])
        if selector is None:
            n_skipped += 1
            continue
        try:
            df = _process_pair(
                row, selector, hotspot_data, fire_state,
                era5_tree, era5_gids,
                clouds_dir, cache=cache,
            )
        except Exception as exc:
            log.warning("[builder] pair %s failed: %s", row["pair_id"], exc)
            df = None

        if df is not None and not df.empty:
            all_dfs.append(df)
        else:
            n_skipped += 1

    print(f"[build_training_data] {len(all_dfs)} pairs OK, {n_skipped} skipped")

    if not all_dfs:
        log.warning("[builder] all pairs produced empty results")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[build_training_data] total rows: {len(full_df):,}")

    # ── K-fold split and save ─────────────────────────────────────────────────
    fold_series = _assign_folds(pair_index, n_folds)
    fold_map    = pair_index.set_index("pair_id").assign(fold=fold_series.values)["fold"]
    full_df["fold"] = full_df["pair_id"].map(fold_map)

    for k in range(1, n_folds + 1):
        fold_dir = train_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_df  = full_df[full_df["fold"] == k].drop(columns="fold")
        train_df = full_df[full_df["fold"] != k].drop(columns="fold")

        test_df.to_parquet(fold_dir / "test.parquet",  index=False)
        train_df.to_parquet(fold_dir / "train.parquet", index=False)
        log.info("[builder] fold_%d  train=%d  test=%d", k, len(train_df), len(test_df))

    print(f"[build_training_data] done  ->  {train_dir}")
