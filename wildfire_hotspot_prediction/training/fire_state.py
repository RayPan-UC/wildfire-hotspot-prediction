"""
training/fire_state.py
----------------------
Build the accumulated fire boundary state at every timestep via a single
forward pass over all revisit steps.

For each timestep:
  1. Run DBSCAN on hotspot coordinates → cluster labels
  2. Build per-cluster hull = unary_union of per-point buffers
  3. Accumulate: boundary_after[t] = boundary_after[t-1] ∪ new_hulls

boundary_before[t] = boundary_after[t-1]  (state before T's new detections)
boundary_after[t]  = boundary_before[t] ∪ step_new_polys[t]

The forward pass guarantees that skip pairs always see the correct T1
boundary state regardless of processing order.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData


@dataclass
class FireState:
    """Accumulated fire boundary state for all timesteps.

    Attributes:
        boundary_before:   Timestamp → accumulated polygon before T's clusters.
                           None for the first timestep.
        boundary_after:    Timestamp → accumulated polygon including T's clusters.
                           None if no valid clusters have been seen yet.
        step_new_polys:    Timestamp → union of cluster hulls added at T.
                           None if T had no valid clusters.
        step_cluster_meta: Timestamp → list of dicts per cluster:
                             {cxy, frp, centroid, max_frp, count}
        boundary_area_km2: Timestamp → float, boundary_after area [km²].
        steps:             Sorted list of timesteps that produced ≥1 valid cluster.
        first_detection:   Earliest overpass timestamp in the dataset.
    """
    boundary_before:   dict = field(default_factory=dict)
    boundary_after:    dict = field(default_factory=dict)
    step_new_polys:    dict = field(default_factory=dict)
    step_cluster_meta: dict = field(default_factory=dict)
    boundary_area_km2: dict = field(default_factory=dict)
    steps:             list = field(default_factory=list)
    first_detection:   pd.Timestamp = None


def save_fire_state(fire_state: FireState, path: Path) -> None:
    """Persist a FireState to disk via pickle.

    Args:
        fire_state: FireState to save.
        path:       Destination file path (e.g. training/fire_state.pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fire_state, f)


def load_fire_state(path: Path) -> FireState:
    """Load a FireState previously saved with save_fire_state.

    Args:
        path: Path to the .pkl file.

    Returns:
        FireState instance.
    """
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def build_fire_state(
    hotspot_data:     HotspotData,
    dbscan_eps_m:     float = 2000.0,
    dbscan_min_pts:   int   = 2,
    hotspot_buffer_m: float = 500.0,
) -> FireState:
    """Run a forward pass over all timesteps to build the accumulated fire state.

    Args:
        hotspot_data:     Output of preprocess_hotspots.
        dbscan_eps_m:     DBSCAN neighbourhood radius [m]. Defaults to 2000.
        dbscan_min_pts:   DBSCAN minimum cluster size. Defaults to 2.
        hotspot_buffer_m: Per-point buffer radius for hull construction [m].
                          Defaults to 500 (matches grid resolution).

    Returns:
        FireState with boundary dicts and cluster metadata for all timesteps.
    """
    gdf    = hotspot_data.gdf
    times  = hotspot_data.overpass_times   # already sorted

    state = FireState()
    if not times:
        return state

    state.first_detection = times[0]
    accumulated = None   # shapely geometry, grows over time

    try:
        from tqdm import tqdm as _tqdm
        _iter = _tqdm(times, desc="fire_state", unit="step", ncols=80)
    except ImportError:
        _iter = times

    for t in _iter:
        mask = gdf["overpass_time"] == t
        sub  = gdf.loc[mask]

        pts     = sub[["x_proj", "y_proj"]].values.astype(np.float64)
        frp_arr = sub["frp"].values.astype(np.float32) if "frp" in sub.columns \
                  else np.ones(len(pts), dtype=np.float32)

        # ── State before this step ────────────────────────────────────────────
        state.boundary_before[t] = accumulated

        # ── DBSCAN ───────────────────────────────────────────────────────────
        cluster_hulls = []
        cluster_meta  = []

        if len(pts) >= dbscan_min_pts:
            labels = DBSCAN(eps=dbscan_eps_m,
                            min_samples=dbscan_min_pts).fit_predict(pts)

            for label in np.unique(labels):
                if label == -1:   # noise
                    continue
                m    = labels == label
                cxy  = pts[m]
                frp  = frp_arr[m]

                hull = unary_union([Point(x, y).buffer(hotspot_buffer_m)
                                    for x, y in cxy])
                cluster_hulls.append(hull)
                cluster_meta.append({
                    "cxy":       cxy,
                    "frp":       frp,
                    "centroid":  cxy.mean(axis=0),
                    "max_frp":   float(frp.max()),
                    "count":     int(m.sum()),
                    "hull_area": float(hull.area),   # m²
                })

        # ── Accumulate boundary ───────────────────────────────────────────────
        if cluster_hulls:
            new_polys   = unary_union(cluster_hulls)
            accumulated = unary_union([accumulated, new_polys]) \
                          if accumulated is not None else new_polys
            state.steps.append(t)
        else:
            new_polys = None

        state.step_new_polys[t]    = new_polys
        state.step_cluster_meta[t] = cluster_meta
        state.boundary_after[t]    = accumulated
        state.boundary_area_km2[t] = accumulated.area / 1e6 \
                                     if accumulated is not None else 0.0

    return state
