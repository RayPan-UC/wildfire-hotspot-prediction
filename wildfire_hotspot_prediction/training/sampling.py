"""
training/sampling.py
--------------------
Sample source A hotspots and receptor B candidate cells for a given pair.

Source A sampling:
    All hotspot detections within the largest cluster at T1 (by hotspot count).
    K=3 nearest source hotspots per receptor B cell.

Receptor B sampling:
    Uniform grid of GRID_RES × GRID_RES cells across the receptor_selector AOI.
    Two-pass filter to exclude already-burning cells:
      Pass 1: polygon containment against receptor_selector
              (already excludes boundary_after[T1] by construction)
      Pass 2: point-distance against T1 hotspots (dist > cell_radius_m)

Labeling:
    burned_B = 1  if any T2 hotspot falls within cell_radius_m of B centre
    burned_B = 0  otherwise
    Cloud-obscured cells are excluded before feature engineering (not sampled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from wildfire_hotspot_prediction.training.fire_state import FireState


# ---------------------------------------------------------------------------
# Shapely containment helper — works with shapely 1.x and 2.x
# ---------------------------------------------------------------------------
try:
    import shapely as _shapely
    _shapely.contains_xy          # raises AttributeError on shapely <2
    def _contains_xy(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return _shapely.contains_xy(geom, xs, ys)
except AttributeError:
    from shapely.prepared import prep as _prep
    def _contains_xy(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        pg = _prep(geom)
        return np.array([pg.contains(Point(x, y)) for x, y in zip(xs, ys)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_sources(
    t1:         pd.Timestamp,
    hotspot_df: pd.DataFrame,
    fire_state: FireState,
    k:          int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select source A hotspots from the main cluster at T1.

    "Main cluster" = largest cluster by hotspot count at T1.

    Args:
        t1:         T1 timestamp.
        hotspot_df: Full hotspot DataFrame (not used directly — metadata comes
                    from fire_state.step_cluster_meta).
        fire_state: FireState from build_fire_state.
        k:          Number of nearest sources per receptor (passed through for
                    caller convenience; not used here). Defaults to 3.

    Returns:
        Tuple of (cxy_main, frp_main):
            cxy_main: (n_main, 2) float64 array of main cluster coordinates.
            frp_main: (n_main,) float32 array of FRP values.
            Returns empty arrays if no clusters at T1.
    """
    cluster_meta = fire_state.step_cluster_meta.get(t1, [])
    if not cluster_meta:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float32)

    main = max(cluster_meta, key=lambda c: c["count"])
    return main["cxy"].copy(), main["frp"].copy()


def sample_receptors(
    receptor_selector: "shapely.geometry.base.BaseGeometry",
    fire_boundaries:   "shapely.geometry.base.BaseGeometry | None",
    t1_hotspot_xy:     np.ndarray,
    t2_hotspot_xy:     np.ndarray | None,
    cloud_tree:        "cKDTree | None",
    grid_res_m:        float = 500.0,
    cell_radius_m:     float = 353.6,
    cloud_radius_m:    float = 750.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and label receptor B candidate cells within the receptor_selector.

    Steps:
        1. Build a uniform grid snapped to grid_res_m over the bounding box.
        2. Keep cells whose centres fall inside receptor_selector.
        3. Exclude cells within cell_radius_m of any T1 hotspot (already burning).
        4. Exclude cells within cloud_radius_m of a cloudy pixel.
        5. Label: 1=burned at T2, 0=unburned.

    Args:
        receptor_selector: Receptor selector polygon from build_receptor_selector.
                           Already excludes boundary_after[T1].
        fire_boundaries:   Accumulated burned area polygon at T1 (currently unused
                           because receptor_selector already excludes it; kept for
                           future use).
        t1_hotspot_xy:     T1 hotspot coordinates (n, 2) for pass-2 exclusion.
        t2_hotspot_xy:     T2 hotspot coordinates (n, 2) for burned label.
                           None if no T2 detections.
        cloud_tree:        cKDTree built from cloudy pixel coordinates, or None.
        grid_res_m:        Grid cell size [m]. Defaults to 500.
        cell_radius_m:     Half-diagonal of a grid cell [m]. Defaults to 353.6
                           (= 500 / √2).
        cloud_radius_m:    Cloud pixel influence radius [m]. Defaults to 750.

    Returns:
        Tuple of (cand_xy, labels):
            cand_xy: (n_cells, 2) float64 array of candidate cell centres.
            labels:  (n_cells,) int8 array — 0=unburned, 1=burned.
    """
    # ── 1. Generate candidate grid snapped to grid_res_m ─────────────────────
    minx, miny, maxx, maxy = receptor_selector.bounds
    xs = np.arange(
        np.floor(minx / grid_res_m) * grid_res_m,
        np.ceil(maxx  / grid_res_m) * grid_res_m + grid_res_m,
        grid_res_m,
    )
    ys = np.arange(
        np.floor(miny / grid_res_m) * grid_res_m,
        np.ceil(maxy  / grid_res_m) * grid_res_m + grid_res_m,
        grid_res_m,
    )
    xx, yy = np.meshgrid(xs, ys)
    cand_xs = xx.ravel()
    cand_ys = yy.ravel()

    # ── 2. Pass-1: containment in receptor_selector ──────────────────────────
    in_aoi = _contains_xy(receptor_selector, cand_xs, cand_ys)
    cand_xs = cand_xs[in_aoi]
    cand_ys = cand_ys[in_aoi]

    if len(cand_xs) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int8)

    cand_xy = np.column_stack([cand_xs, cand_ys])

    # ── 3. Pass-2: exclude cells too close to T1 hotspots ────────────────────
    if t1_hotspot_xy is not None and len(t1_hotspot_xy) > 0:
        t1_tree = cKDTree(t1_hotspot_xy)
        dists, _ = t1_tree.query(cand_xy)
        cand_xy  = cand_xy[dists > cell_radius_m]

    if len(cand_xy) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int8)

    # ── 4. Exclude cloud-obscured cells (before feature engineering) ─────────
    if cloud_tree is not None:
        dists_cloud, _ = cloud_tree.query(cand_xy)
        cand_xy = cand_xy[dists_cloud > cloud_radius_m]

    if len(cand_xy) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int8)

    # ── 5. Label ──────────────────────────────────────────────────────────────
    labels = np.zeros(len(cand_xy), dtype=np.int8)

    if t2_hotspot_xy is not None and len(t2_hotspot_xy) > 0:
        t2_tree = cKDTree(t2_hotspot_xy)
        dists_t2, _ = t2_tree.query(cand_xy)
        labels[dists_t2 <= cell_radius_m] = 1

    return cand_xy, labels
