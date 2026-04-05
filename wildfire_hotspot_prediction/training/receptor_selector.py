"""
training/receptor_selector.py
------------------------------
Build the receptor selector polygon (AOI ring) for a given T1.

Formula
-------
    smooth   = boundary_after[T1].buffer(+smooth_m).buffer(-smooth_m)
    selector = smooth.buffer(spread_m) − smooth

The morphological open/close (buffer+, buffer−) fills interior holes and
removes narrow protrusions from the accumulated fire boundary.
The outer ring (spread_m = 20 km) defines the candidate zone where
receptors are sampled — already-burned area is excluded by construction.
"""

from __future__ import annotations

import pandas as pd

from wildfire_hotspot_prediction.training.fire_state import FireState


def build_receptor_selector(
    t1:       pd.Timestamp,
    fire_state: FireState,
    smooth_m: float = 5_000.0,
    spread_m: float = 20_000.0,
):
    """Compute the receptor selector (AOI ring) for a given T1.

    Args:
        t1:        Source overpass timestamp.
        fire_state: FireState from build_fire_state.
        smooth_m:  Morphological smoothing radius [m]. Defaults to 5000.
        spread_m:  Outer ring buffer radius [m]. Defaults to 20000.

    Returns:
        Shapely geometry (the 20 km donut ring) or None if no boundary at T1.
    """
    boundary = fire_state.boundary_after.get(t1)
    if boundary is None or boundary.is_empty:
        return None

    # Morphological smooth: fill holes, remove narrow protrusions
    smooth = boundary.buffer(smooth_m).buffer(-smooth_m)
    if smooth is None or smooth.is_empty:
        smooth = boundary   # fallback if smoothing collapses small geometry

    # 20 km donut ring: outer buffer minus smooth interior
    selector = smooth.buffer(spread_m).difference(smooth)
    if selector.is_empty:
        return None

    return selector
