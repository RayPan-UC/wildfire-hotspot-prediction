"""
training/pair_index.py
----------------------
Build the timeline pair index from preprocessed hotspot time steps.

A pair (T1 → T2) is valid when:
  - delta_t is within [min_time_range, max_time_range]

Pair types:
  - n_steps=1: consecutive adjacent revisit; used for train AND test
  - n_steps>1: skip pairs (N hops ahead); training only

Output schema:
    pair_id    int       unique pair index
    T1         Timestamp source revisit timestamp
    T2         Timestamp receptor revisit timestamp
    n_steps    int       number of hops (1 = consecutive, 2 = skip1, ...)
    delta_t_h  float     time difference T2 - T1 in hours
    train_only bool      True for skip pairs
"""

from __future__ import annotations

import pandas as pd

from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData


def build_pair_index(
    hotspot_data:   HotspotData,
    max_steps:      int   = 1,
    max_time_range: float = 24.0,
    min_time_range: float = 0.25,
) -> pd.DataFrame:
    """Build all valid (T1, T2) pairs from the hotspot overpass times.

    For each overpass T1, looks ahead up to max_steps hops to find T2
    candidates within the allowed time window.

    Args:
        hotspot_data:   Output of preprocess_hotspots.
        max_steps:      Maximum number of hops between T1 and T2.
                        1 = consecutive only, 2 = consecutive + skip1, etc.
        max_time_range: Maximum allowed T2 - T1 gap in hours. Defaults to 24.
        min_time_range: Minimum allowed T2 - T1 gap in hours. Defaults to 0.25.

    Returns:
        DataFrame with cols: pair_id, T1, T2, n_steps, delta_t_h, train_only.
        Empty DataFrame if fewer than 2 overpass times.
    """
    times = hotspot_data.overpass_times   # already sorted

    if len(times) < 2:
        return pd.DataFrame(columns=["pair_id", "T1", "T2", "n_steps",
                                      "delta_t_h", "train_only"])

    rows = []
    pair_id = 0

    for i, t1 in enumerate(times):
        for step in range(1, max_steps + 1):
            j = i + step
            if j >= len(times):
                break
            t2 = times[j]
            delta_t_h = (t2 - t1).total_seconds() / 3600.0
            if delta_t_h < min_time_range or delta_t_h > max_time_range:
                continue
            rows.append({
                "pair_id":    pair_id,
                "T1":         t1,
                "T2":         t2,
                "n_steps":    step,
                "delta_t_h":  round(delta_t_h, 4),
                "train_only": step > 1,
            })
            pair_id += 1

    return pd.DataFrame(rows)
