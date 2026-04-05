"""
preprocess/__init__.py
----------------------
Unified preprocess() entry point — runs all preprocessing stages in order.
"""

from __future__ import annotations

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.preprocess.hotspots           import preprocess_hotspots
from wildfire_hotspot_prediction.preprocess.clouds             import preprocess_clouds
from wildfire_hotspot_prediction.preprocess.environment        import preprocess_environment
from wildfire_hotspot_prediction.preprocess.fire_weather_index import build_fire_weather_index
from wildfire_hotspot_prediction.preprocess.grid               import build_grid


def preprocess(study: Study) -> None:
    """Run all preprocessing stages in order.

    Stages:
        1. preprocess_hotspots  — clean & snap FIRMS detections to 500 m grid
        2. preprocess_environment — reproject ERA5/terrain/landcover
        3. preprocess_clouds    — match cloud masks to each time step
        4. build_fire_weather_index — FFMC / ISI / ROS
        5. build_grid           — define static 500 m grid with terrain features

    Args:
        study: Study instance.
    """
    print("[preprocess] === stage 1/5: hotspots ===")
    hotspot_data = preprocess_hotspots(study)

    print("[preprocess] === stage 2/5: environment ===")
    preprocess_environment(study)

    print("[preprocess] === stage 3/5: clouds ===")
    preprocess_clouds(study, hotspot_data)

    print("[preprocess] === stage 4/5: fire weather index ===")
    build_fire_weather_index(study)

    print("[preprocess] === stage 5/5: grid ===")
    build_grid(study)

    print("[preprocess] done.")
