"""
wildfire_hotspot_prediction
------------------
A library for building wildfire spread prediction datasets and models.

Typical usage::

    import wildfire_hotspot_prediction as wfsm

    study = wfsm.define_study(
        name       = "fort_mcmurray_2016",
        bbox       = (-113.2, 55.8, -109.3, 57.6),
        start_date = "2016-05-01",
        end_date   = "2016-05-31",
    )

    wfsm.collect_hotspots(study)
    wfsm.collect_environment(study)
    wfsm.collect_clouds(study)

    wfsm.preprocess(study)

    wfsm.build_training_data(study)

    wfsm.train(study)          # use_all_data=False → per-fold models
    wfsm.predict(study)
"""

import os
import sys
from pathlib import Path

# Windows terminals default to cp1252 — force utf-8 so Unicode arrows print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["PROJ_DATA"] = str(Path(__file__).parent / "utils" / "proj_data")

from wildfire_hotspot_prediction.study import Study, define_study

from wildfire_hotspot_prediction.collect            import collect
from wildfire_hotspot_prediction.collect.hotspots    import collect_hotspots
from wildfire_hotspot_prediction.collect.environment import collect_environment
from wildfire_hotspot_prediction.collect.clouds      import collect_clouds

from wildfire_hotspot_prediction.preprocess            import preprocess
from wildfire_hotspot_prediction.preprocess.hotspots   import preprocess_hotspots
from wildfire_hotspot_prediction.preprocess.clouds             import preprocess_clouds
from wildfire_hotspot_prediction.preprocess.environment        import preprocess_environment
from wildfire_hotspot_prediction.preprocess.fire_weather_index import build_fire_weather_index
from wildfire_hotspot_prediction.preprocess.grid               import build_grid

from wildfire_hotspot_prediction.training.pair_index import build_pair_index
from wildfire_hotspot_prediction.training.fire_state import build_fire_state
from wildfire_hotspot_prediction.training.builder    import build_training_data

from wildfire_hotspot_prediction.model.train     import train
from wildfire_hotspot_prediction.model.evaluate  import evaluate
from wildfire_hotspot_prediction.predict.predict import predict

from wildfire_hotspot_prediction.export.render import export_render

from wildfire_hotspot_prediction.build_prediction_data import WildfirePredictor
from wildfire_hotspot_prediction.build_prediction_data.feature_builder import build_prediction_features

from wildfire_hotspot_prediction.pipeline import run_pipeline

__all__ = [
    "Study", "define_study",
    "collect", "collect_hotspots", "collect_environment", "collect_clouds",
    "preprocess",
    "preprocess_hotspots", "preprocess_clouds", "preprocess_environment",
    "build_fire_weather_index",
    "build_grid",
    "build_pair_index", "build_fire_state", "build_training_data",
    "train", "predict", "evaluate",
    "export_render",
    "WildfirePredictor",
    "build_prediction_features",
    "run_pipeline",
]
