"""
Full pipeline example.

Prerequisites
-------------
1. Install the package in editable mode (once):
       cd wildfire-hotspot-prediction
       pip install -e ".[dev]"

2. Build the web visualizer (once, requires Node.js):
       cd visualize
       npm install
       npm run build
       cd ..

3. Create tests/.env with your API credentials:
       FIRMS_API_KEY=...
       CDS_KEY=...                  # Copernicus Climate Data Store
       EARTHDATA_TOKEN=...          # NASA Earthdata Bearer token

Run
---
    python tests/example_usage.py

Pipeline stages
---------------
collect             → data_raw/
preprocess          → data_processed/
build_training_data → data_processed/training/
                          pair_index.parquet   (stage 1)
                          fire_state.pkl       (stage 2)
                          selectors.parquet    (stage 3)
                          fold_k/train|test.parquet (stage 4)
train               → models/
predict + evaluate  → predictions/
WildfirePredictor     → importable library for wildfire-decision-support
export_render       → data_render/ + http://localhost:8765

export_render notes
-------------------
- Reads pre-built fold parquets; skips data that hasn't been generated yet.
- Loads fire_state.pkl / pair_index.parquet / selectors.parquet if present
  (avoids recomputing).
- Starts a local HTTP server (port 8765) and opens the browser.
- Press Ctrl+C to stop the server.
- Re-run export_render after each pipeline stage to see updated data.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import wildfire_hotspot_prediction as whp


study = whp.define_study(
    name       = "fort_mcmurray_2016",
    bbox       = (-113.2, 55.8, -109.3, 57.6),
    start_date = "2016-05-01",
    end_date   = "2016-05-31",
)

whp.collect(
    study,
    firms_api_key   = os.environ.get("FIRMS_API_KEY"),
    cds_key         = os.environ.get("CDS_KEY"),
    earthdata_token = os.environ.get("EARTHDATA_TOKEN"),
)

whp.preprocess(study)

whp.build_training_data(study, n_folds=3, override_exist=False)

whp.train(study, use_all_data=False, override_exist=False)   # per-fold models for evaluate()

whp.predict(study)
whp.evaluate(study)

whp.train(study, use_all_data=True, override_exist=False)    # full model for WildfirePredictor

# WildfirePredictor is used as a library by wildfire-decision-support:
#
#   from wildfire_hotspot_prediction import WildfirePredictor
#   predictor = WildfirePredictor(models_dir=Path("path/to/models"))
#   out_df = predictor.predict(feature_df, threshold=0.3)   # adds prob + pred cols
#
predictor = whp.WildfirePredictor(study.models_dir)
print(predictor)

whp.export_render(study)
