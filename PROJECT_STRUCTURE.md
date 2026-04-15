# Wildfire Hotspot Prediction — Project Structure

## Overview

End-to-end machine learning pipeline for predicting wildfire spread at 500m grid resolution. Integrates VIIRS satellite hotspot detections, ERA5 weather reanalysis, terrain/fuel type rasters, and Canadian FBP fire weather indices to train binary classifiers that predict whether unburned cells will ignite within a given time window.

**Core pipeline**: Collect → Preprocess → Build Training Data → Train → Predict → Evaluate → Visualize

---

## Directory Structure

```
wildfire-hotspot-prediction/
├── pyproject.toml                          # Package metadata & dependencies
├── requirements.txt
├── models/                                 # Pre-trained model artifacts (Zenodo)
│   ├── model_full_xgb.pkl                  #   XGBoost classifier
│   ├── model_full_rf.pkl                   #   Random Forest classifier
│   ├── model_full_lr.pkl                   #   Logistic Regression classifier
│   └── model_full_thresholds.json          #   Optimal decision thresholds
│
├── wildfire_hotspot_prediction/            # Main Python package
│   ├── __init__.py                         # Public API & env setup (PROJ_DATA, UTF-8)
│   ├── study.py                            # Study dataclass — bbox, dates, directory layout
│   ├── pipeline.py                         # run_pipeline() orchestrator (stub)
│   ├── models.py                           # Download pre-trained models from Zenodo
│   │
│   ├── collect/                            # Stage 1: Raw data acquisition
│   │   ├── __init__.py                     #   collect() — orchestrates all downloads
│   │   ├── hotspots.py                     #   VIIRS hotspots from NASA FIRMS API
│   │   ├── environment.py                  #   ERA5 weather, MRDEM terrain, NRCan fuel type
│   │   └── clouds.py                       #   VIIRS cloud masks via NASA CMR
│   │
│   ├── preprocess/                         # Stage 2: Clean, reproject, derive indices
│   │   ├── __init__.py                     #   preprocess() — chains all substages
│   │   ├── hotspots.py                     #   Snap to grid, group overpass events
│   │   ├── environment.py                  #   Reproject rasters, parse ERA5 → parquet
│   │   ├── clouds.py                       #   Match cloud granules to overpass timestamps
│   │   ├── grid.py                         #   Build 500m grid with static features
│   │   └── fire_weather_index.py           #   Compute FFMC, ISI, ROS (Canadian FBP system)
│   │
│   ├── training/                           # Stage 3: Training dataset construction
│   │   ├── __init__.py
│   │   ├── pair_index.py                   #   Build (T1, T2) overpass pair timeline
│   │   ├── fire_state.py                   #   Accumulate fire boundary at each timestep
│   │   ├── receptor_selector.py            #   Define 20km candidate zone around fire front
│   │   ├── sampling.py                     #   Sample source A hotspots & receptor B cells
│   │   ├── features.py                     #   Join static/weather/FWI features to B cells
│   │   ├── sampling_path.py                #   Path-based features (grade, wind alignment)
│   │   └── builder.py                      #   Main orchestrator → k-fold parquets
│   │
│   ├── model/                              # Stage 4: Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py                        #   Train XGBoost / RF / LR per fold or full
│   │   └── evaluate.py                     #   AUC-ROC, PR-AUC, F1 metrics
│   │
│   ├── predict/                            # Stage 5: Inference on held-out test sets
│   │   ├── __init__.py
│   │   └── predict.py                      #   Apply per-fold models to test parquets
│   │
│   ├── build_prediction_data/              # Operational inference (real-time prediction)
│   │   ├── __init__.py                     #   run_prediction_pipeline() wrapper
│   │   ├── feature_builder.py              #   Build features for arbitrary (T1, T1+Δt)
│   │   ├── predictor.py                    #   WildfirePredictor class — importable library
│   │   └── era5_check.py                   #   Check ERA5 availability for target timestamp
│   │
│   ├── export/                             # Stage 6: Export to GeoJSON + HTTP server
│   │   ├── __init__.py
│   │   └── render.py                       #   Convert parquets → GeoJSON, serve on :8765
│   │
│   └── utils/                              # Shared utilities
│       ├── __init__.py
│       ├── geo.py                          #   Grid ID snap/decode, geometry smoothing
│       ├── raster.py                       #   RasterSampler — fast in-memory raster lookup
│       └── proj_data/                      #   Bundled PROJ datum files (Windows fix)
│
├── tests/
│   ├── conftest.py                         # Pytest config, loads .env credentials
│   ├── example_usage.py                    # Full pipeline walkthrough (single script)
│   └── example_usage_steps.py              # Step-by-step pipeline example
│
└── visualize/                              # Frontend: interactive map visualization
    ├── package.json                        #   deck.gl 9 + MapLibre GL 4 + Vite 5
    ├── vite.config.js                      #   Dev proxy /data/ → :8765; two HTML entries
    ├── index.html                          #   Main map page (sidebar + deck.gl overlay)
    ├── fire_growth.html                    #   Fire growth chart page
    ├── src/
    │   ├── main.js                         #   Map init, layer orchestration, event handling
    │   ├── data.js                         #   Fetch helpers (meta, pairs, boundaries, etc.)
    │   ├── layers.js                       #   deck.gl layer factories (boundary, receptor, source)
    │   ├── ui.js                           #   Panel controls, fold/pair selectors, cell info
    │   └── growth.js                       #   Fire growth chart rendering
    └── styles/
        ├── main.scss                       #   Main stylesheet
        ├── _variables.scss                 #   SCSS variables
        └── _panel.scss                     #   Sidebar panel styles
```

---

## Data Flow

```
                        ┌─────────────────────────────────────┐
                        │         External Data Sources        │
                        │  NASA FIRMS · ERA5 · MRDEM · NRCan  │
                        │        NASA CMR (cloud masks)        │
                        └───────────────┬─────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   1. collect()     │
                              │   Raw downloads    │
                              └─────────┬─────────┘
                                        │
                    data_raw/firms/hotspots_raw.csv
                    data_raw/weather/era5.nc
                    data_raw/terrain/{dtm,slope,aspect}.tif
                    data_raw/landcover/fuel_type.tif
                    data_raw/clouds/CLDMSK_*.npy
                                        │
                              ┌─────────▼─────────┐
                              │  2. preprocess()   │
                              │  Clean & reproject │
                              └─────────┬─────────┘
                                        │
                    data_processed/firms/hotspots.parquet
                    data_processed/weather/era5.parquet
                    data_processed/weather/{ffmc_daily,isi_hourly,ros_hourly}.parquet
                    data_processed/terrain/{dtm,slope,aspect}.tif  (EPSG:3978)
                    data_processed/landcover/fuel_type.tif         (EPSG:3978)
                    data_processed/clouds/<timestamp>.parquet
                    data_processed/grid_static.parquet
                                        │
                       ┌────────────────▼────────────────┐
                       │  3. build_training_data()       │
                       │  Pair index → fire state →      │
                       │  receptor selection → features   │
                       └────────────────┬────────────────┘
                                        │
                    training/pair_index.parquet
                    training/fire_state.pkl
                    training/selectors.parquet
                    training/fold_k/{train,test}.parquet
                                        │
                  ┌─────────────────────▼──────────────────────┐
                  │                                            │
         ┌────────▼────────┐                        ┌─────────▼─────────┐
         │  4. train()     │                        │  build_prediction  │
         │  3 classifiers  │                        │  _features()       │
         │  per fold + full│                        │  (operational)     │
         └────────┬────────┘                        └─────────┬─────────┘
                  │                                           │
    models/model_{fold_k,full}_{xgb,rf,lr}.pkl     WildfirePredictor
    models/feature_cols.json                        → probability map
    models/model_full_thresholds.json
                  │
         ┌────────▼────────┐
         │  5. predict()   │
         │  Test inference  │
         └────────┬────────┘
                  │
    predictions/fold_k/<model>_predictions.parquet
                  │
         ┌────────▼────────┐
         │  6. evaluate()  │
         │  Metrics & F1   │
         └────────┬────────┘
                  │
    predictions/metrics.json
                  │
         ┌────────▼────────┐
         │ 7. export_render │
         │  GeoJSON + HTTP  │
         └────────┬────────┘
                  │
    data_render/{meta,fire_growth}.json
    data_render/boundaries/*.geojson
    data_render/pairs/<id>/{receptors,sources,selector}.geojson
                  │
         ┌────────▼────────┐
         │  8. visualize/  │
         │  deck.gl + map  │
         └─────────────────┘
```

---

## Module Details

### Study (`study.py`)

`define_study()` creates a `Study` dataclass that defines:
- **name** — project identifier (e.g. `"fort_mcmurray_2016"`)
- **bbox** — `(lon_min, lat_min, lon_max, lat_max)` in WGS84
- **start_date / end_date** — event date range (ISO strings)
- **project_dir** — root for all outputs; auto-creates subdirectory tree

All downstream modules receive a `Study` instance to locate input/output paths.

### Collect (`collect/`)

| Module | Data Source | Output |
|---|---|---|
| `hotspots.py` | NASA FIRMS API (VIIRS SP/NRT) | `hotspots_raw.csv` |
| `environment.py` | ERA5-Land (CDS API), MRDEM-30 (AWS COG), NRCan FBP fuel type | `era5.nc`, `dtm/slope/aspect.tif`, `fuel_type.tif` |
| `clouds.py` | NASA CMR → CLDMSK_L2_VIIRS_SNPP HDF5 | `CLDMSK_*.npy` (cloudy pixel coords) |

### Preprocess (`preprocess/`)

| Module | Function | Key Logic |
|---|---|---|
| `hotspots.py` | `preprocess_hotspots()` | Reproject to EPSG:3978, group detections into overpass events (>10 min gap) |
| `environment.py` | `_preprocess_era5()` | K→°C, compute wind speed/dir/RH, snap to 500m grid IDs via KDTree |
| `clouds.py` | `preprocess_clouds()` | Match `.npy` granules to overpass timestamps (±10 min) |
| `grid.py` | `build_grid()` | 500m meshgrid over bbox, sample DTM/slope/aspect/fuel_type per cell |
| `fire_weather_index.py` | `build_fire_weather_index()` | FFMC (daily accumulation), ISI, ROS per fuel type (Canadian FBP) |

### Training Data (`training/`)

Four stages executed by `build_training_data()`:

1. **Pair Index** (`pair_index.py`) — enumerate all valid (T1, T2) overpass pairs within configurable time range
2. **Fire State** (`fire_state.py`) — DBSCAN clustering + forward-pass boundary accumulation at each timestep
3. **Receptor Selector** (`receptor_selector.py`) — morphological smooth + 20km buffer donut around fire front
4. **Sampling + Features** (`builder.py` → `sampling.py`, `features.py`, `sampling_path.py`):
   - Sample source A hotspots (largest cluster at T1)
   - Sample receptor B cells (500m grid within selector, exclude burning/cloudy)
   - Label: 1 if T2 hotspot within 353.6m, else 0 (cloud → -1)
   - Join features: static (terrain, fuel), weather, FWI, fire geometry, path-based
   - Temporal k-fold split → `fold_k/{train,test}.parquet`

### Feature Set

| Category | Features |
|---|---|
| **Static** | dtm, slope, aspect, fuel_type (one-hot 101–122) |
| **Weather** | temp_c, rh, wind_speed, wind_dir |
| **FWI** | ffmc, isi, ros |
| **Fire geometry** | fire_age_h, perimeter_m, compactness, growth_rate_km2h, frp_per_area_km2, new_area_km2 |
| **Spatial** | dist_to_fire_front, ab_dist_m, delta_t_h |
| **Path** | grade, slope_mean, slope_std, wind_speed_mean, wind_alignment_mean/max, wind_align_product |
| **Derived** | frp_x_wind, cluster_count, cluster_frp_sum, etc. |

### Model (`model/`)

Three classifiers trained per temporal k-fold:
- **XGBoost** (primary) — `XGBClassifier`
- **Random Forest** — `RandomForestClassifier`
- **Logistic Regression** — `LogisticRegression` with `StandardScaler`

`train(use_all_data=True)` produces full-data models for operational inference.

### Operational Inference (`build_prediction_data/`)

`WildfirePredictor` loads a full model and provides:
- `predict_proba(df)` → probability array
- `predict(df, threshold)` → DataFrame with `prob` and `pred` columns

`build_prediction_features()` constructs a feature DataFrame for any arbitrary (T1, T1+Δt) pair, reusing the same pipeline logic as training.

### Visualization (`visualize/`)

Interactive web map built with **deck.gl 9** + **MapLibre GL 4**, bundled by **Vite 5**.

- `export_render()` converts parquets → GeoJSON and starts an HTTP server on port 8765
- Vite dev server proxies `/data/` requests to the Python server
- **Layers**: fire boundary (orange), receptor selector (yellow), receptor cells (red=burned / blue=unburned / gray=cloud), source hotspots (orange)
- **Panels**: fold/pair selectors, layer toggles, pair statistics, cell info on click

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| 500m grid resolution | Matches VIIRS sensor pixel size |
| EPSG:3978 (NAD83 / Canada Atlas Lambert) | Equal-area projection for fair distance/area calculations |
| (T1, T2) pair framework | Models fire spread over variable time windows (3–24h) |
| Forward-pass fire state | Guarantees correct boundary at any skip-pair without re-computation |
| FFMC/ISI/ROS from FBP | Physics-informed features (Canadian Forest Fire Behavior Prediction system) |
| ROS-based wind interpolation along paths | Estimates fire arrival time at each sample point for physically-grounded wind features |
| Cloud exclusion (not labeling) | Cloud-obscured cells cannot be reliably labeled; excluded before feature engineering |
| Temporal k-fold | Splits on T1 timestamps to prevent temporal data leakage |

---

## Dependencies

**Python** (≥3.10): numpy, pandas, geopandas, shapely, pyproj, rasterio, scipy, scikit-learn, xgboost, xarray, netCDF4, pyarrow, requests, h5py, tqdm, cdsapi

**Frontend**: deck.gl 9, maplibre-gl 4, vite 5, sass
