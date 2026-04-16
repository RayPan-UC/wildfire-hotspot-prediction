"""
export/render.py
----------------
Export processed parquets → data_render/ GeoJSON files,
then start a local HTTP server and print the URL.

Usage::

    import wildfire_hotspot_prediction as whp
    whp.export_render(study)
    # → exports data_render/
    # → Visualization ready → http://localhost:8765

data_render/ layout
-------------------
meta.json
fire_growth.json
boundaries/<YYYY-MM-DDTHHMM>.geojson   (fire boundary polygon at each T)
pairs/index.json                        (all pairs with T1, T2, fold, label counts)
pairs/<pair_id>/
    receptors.geojson   (B cells: label + key features)
    sources.geojson     (source A hotspot points)
    selector.geojson    (receptor selector polygon)
predictions/<pair_id>/
    predicted.geojson   (future: predicted probability per B cell)
"""

from __future__ import annotations

import functools
import http.server
import json
import logging
import threading
import webbrowser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform as shp_transform

import geopandas as gpd

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.preprocess.hotspots import HotspotData
from wildfire_hotspot_prediction.training.fire_state import (
    build_fire_state, load_fire_state,
)
from wildfire_hotspot_prediction.training.pair_index import build_pair_index
from wildfire_hotspot_prediction.training.receptor_selector import build_receptor_selector
from wildfire_hotspot_prediction.training.builder import _assign_folds

log = logging.getLogger(__name__)

# Repo root → visualize/dist  (render.py is at package/export/render.py)
_REPO_ROOT = Path(__file__).parent.parent.parent
_WEB_DIST  = _REPO_ROOT / "visualize" / "dist"

_PROJ_TO_WGS84 = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _reproject_geom(geom):
    """Reproject a Shapely geometry from EPSG:3978 → WGS84."""
    return shp_transform(_PROJ_TO_WGS84.transform, geom)


def _geom_to_geojson(geom) -> dict:
    """Shapely geometry → GeoJSON dict (WGS84)."""
    return json.loads(gpd.GeoSeries([geom], crs="EPSG:3978")
                      .to_crs("EPSG:4326")[0].__geo_interface__.__class__
                      .__name__)  # placeholder — see below


# Use __geo_interface__ directly
def _geom_to_feature(geom, props: dict = None) -> dict:
    wgs = _reproject_geom(geom)
    return {
        "type": "Feature",
        "geometry": wgs.__geo_interface__,
        "properties": props or {},
    }


def _write_geojson(path: Path, features: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, allow_nan=False), encoding="utf-8")


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str, allow_nan=False), encoding="utf-8")


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_hotspot_data(proc_dir: Path) -> HotspotData:
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


# ── Export functions ──────────────────────────────────────────────────────────

def _export_meta(study: Study, pair_index: pd.DataFrame, n_folds: int, out_dir: Path,
                 fold_models: "dict | None" = None):
    lon_min, lat_min, lon_max, lat_max = study.bbox
    center = [(lon_min + lon_max) / 2, (lat_min + lat_max) / 2]
    thresholds = {}
    if fold_models:
        for fold_k, data in fold_models.items():
            thresholds[str(fold_k)] = data["thresholds"]
    _write_json(out_dir / "meta.json", {
        "study_name":  study.name,
        "bbox":        list(study.bbox),
        "center":      center,
        "start_date":  study.start_date,
        "end_date":    study.end_date,
        "n_folds":     n_folds,
        "n_pairs":     len(pair_index),
        "has_predictions": (study.predictions_dir / "pair_001").exists(),
        "thresholds":  thresholds,
    })
    log.info("[export] meta.json")


def _export_fire_growth(fire_state, out_dir: Path):
    rows = []
    for t in sorted(fire_state.boundary_area_km2.keys()):
        rows.append({
            "time":        t.isoformat(),
            "area_km2":    round(fire_state.boundary_area_km2[t], 3),
            "perimeter_m": round(fire_state.boundary_after[t].length, 1)
                           if fire_state.boundary_after.get(t) else 0,
        })
    _write_json(out_dir / "fire_growth.json", rows)
    log.info("[export] fire_growth.json (%d points)", len(rows))


def _export_boundaries(fire_state, out_dir: Path):
    bnd_dir = out_dir / "boundaries"
    for t, geom in fire_state.boundary_after.items():
        if geom is None or geom.is_empty:
            continue
        fname = t.strftime("%Y-%m-%dT%H%M") + ".geojson"
        _write_geojson(bnd_dir / fname, [_geom_to_feature(geom)])
    log.info("[export] boundaries/ (%d files)", len(fire_state.boundary_after))


def _load_fold_models(models_dir: Path, n_folds: int) -> dict:
    """Load per-fold (xgb, rf, lr) models + thresholds.

    Returns dict[fold_k → {'models': {name → model}, 'thresholds': {name → thr}}].
    Missing files are silently skipped (predictions for that fold fall back to
    label coloring in the UI).
    """
    import pickle
    out = {}
    for k in range(1, n_folds + 1):
        thr_path = models_dir / f"model_fold_{k}_thresholds.json"
        if not thr_path.exists():
            continue
        thresholds = json.loads(thr_path.read_text())
        models = {}
        for name in ("xgb", "rf", "lr"):
            pkl = models_dir / f"model_fold_{k}_{name}.pkl"
            if pkl.exists():
                with open(pkl, "rb") as f:
                    models[name] = pickle.load(f)
        if models:
            out[k] = {"models": models, "thresholds": thresholds}
    return out


def _export_pairs(
    pair_index:   pd.DataFrame,
    fire_state,
    era5:         pd.DataFrame,
    training_dir: Path,
    out_dir:      Path,
    sel_map:      "dict | None" = None,
    fold_models:  "dict | None" = None,
):
    # Load all fold parquets → dict[pair_id → df]
    pair_dfs: dict[str, pd.DataFrame] = {}
    for fold_dir in sorted(training_dir.glob("fold_*")):
        for split in ("train", "test"):
            p = fold_dir / f"{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                for pid, grp in df.groupby("pair_id"):
                    pair_dfs[pid] = grp

    # Feature matrix helper from train module (handles fuel one-hot + NaN fill)
    from wildfire_hotspot_prediction.model.train import _prepare_X as _train_prepare_X

    pairs_dir = out_dir / "pairs"

    # Build index
    index_rows = []

    for _, row in pair_index.iterrows():
        pid  = row["pair_id"]
        t1   = row["T1"]
        t2   = row["T2"]
        fold = int(row["fold"]) if "fold" in row else None
        pair_dir = pairs_dir / str(pid)

        # ── receptors.geojson ─────────────────────────────────────────────────
        if pid in pair_dfs:
            df = pair_dfs[pid]
            features = []
            lons, lats = _PROJ_TO_WGS84.transform(
                df["b_x"].values, df["b_y"].values
            )
            label_counts = df["label"].value_counts().to_dict()

            # Per-model predicted probability (None if fold's models unavailable)
            probs = {"xgb": None, "rf": None, "lr": None}
            if fold_models and fold is not None and fold in fold_models:
                fm = fold_models[fold]["models"]
                X  = _train_prepare_X(df.copy())
                for name, mdl in fm.items():
                    try:
                        probs[name] = mdl.predict_proba(X)[:, 1]
                    except Exception as e:
                        log.warning("[export] %s predict failed on pair %s: %s",
                                    name, pid, e)

            for i in range(len(df)):
                r = df.iloc[i]
                props = {
                    # label
                    "label":               int(r["label"]),
                    # distance
                    "dist_to_fire_front":  _safe_float(r.get("dist_to_fire_front")),
                    # weather
                    "wind_speed":          _safe_float(r.get("wind_speed")),
                    "temp_c":              _safe_float(r.get("temp_c")),
                    "rh":                  _safe_float(r.get("rh")),
                    # FWI
                    "ros":                 _safe_float(r.get("ros")),
                    "ffmc":                _safe_float(r.get("ffmc")),
                    "isi":                 _safe_float(r.get("isi")),
                    # static
                    "slope":               _safe_float(r.get("slope")),
                    "aspect":              _safe_float(r.get("aspect")),
                    "fuel_type":           int(r["fuel_type"]) if pd.notna(r.get("fuel_type")) else None,
                    # path (A→B)
                    "wind_alignment_mean": _safe_float(r.get("wind_alignment_mean")),
                    "wind_alignment_max":  _safe_float(r.get("wind_alignment_max")),
                    "wind_speed_mean":     _safe_float(r.get("wind_speed_mean")),
                    "grade":               _safe_float(r.get("grade")),
                    "slope_mean":          _safe_float(r.get("slope_mean")),
                }
                # Per-model predicted probability
                for name, arr in probs.items():
                    if arr is not None:
                        props[f"prob_{name}"] = _safe_float(arr[i])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lons[i]), float(lats[i])],
                    },
                    "properties": props,
                })
            _write_geojson(pair_dir / "receptors.geojson", features)
        else:
            label_counts = {}

        # ── sources.geojson ───────────────────────────────────────────────────
        cluster_meta = fire_state.step_cluster_meta.get(t1, [])
        if cluster_meta:
            main = max(cluster_meta, key=lambda c: c["count"])
            cxy  = main["cxy"]
            frp  = main["frp"]
            lons_s, lats_s = _PROJ_TO_WGS84.transform(cxy[:, 0], cxy[:, 1])
            src_features = [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lons_s[i]), float(lats_s[i])]},
                    "properties": {"frp": float(frp[i])},
                }
                for i in range(len(cxy))
            ]
            _write_geojson(pair_dir / "sources.geojson", src_features)

        # ── selector.geojson ──────────────────────────────────────────────────
        # Use cached selector from selectors.parquet if available;
        # fall back to recomputing from ERA5 wind.
        if sel_map is not None:
            selector = sel_map.get(pid)
        else:
            selector = build_receptor_selector(t1, fire_state)
        if selector is not None:
            _write_geojson(pair_dir / "selector.geojson",
                           [_geom_to_feature(selector)])

        index_rows.append({
            "pair_id":     pid,
            "T1":          t1.isoformat(),
            "T2":          t2.isoformat(),
            "delta_t_h":   float(row["delta_t_h"]),
            "fold":        fold,
            "n_burned":    int(label_counts.get(1, 0)),
            "n_unburned":  int(label_counts.get(0, 0)),
            "n_cloud":     int(label_counts.get(2, 0)),
        })

    _write_json(pairs_dir / "index.json", index_rows)
    log.info("[export] pairs/ (%d pairs)", len(index_rows))


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f or f == float("inf") or f == float("-inf")) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ── HTTP server ───────────────────────────────────────────────────────────────

def _make_handler(web_dir: Path, data_dir: Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = path.split("?", 1)[0].split("#", 1)[0]
            if path.startswith("/data/"):
                rel = path[6:].lstrip("/")
                target = data_dir / rel if rel else data_dir
            else:
                rel    = path.lstrip("/")
                target = web_dir / rel if rel else web_dir / "index.html"
                if target.is_dir():
                    target = target / "index.html"
            return str(target)

        def log_message(self, fmt, *args):  # suppress access logs
            pass

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    return Handler


def _serve(web_dir: Path, data_dir: Path, port: int):
    Handler = _make_handler(web_dir, data_dir)
    with http.server.HTTPServer(("", port), Handler) as httpd:
        httpd.serve_forever()


# ── Public API ────────────────────────────────────────────────────────────────

def export_render(study: Study, n_folds: int = 3, port: int = 8765) -> None:
    """Export training data to data_render/ and launch the visualization server.

    Exports GeoJSON files from preprocessed parquets, then starts a local
    HTTP server and prints the URL. Opens the browser automatically.

    Args:
        study:   Study instance.
        n_folds: Number of folds (must match build_training_data). Defaults to 3.
        port:    Local server port. Defaults to 8765.
    """
    out_dir  = study.data_render_dir
    proc_dir = study.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[export_render] starting visualization export ...")
    # ── Check web assets ──────────────────────────────────────────────────────
    if not (_WEB_DIST / "index.html").exists():
        print(
            "\n  [export] Web assets not found. Run once:\n"
            f"    cd {_REPO_ROOT / 'visualize'}\n"
            "    npm install && npm run build\n"
        )
        return

    # ── Load data — use cached intermediates when available ───────────────────
    print("[export] loading data ...")
    train_dir = proc_dir / "training"

    hotspot_data = _load_hotspot_data(proc_dir)
    era5         = pd.read_parquet(proc_dir / "weather" / "era5.parquet")

    # fire_state: load from pkl if stage 2 has already run
    fs_pkl = train_dir / "fire_state.pkl"
    if fs_pkl.exists():
        print("[export] loading fire_state.pkl ...")
        fire_state = load_fire_state(fs_pkl)
    else:
        print("[export] building fire_state (pkl not found) ...")
        fire_state = build_fire_state(hotspot_data)

    # pair_index: load from parquet if stage 1 has already run
    pi_path = train_dir / "pair_index.parquet"
    if pi_path.exists():
        print("[export] loading pair_index.parquet ...")
        pair_index = pd.read_parquet(pi_path)
    else:
        print("[export] building pair_index (parquet not found) ...")
        pair_index = build_pair_index(hotspot_data)

    fold_series        = _assign_folds(pair_index, n_folds)
    pair_index["fold"] = fold_series.values

    # selectors: load from parquet if stage 3 has already run
    sel_path = train_dir / "selectors.parquet"
    if sel_path.exists():
        print("[export] loading selectors.parquet ...")
        sel_gdf = gpd.read_parquet(sel_path)
        sel_map = {
            int(r["pair_id"]): r.geometry
            for _, r in sel_gdf.iterrows()
            if r.geometry is not None and not r.geometry.is_empty
        }
    else:
        sel_map = None   # _export_pairs will recompute per-pair

    # Fold models (for per-receptor predicted probability). Missing → UI
    # falls back to label coloring.
    fold_models = _load_fold_models(study.models_dir, n_folds)
    if fold_models:
        have = sorted(fold_models.keys())
        names = sorted({n for k in have for n in fold_models[k]["models"]})
        print(f"[export] fold models loaded for folds {have} "
              f"(models: {','.join(names)})")
    else:
        print("[export] no fold models found — predictions will be omitted")

    # ── Export ────────────────────────────────────────────────────────────────
    _export_meta(study, pair_index, n_folds, out_dir, fold_models=fold_models)
    _export_fire_growth(fire_state, out_dir)
    _export_boundaries(fire_state, out_dir)
    _export_pairs(pair_index, fire_state, era5, study.training_dir, out_dir,
                  sel_map=sel_map, fold_models=fold_models)

    print(f"[export] data_render/ → {out_dir}")

    # ── Start server in background thread ─────────────────────────────────────
    url = f"http://localhost:{port}"
    t   = threading.Thread(
        target=_serve,
        args=(_WEB_DIST, out_dir, port),
        daemon=True,
    )
    t.start()

    print(f"\n  Visualization ready → \033[4;36m{url}\033[0m\n")
    webbrowser.open(url)

    # Block until Ctrl+C
    try:
        t.join()
    except KeyboardInterrupt:
        print("\n[export] server stopped.")
