"""
predict/predict.py
------------------
Apply trained per-fold models to their held-out test sets.

Reads:
    models/feature_cols.json
    models/model_fold_<k>_<name>.pkl   (xgb / rf / lr)
    data_processed/training/fold_<k>/test.parquet

Outputs:
    predictions/fold_<k>/<name>_predictions.parquet
        cols: pair_id, b_grid_id, b_x, b_y, label, prob, pred
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_hotspot_prediction.study import Study

log = logging.getLogger(__name__)

_MODEL_NAMES = ("xgb", "rf", "lr")


def _load_thresholds(models_dir: Path, stem: str) -> dict[str, float]:
    """Load per-model thresholds saved by train(). Falls back to 0.5."""
    path = models_dir / f"{stem}_thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def predict(
    study:   Study,
    n_folds: int = 3,
) -> Path:
    """Apply all trained per-fold models to their held-out test sets.

    Args:
        study:   Study instance.
        n_folds: Number of folds. Defaults to 3.

    Returns:
        Path to predictions/ directory.
    """
    print("[predict] starting prediction ...")
    models_dir   = study.models_dir
    training_dir = study.data_processed_dir / "training"
    preds_dir    = study.predictions_dir
    preds_dir.mkdir(parents=True, exist_ok=True)

    feat_path = models_dir / "feature_cols.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found — run train() first.")
    feature_cols = json.loads(feat_path.read_text())

    for k in range(1, n_folds + 1):
        test_path = training_dir / f"fold_{k}" / "test.parquet"
        if not test_path.exists():
            log.warning("[predict] fold_%d/test.parquet not found — skipping", k)
            continue

        df = pd.read_parquet(test_path)
        if df.empty:
            continue

        # Expand fuel_type dummies aligned to feature_cols
        if "fuel_type" in df.columns:
            fuel_dummy_cols = [c for c in feature_cols if c.startswith("fuel_type_")]
            dummies = pd.get_dummies(df["fuel_type"], prefix="fuel_type", dtype=np.float32)
            for c in fuel_dummy_cols:
                if c not in dummies:
                    dummies[c] = np.float32(0.0)
            df = df.drop(columns=["fuel_type"]).join(dummies[fuel_dummy_cols])

        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.float32(0.0)

        X = df[feature_cols].values.astype(np.float32)
        for i in np.where(np.any(np.isnan(X), axis=0))[0]:
            col = X[:, i]
            median = float(np.nanmedian(col))
            col[np.isnan(col)] = median if np.isfinite(median) else 0.0

        fold_pred_dir = preds_dir / f"fold_{k}"
        fold_pred_dir.mkdir(parents=True, exist_ok=True)

        thresholds = _load_thresholds(models_dir, f"model_fold_{k}")

        for name in _MODEL_NAMES:
            model_path = models_dir / f"model_fold_{k}_{name}.pkl"
            if not model_path.exists():
                log.warning("[predict] %s not found — skipping", model_path.name)
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            prob      = model.predict_proba(X)[:, 1].astype(np.float32)
            threshold = thresholds.get(name, 0.5)
            pred      = (prob >= threshold).astype(np.int8)
            log.info("[predict] fold_%d %s  threshold=%.4f", k, name, threshold)

            out_df = pd.DataFrame({
                "pair_id":   df["pair_id"].values,
                "b_grid_id": df["b_grid_id"].values,
                "b_x":       df["b_x"].values,
                "b_y":       df["b_y"].values,
                "label":     df["label"].values,
                "prob":      prob,
                "pred":      pred,
            })
            out_df.to_parquet(fold_pred_dir / f"{name}_predictions.parquet", index=False)

        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        print(f"  fold_{k}  {len(df):,} rows  (burned={n_pos:,}  unburned={n_neg:,})")

    print(f"[predict] done  ->  {preds_dir}")
    return preds_dir
