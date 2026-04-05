"""
model/train.py
--------------
Train wildfire spread prediction models from k-fold parquets.

Three models are trained per fold:
    xgb  XGBClassifier   (primary)
    rf   RandomForestClassifier
    lr   LogisticRegression  (with StandardScaler pipeline)


Outputs:
    models/model_fold_<k>_<name>.pkl   (use_all_data=False)
    models/model_full_<name>.pkl       (use_all_data=True)
    models/feature_cols.json
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wildfire_hotspot_prediction.study import Study

log = logging.getLogger(__name__)

MODELS = ("xgb", "rf", "lr")

# All FBP fuel type codes (101–122) — full Canada FBP catalogue
# Mirrors the keys of _FBP_PARAMS in preprocess/fire_weather_index.py
_FUEL_TYPE_CODES = [
    101, 102, 103, 104, 105, 106, 107,   # C-1 … C-7
    108, 109,                             # D-1, D-2
    110, 111, 112, 113,                   # M-1 … M-4
    114, 115,                             # O-1a, O-1b
    116, 117, 118,                        # S-1 … S-3
    119, 120, 121, 122,                   # Non-fuel, Water, Urban, Unknown
]

_FUEL_DUMMY_COLS: list[str] = [f"fuel_type_{c}" for c in _FUEL_TYPE_CODES]

# Base feature columns (no fuel_type — replaced by dummies above)
_BASE_FEATURE_COLS = [
    # distance A→B
    "dist", "log_dist",
    # source FRP
    "frp_A",
    # static
    "dtm", "slope", "aspect",
    # weather
    "temp_c", "rh", "wind_speed", "wind_dir",
    # FWI
    "ffmc", "isi", "ros",
    # path (A→B)
    "grade", "slope_mean", "slope_std",
    "wind_speed_mean", "wind_alignment_mean",
    "wind_alignment_max", "wind_align_product",
    # interaction
    "frp_x_wind",
    # cluster
    "cluster_frp_sum", "cluster_n_pts", "cluster_hull_area", "cluster_density",
    # fire geometry
    "fire_age_h", "perimeter_m", "compactness",
    "growth_rate_km2h", "frp_per_area_km2", "new_area_km2",
    # distance to fire front
    "dist_to_fire_front",
    # pair metadata
    "delta_t_h",
]

# Full feature list saved to feature_cols.json and used by predict()
FEATURE_COLS: list[str] = _BASE_FEATURE_COLS + _FUEL_DUMMY_COLS


def _add_fuel_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode fuel_type using the canonical FBP code list."""
    dummies = pd.get_dummies(df["fuel_type"], prefix="fuel_type", dtype=np.float32)
    for c in _FUEL_DUMMY_COLS:
        if c not in dummies:
            dummies[c] = np.float32(0.0)
    dummies = dummies[_FUEL_DUMMY_COLS]
    return df.drop(columns=["fuel_type"]).join(dummies)


def _load_fold(fold_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(fold_dir / f"{split}.parquet")


def _prepare_X(df: pd.DataFrame) -> np.ndarray:
    """Expand fuel_type dummies and return feature matrix aligned to FEATURE_COLS.

    NaN values are filled with the column median (computed from the same batch),
    so all models receive a finite float32 matrix.
    """
    if "fuel_type" in df.columns:
        df = _add_fuel_dummies(df)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.float32(0.0)
    X = df[FEATURE_COLS].values.astype(np.float32)

    nan_cols = np.where(np.any(np.isnan(X), axis=0))[0]
    if nan_cols.size:
        col_names = [FEATURE_COLS[i] for i in nan_cols]
        log.warning("[train] NaN in %d column(s): %s — filling with median",
                    len(nan_cols), col_names)
        for i in nan_cols:
            col = X[:, i]
            median = float(np.nanmedian(col))
            col[np.isnan(col)] = median if np.isfinite(median) else 0.0

    return X


def _build_models(spw: float) -> dict:
    """Instantiate all three model objects."""
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=1.0,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        xgb = None

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    models = {"rf": rf, "lr": lr}
    if xgb is not None:
        models["xgb"] = xgb
    return models


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J = TPR - FPR, return threshold at the maximum.

    Capped at 0.95 to prevent near-zero recall.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j      = tpr - fpr
    opt    = float(thresholds[np.argmax(j)])
    return min(opt, 0.95)


def _oof_threshold(model, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray,
                   max_rows: int = 200_000) -> float:
    """Compute Youden's J threshold via GroupKFold(5) OOF predictions.

    Groups = pair_id, preventing pair-level leakage across OOF folds.

    To keep CV fast on large datasets, subsamples up to max_rows rows:
    all positive (burned) samples are kept; negatives are randomly sampled.
    Falls back to 0.5 if fewer than 2 positive samples.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict

    n_pos = int((y == 1).sum())
    if n_pos < 2:
        return 0.5

    # Subsample if needed — keep all positives, sample negatives
    if len(y) > max_rows:
        rng     = np.random.default_rng(42)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg   = max(max_rows - len(pos_idx), len(pos_idx))  # at least pos:1 ratio
        n_neg   = min(n_neg, len(neg_idx))
        neg_sel = rng.choice(neg_idx, n_neg, replace=False)
        idx     = np.concatenate([pos_idx, neg_sel])
        rng.shuffle(idx)
        X_s, y_s, g_s = X[idx], y[idx], groups[idx]
        print(f"      OOF subsample: {len(y_s):,} rows  (pos={len(pos_idx):,}  neg={n_neg:,})")
    else:
        X_s, y_s, g_s = X, y, groups

    cv       = GroupKFold(n_splits=min(5, len(np.unique(g_s))))
    oof_prob = cross_val_predict(
        model, X_s, y_s,
        cv=cv, groups=g_s,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return _youden_threshold(y_s, oof_prob)


def _fit_and_save(X: np.ndarray, y: np.ndarray,
                  groups: np.ndarray,
                  models_dir: Path, stem: str) -> None:
    """Train all models, compute OOF Youden threshold, and save to disk."""
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    spw   = round(n_neg / n_pos, 2)

    thresholds: dict[str, float] = {}

    for name, model in _build_models(spw).items():
        print(f"    training {name} ...")
        model.fit(X, y)
        out = models_dir / f"{stem}_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)

        print(f"    computing OOF threshold for {name} ...")
        thr = _oof_threshold(model, X, y, groups)
        thresholds[name] = round(thr, 6)
        print(f"      threshold = {thr:.4f}")
        log.info("[train] saved %s  threshold=%.4f", out.name, thr)

    thr_path = models_dir / f"{stem}_thresholds.json"
    thr_path.write_text(json.dumps(thresholds, indent=2))
    log.info("[train] saved %s", thr_path.name)


def _stem_exists(models_dir: Path, stem: str) -> bool:
    """Return True if all model pkl files for this stem already exist."""
    return all(
        (models_dir / f"{stem}_{name}.pkl").exists()
        for name in MODELS
    )


def train(
    study:          Study,
    use_all_data:   bool = False,
    n_folds:        int  = 3,
    override_exist: bool = False,
) -> None:
    """Train XGB, RF and LR models from k-fold training parquets.

    Args:
        study:          Study instance.
        use_all_data:   If True, train on all data combined (model_full_*.pkl).
                        If False, train one set per fold (model_fold_<k>_*.pkl).
        n_folds:        Number of folds. Defaults to 3.
        override_exist: If False (default), skip stems whose pkl files already
                        exist. Set True to force retraining.

    Saves:
        models/model_fold_<k>_xgb.pkl
        models/model_fold_<k>_rf.pkl
        models/model_fold_<k>_lr.pkl
        models/feature_cols.json
    """
    print(f"[train] starting model training  ({len(FEATURE_COLS)} features) ...")
    training_dir = study.data_processed_dir / "training"
    models_dir   = study.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_path = models_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(FEATURE_COLS, indent=2))

    if use_all_data:
        stem = "model_full"
        if not override_exist and _stem_exists(models_dir, stem):
            print(f"  [skip] {stem} already exists")
        else:
            dfs = []
            for k in range(1, n_folds + 1):
                fold_dir = training_dir / f"fold_{k}"
                for split in ("train", "test"):
                    try:
                        dfs.append(_load_fold(fold_dir, split))
                    except FileNotFoundError:
                        pass
            if not dfs:
                log.error("[train] no training data found")
                return
            df = pd.concat(dfs, ignore_index=True)
            X      = _prepare_X(df)
            y      = df["label"].values.astype(np.int8)
            groups = df["pair_id"].values
            print(f"  full dataset: {len(df):,} rows")
            _fit_and_save(X, y, groups, models_dir, stem)

    else:
        for k in range(1, n_folds + 1):
            stem = f"model_fold_{k}"
            if not override_exist and _stem_exists(models_dir, stem):
                print(f"  [skip] {stem} already exists")
                continue
            fold_dir = training_dir / f"fold_{k}"
            try:
                df = _load_fold(fold_dir, "train")
            except FileNotFoundError:
                log.warning("[train] fold_%d train.parquet not found — skipping", k)
                continue
            X      = _prepare_X(df)
            y      = df["label"].values.astype(np.int8)
            groups = df["pair_id"].values
            print(f"  fold_{k}: {len(df):,} rows  (burned={int((y==1).sum()):,}  unburned={int((y==0).sum()):,})")
            _fit_and_save(X, y, groups, models_dir, stem)

    print(f"[train] done  ->  {models_dir}")
