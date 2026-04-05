"""
model/evaluate.py
-----------------
Compute classification metrics from per-fold predictions for all models.

Reads:
    predictions/fold_<k>/<name>_predictions.parquet

Outputs:
    predictions/metrics.json
    — prints a per-fold × per-model summary table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_hotspot_prediction.study import Study

log = logging.getLogger(__name__)

_MODEL_NAMES = ("xgb", "rf", "lr")


def evaluate(
    study:   Study,
    n_folds: int = 3,
) -> dict:
    """Compute AUC-ROC, F1, precision and recall for all models and folds.

    Args:
        study:   Study instance.
        n_folds: Number of folds. Defaults to 3.

    Returns:
        Nested dict: {model_name: {fold_k: metrics, overall: metrics}}.
    """
    print("[evaluate] computing metrics ...")
    preds_dir = study.predictions_dir

    result = {}

    for name in _MODEL_NAMES:
        fold_rows = []
        all_dfs   = []

        for k in range(1, n_folds + 1):
            path = preds_dir / f"fold_{k}" / f"{name}_predictions.parquet"
            if not path.exists():
                continue

            df = pd.read_parquet(path)
            all_dfs.append(df)

            m = _metrics(df["label"].values, df["prob"].values, df["pred"].values)
            m.update({
                "fold":       k,
                "n_burned":   int((df["label"] == 1).sum()),
                "n_unburned": int((df["label"] == 0).sum()),
            })
            fold_rows.append(m)

        if not fold_rows:
            continue

        all_df  = pd.concat(all_dfs, ignore_index=True)
        overall = _metrics(all_df["label"].values, all_df["prob"].values, all_df["pred"].values)
        overall.update({
            "fold":       "all",
            "n_burned":   int((all_df["label"] == 1).sum()),
            "n_unburned": int((all_df["label"] == 0).sum()),
        })

        result[name] = {"folds": fold_rows, "overall": overall}

    if not result:
        print("[evaluate] no predictions found — run predict() first")
        return {}

    out_path = preds_dir / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    _print_table(result, n_folds)
    print(f"[evaluate] metrics saved  ->  {out_path}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
    )
    has_both = len(np.unique(y_true)) > 1
    auc    = float(roc_auc_score(y_true, y_prob))         if has_both else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else float("nan")
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "auc":    round(auc,    4),
        "pr_auc": round(pr_auc, 4),
        "f1":     round(f1,     4),
        "precision": round(pre, 4),
        "recall":    round(rec, 4),
    }


def _print_table(result: dict, n_folds: int):
    col_w = 56
    sep   = "-" * col_w
    hdr   = f"  {'fold':>4}  {'AUC':>6}  {'PR-AUC':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}"

    for name, data in result.items():
        print(f"\n  [{name.upper()}]")
        print(sep)
        print(hdr)
        print(sep)
        for m in data["folds"]:
            print(f"  {str(m['fold']):>4}  {m['auc']:>6.4f}  {m['pr_auc']:>6.4f}"
                  f"  {m['f1']:>6.4f}  {m['precision']:>6.4f}  {m['recall']:>6.4f}")
        o = data["overall"]
        print(sep)
        print(f"  {'all':>4}  {o['auc']:>6.4f}  {o['pr_auc']:>6.4f}"
              f"  {o['f1']:>6.4f}  {o['precision']:>6.4f}  {o['recall']:>6.4f}")
        print(sep)
