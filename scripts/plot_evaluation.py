"""
plot_evaluation.py
------------------
Generate evaluation charts for wildfire hotspot prediction.

Produces per fold:
  predictions/fold_N/roc_curves.png
  predictions/fold_N/pr_curves.png
  predictions/fold_N/confusion_<model>.png
  predictions/fold_N/feat_imp_<model>.png      (xgb, rf only)
  predictions/fold_N/coef_lr.png
  predictions/fold_N/model_comparison.png
  predictions/fold_N/per_pair_metrics_<model>.png

Produces cross-fold summary:
  predictions/charts/cross_fold_comparison.png
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("C:/Users/Ray Pan/OneDrive/github/fort_mcmurray_2016")
PRED_DIR   = BASE_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
TRAIN_DIR  = BASE_DIR / "data_processed" / "training"
CHART_DIR  = PRED_DIR / "charts"

N_FOLDS = 3
MODELS  = ("xgb", "rf", "lr")
MODEL_LABELS = {"xgb": "XGBoost", "rf": "Random Forest", "lr": "Logistic Regression"}
MODEL_COLORS = {"xgb": "#e74c3c", "rf": "#2ecc71", "lr": "#3498db"}

FEATURE_COLS = json.loads((MODELS_DIR / "feature_cols.json").read_text())

# Reuse _prepare_X from train module (handles fuel_type one-hot + NaN fill)
from wildfire_hotspot_prediction.model.train import _prepare_X


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_model(fold: int, name: str):
    path = MODELS_DIR / f"model_fold_{fold}_{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_threshold(fold: int, name: str) -> float:
    thr = json.loads((MODELS_DIR / f"model_fold_{fold}_thresholds.json").read_text())
    return thr[name]


# ── Per-fold charts ───────────────────────────────────────────────────────────
all_fold_metrics = {}
all_fold_imps    = {}

for fold in range(1, N_FOLDS + 1):
    fold_name = f"fold_{fold}"
    fold_pred_dir = PRED_DIR / fold_name
    fold_pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  {fold_name.upper()}")
    print(f"{'#' * 60}")

    # Load test data
    test_df = pd.read_parquet(TRAIN_DIR / fold_name / "test.parquet")
    X_test  = _prepare_X(test_df)
    y_test  = test_df["label"].values

    print(f"  Test: {len(test_df):,} rows  (burned={int(y_test.sum()):,}  "
          f"unburned={int((y_test == 0).sum()):,})")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr,  ax_pr  = plt.subplots(figsize=(8, 6))
    metrics_summary = {}

    for mname in MODELS:
        model   = _load_model(fold, mname)
        opt_thr = _load_threshold(fold, mname)
        label   = MODEL_LABELS[mname]
        color   = MODEL_COLORS[mname]

        print(f"\n  {label}  (threshold={opt_thr:.4f})")

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= opt_thr).astype(int)

        roc_val  = roc_auc_score(y_test, y_prob)
        ap_val   = average_precision_score(y_test, y_prob)
        prec_val = precision_score(y_test, y_pred, zero_division=0)
        rec_val  = recall_score(y_test, y_pred, zero_division=0)
        f1_val   = f1_score(y_test, y_pred, zero_division=0)

        metrics_summary[mname] = {
            "auc": roc_val, "avg_prec": ap_val,
            "precision": prec_val, "recall": rec_val,
            "f1": f1_val, "threshold": opt_thr,
        }
        print(f"    AUC={roc_val:.4f}  AP={ap_val:.4f}  "
              f"P={prec_val:.4f}  R={rec_val:.4f}  F1={f1_val:.4f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_val:.3f})")

        # PR curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(rec_arr, prec_arr, color=color, label=f"{label} (AP={ap_val:.4f})")

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=["No Fire", "Fire"],
        ).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"{label}  [{fold_name}]  (thr={opt_thr:.3f})")
        fig_cm.tight_layout()
        fig_cm.savefig(fold_pred_dir / f"confusion_{mname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_cm)

        # Feature importance (tree-based)
        if mname in ("xgb", "rf"):
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            imps  = clf.feature_importances_
            order = np.argsort(imps)[::-1]
            fig_fi, ax_fi = plt.subplots(figsize=(12, 5))
            top_n = min(20, len(order))
            ax_fi.bar(range(top_n), imps[order[:top_n]], color=color, alpha=0.85)
            ax_fi.set_xticks(range(top_n))
            ax_fi.set_xticklabels([FEATURE_COLS[i] for i in order[:top_n]],
                                  rotation=45, ha="right", fontsize=9)
            ax_fi.set_title(f"{label} [{fold_name}] — Top {top_n} Feature Importances")
            ax_fi.set_ylabel("Importance")
            ax_fi.grid(axis="y", alpha=0.3)
            fig_fi.tight_layout()
            fig_fi.savefig(fold_pred_dir / f"feat_imp_{mname}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_fi)

            # Collect for cross-fold
            all_fold_imps.setdefault(mname, []).append(imps)

        # LR coefficients
        if mname == "lr":
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            coefs = clf.coef_[0]
            order_lr = np.argsort(np.abs(coefs))[::-1]
            fig_lr, ax_lr = plt.subplots(figsize=(10, 7))
            top_n = min(20, len(order_lr))
            colors_lr = ["#d62728" if coefs[i] > 0 else "#1f77b4" for i in order_lr[:top_n]]
            ax_lr.barh(range(top_n), coefs[order_lr[:top_n]][::-1],
                       color=colors_lr[::-1], alpha=0.85)
            ax_lr.set_yticks(range(top_n))
            ax_lr.set_yticklabels([FEATURE_COLS[order_lr[top_n - 1 - i]] for i in range(top_n)],
                                  fontsize=9)
            ax_lr.axvline(0, color="black", lw=0.8)
            ax_lr.set_xlabel("Standardised Coefficient (red = increases fire probability)")
            ax_lr.set_title(f"Logistic Regression [{fold_name}] — Coefficients")
            ax_lr.grid(axis="x", alpha=0.3)
            fig_lr.tight_layout()
            fig_lr.savefig(fold_pred_dir / "coef_lr.png", dpi=150, bbox_inches="tight")
            plt.close(fig_lr)

        # Per-pair metrics
        pair_idx = pd.read_parquet(TRAIN_DIR / "pair_index.parquet")
        test_pairs = test_df[["pair_id"]].drop_duplicates().merge(
            pair_idx[["pair_id", "T1", "T2"]], on="pair_id",
        ).sort_values("T1")

        pair_aucs, pair_recs, pair_precs = [], [], []
        pair_labels_list = []
        for _, prow in test_pairs.iterrows():
            mask = test_df["pair_id"] == prow["pair_id"]
            ys = y_test[mask]
            yp = y_prob[mask]
            yd = y_pred[mask]
            if ys.sum() > 0 and (ys == 0).sum() > 0:
                pair_aucs.append(roc_auc_score(ys, yp))
            else:
                pair_aucs.append(np.nan)
            pair_recs.append(recall_score(ys, yd, zero_division=0))
            pair_precs.append(precision_score(ys, yd, zero_division=0))
            pair_labels_list.append(pd.Timestamp(prow["T1"]).strftime("%m/%d %H:%M"))

        pair_aucs  = np.array(pair_aucs, dtype=float)
        pair_recs  = np.array(pair_recs, dtype=float)
        pair_precs = np.array(pair_precs, dtype=float)
        x_idx = np.arange(len(pair_labels_list))

        fig_pp, axes_pp = plt.subplots(3, 1,
                                       figsize=(max(12, len(pair_labels_list) * 0.6), 10),
                                       sharex=True)
        fig_pp.suptitle(f"{label} — Per-Test-Pair Performance\n"
                        f"(threshold={opt_thr:.3f}, {fold_name})", fontsize=12)
        for ax, vals, ylabel, c in zip(
                axes_pp,
                [pair_aucs, pair_recs, pair_precs],
                ["ROC-AUC", "Recall", "Precision"],
                ["steelblue", "darkorange", "seagreen"]):
            ax.bar(x_idx, np.nan_to_num(vals), color=c, alpha=0.85, width=0.7)
            mean_val = float(np.nanmean(vals))
            ax.axhline(mean_val, color="darkred", lw=1.5, ls="--",
                       label=f"Mean = {mean_val:.3f}")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.08)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
        axes_pp[-1].set_xticks(x_idx)
        axes_pp[-1].set_xticklabels(pair_labels_list, rotation=45, ha="right", fontsize=8)
        axes_pp[-1].set_xlabel("Test pair T1 timestamp")
        fig_pp.tight_layout()
        fig_pp.savefig(fold_pred_dir / f"per_pair_metrics_{mname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_pp)

    # Save ROC curves
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax_roc.set(xlabel="FPR", ylabel="TPR", title=f"ROC Curves [{fold_name}]")
    ax_roc.legend(); ax_roc.grid(alpha=0.3); fig_roc.tight_layout()
    fig_roc.savefig(fold_pred_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    # Save PR curves
    prevalence = y_test.mean()
    ax_pr.axhline(prevalence, color="gray", ls="--", alpha=0.5,
                  label=f"Baseline ({prevalence:.4f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curves [{fold_name}]")
    ax_pr.legend(); ax_pr.grid(alpha=0.3); fig_pr.tight_layout()
    fig_pr.savefig(fold_pred_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    # Model comparison (6-panel)
    mnames_list = list(MODELS)
    short_names = [MODEL_LABELS[m] for m in mnames_list]
    palette     = [MODEL_COLORS[m] for m in mnames_list]
    metric_keys = ["auc", "avg_prec", "precision", "recall", "f1"]
    metric_lbls = ["ROC AUC", "Avg Precision", "Precision\n(at threshold)",
                   "Recall\n(at threshold)", "F1\n(at threshold)"]

    fig_mc, axes_mc = plt.subplots(2, 3, figsize=(14, 8))
    fig_mc.suptitle(f"Model Comparison [{fold_name}]", fontsize=13, fontweight="bold")
    for ax, key, lbl in zip(axes_mc.ravel(), metric_keys, metric_lbls):
        vals = [metrics_summary[m][key] for m in mnames_list]
        bars = ax.bar(short_names, vals, color=palette, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, min(1.0, max(vals) * 1.25 + 0.05))
        ax.set_title(lbl, fontsize=10)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
    # Hide unused subplot
    axes_mc.ravel()[-1].set_visible(False)
    fig_mc.tight_layout()
    fig_mc.savefig(fold_pred_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mc)

    all_fold_metrics[fold_name] = metrics_summary
    print(f"\n  Charts saved -> {fold_pred_dir}")


# ── Cross-fold comparison ────────────────────────────────────────────────────
CHART_DIR.mkdir(parents=True, exist_ok=True)

fold_names  = [f"fold_{k}" for k in range(1, N_FOLDS + 1)]
fold_colors = ["#2196F3", "#FF5722", "#4CAF50"]
metric_keys = ["auc", "avg_prec", "precision", "recall", "f1"]
metric_lbls = ["ROC AUC", "Avg Precision", "Precision", "Recall", "F1"]

fig_cf = plt.figure(figsize=(18, 14))
fig_cf.suptitle("Cross-Fold Model Comparison", fontsize=14, fontweight="bold", y=0.995)
gs = GridSpec(3, 3, figure=fig_cf, hspace=0.52, wspace=0.35,
              top=0.96, bottom=0.06, left=0.07, right=0.97)

mnames_list = list(MODELS)
x = np.arange(len(mnames_list))
width = 0.22
offsets = np.linspace(-(N_FOLDS - 1) / 2, (N_FOLDS - 1) / 2, N_FOLDS) * width

# Row 0-1: metric subplots (5 metrics)
for idx, (mkey, mlbl) in enumerate(zip(metric_keys, metric_lbls)):
    ax = fig_cf.add_subplot(gs[idx // 3, idx % 3])
    for fi, (fn, fc) in enumerate(zip(fold_names, fold_colors)):
        vals = [all_fold_metrics[fn][m][mkey] for m in mnames_list]
        bars = ax.bar(x + offsets[fi], vals, width, label=fn, color=fc, alpha=0.82)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in mnames_list], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title(mlbl, fontsize=10)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

# Blank the 6th cell in 2x3
ax_blank = fig_cf.add_subplot(gs[1, 2])
ax_blank.set_visible(False)

# Row 2: feature importances (RF + XGB averaged across folds)
for col_i, imp_model in enumerate(["rf", "xgb"]):
    if imp_model not in all_fold_imps:
        continue
    ax_imp = fig_cf.add_subplot(gs[2, col_i])
    avg_imp = np.mean(all_fold_imps[imp_model], axis=0)
    order = np.argsort(avg_imp)[::-1]
    top_n = min(15, len(order))
    colors_imp = ["#e05c00" if i < 3 else "#888" for i in range(top_n)]
    ax_imp.barh(range(top_n), avg_imp[order[:top_n]][::-1],
                color=colors_imp[::-1], alpha=0.85)
    ax_imp.set_yticks(range(top_n))
    ax_imp.set_yticklabels([FEATURE_COLS[order[top_n - 1 - i]] for i in range(top_n)],
                           fontsize=8)
    ax_imp.set_xlabel("Avg importance (3 folds)", fontsize=8)
    ax_imp.set_title(f"{MODEL_LABELS[imp_model]} — Top Features", fontsize=9)
    ax_imp.grid(axis="x", alpha=0.3)

# Row 2, col 2: summary text
ax_txt = fig_cf.add_subplot(gs[2, 2])
ax_txt.axis("off")
lines = [f"Features: {len(FEATURE_COLS)}", ""]
for fn in fold_names:
    lines.append(f"{fn}:")
    for m in mnames_list:
        d = all_fold_metrics[fn][m]
        lines.append(f"  {MODEL_LABELS[m]}: AUC={d['auc']:.3f}  AP={d['avg_prec']:.4f}  "
                     f"F1={d['f1']:.3f}")
    lines.append("")
ax_txt.text(0.03, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
            fontsize=8, fontfamily="monospace", verticalalignment="top")

fig_cf.savefig(CHART_DIR / "cross_fold_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig_cf)
print(f"\n[plot] cross_fold_comparison.png -> {CHART_DIR}")

print("\nDone. All charts saved.")
