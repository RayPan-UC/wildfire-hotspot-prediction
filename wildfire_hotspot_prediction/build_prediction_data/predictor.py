"""
build_prediction_data/predictor.py
------------------------------------
WildfirePredictor — load a full-data trained model and run inference.

Designed to be called from wildfire-decision-support (or any system) as a
library:

    from wildfire_hotspot_prediction import WildfirePredictor

    predictor = WildfirePredictor(models_dir=Path("path/to/models"))
    prob = predictor.predict_proba(feature_df)           # np.ndarray
    out  = predictor.predict(feature_df, threshold=0.3)  # df + prob/pred cols

The Predictor loads:
    models_dir/model_full_{model_name}.pkl

Feature column names are read from the booster's internal metadata
(``model.get_booster().feature_names`` for XGBoost).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class WildfirePredictor:
    """Load a trained full-data model and predict wildfire spread probability.

    Args:
        models_dir:  Directory containing ``model_full_*.pkl`` and
                     ``feature_cols.json`` (i.e. ``study.models_dir``).
        model_name:  Which model to load: ``"xgb"`` (default), ``"rf"``,
                     or ``"lr"``.
    """

    def __init__(self, models_dir: Path | str, model_name: str = "xgb") -> None:
        self.models_dir  = Path(models_dir)
        self.model_name  = model_name

        # Load model
        model_path = self.models_dir / f"model_full_{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found — run train(use_all_data=True) first."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Derive feature columns: prefer feature_cols.json, then model metadata
        import json
        fc_path = self.models_dir / "feature_cols.json"
        if fc_path.exists():
            self.feature_cols: list[str] = json.loads(fc_path.read_text())
        else:
            booster = getattr(self.model, "get_booster", None)
            if booster is not None:
                self.feature_cols = booster().feature_names or []
            else:
                self.feature_cols = list(getattr(self.model, "feature_names_in_", []))

        self._fuel_dummy_cols = [c for c in self.feature_cols if c.startswith("fuel_type_")]

        log.info("[WildfirePredictor] loaded %s  (%d features)", model_path.name,
                 len(self.feature_cols))

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return spread probability for each candidate cell.

        Args:
            df: DataFrame with columns matching ``feature_cols.json``.
                May include a raw ``fuel_type`` column (int) — it will be
                expanded to dummies automatically.

        Returns:
            1-D float32 array of probabilities (length = len(df)).
        """
        X = self._prepare_X(df)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Return *df* with ``prob`` and ``pred`` columns appended.

        Args:
            df:        Feature DataFrame (see :meth:`predict_proba`).
            threshold: Decision threshold for the binary label. Defaults to 0.5.

        Returns:
            Copy of *df* with two extra columns:
            - ``prob``  (float32) — spread probability
            - ``pred``  (int8)    — 1 if prob ≥ threshold else 0
        """
        prob = self.predict_proba(df)
        out  = df.copy()
        out["prob"] = prob
        out["pred"] = (prob >= threshold).astype(np.int8)
        return out

    def __repr__(self) -> str:
        return (f"WildfirePredictor(model={self.model_name!r}, "
                f"features={len(self.feature_cols)}, "
                f"dir={self.models_dir})")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Expand fuel_type int → dummies aligned to training columns
        if "fuel_type" in df.columns and self._fuel_dummy_cols:
            dummies = pd.get_dummies(df["fuel_type"], prefix="fuel_type",
                                     dtype=np.float32)
            for c in self._fuel_dummy_cols:
                if c not in dummies:
                    dummies[c] = np.float32(0.0)
            df = df.drop(columns=["fuel_type"]).join(dummies[self._fuel_dummy_cols])

        # Fill any missing feature columns with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.float32(0.0)

        X = df[self.feature_cols].values.astype(np.float32)
        for i in np.where(np.any(np.isnan(X), axis=0))[0]:
            col = X[:, i]
            median = float(np.nanmedian(col))
            col[np.isnan(col)] = median if np.isfinite(median) else 0.0
        return X
