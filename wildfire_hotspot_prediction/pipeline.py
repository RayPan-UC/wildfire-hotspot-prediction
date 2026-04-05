"""
pipeline.py
-----------
End-to-end pipeline that chains all stages in order.
Can be used as a convenience wrapper or as a reference for the execution order.

Stage order:
  1. collect   — download raw data
  2. preprocess — clean, clip, index
  3. training  — build pair index, fire state, training dataset
  4. model     — train models
  5. predict   — generate predictions
"""

from __future__ import annotations

from wildfire_hotspot_prediction.study import Study


def run_pipeline(
    study:           Study,
    skip_collect:    bool = False,
    skip_preprocess: bool = False,
    skip_training:   bool = False,
    skip_model:      bool = False,
    skip_predict:    bool = False,
) -> None:
    """Run the full wildfire spread prediction pipeline.

    Each stage can be skipped independently (e.g. if data was already
    downloaded in a previous run).

    Args:
        study:           Study instance defining AOI and time range.
        skip_collect:    Skip data download stage.
        skip_preprocess: Skip preprocessing stage.
        skip_training:   Skip training data construction stage.
        skip_model:      Skip model training stage.
        skip_predict:    Skip prediction stage.

    Example::

        study = define_study(...)
        run_pipeline(study, skip_collect=True)   # data already downloaded
    """
    raise NotImplementedError
