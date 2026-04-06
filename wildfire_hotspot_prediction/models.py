"""
wildfire_hotspot_prediction/models.py
--------------------------------------
Download pre-trained model files from Zenodo if not already present.

Models hosted at: https://zenodo.org/records/19435138
Archive contains: model_full_xgb.pkl, model_full_rf.pkl,
                  model_full_lr.pkl, model_full_thresholds.json
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_ZENODO_ARCHIVE = "https://zenodo.org/api/records/19435138/files-archive"

_REQUIRED = [
    "model_full_xgb.pkl",
    "model_full_thresholds.json",
]


def ensure_models(study) -> None:
    """Download model files from Zenodo if any required file is missing.

    Args:
        study: Study instance — models are written to study.models_dir.
    """
    models_dir = study.models_dir
    missing = [f for f in _REQUIRED if not (models_dir / f).exists()]

    if not missing:
        log.info("[models] all model files present — OK")
        return

    log.info("[models] %d file(s) missing — downloading from Zenodo ...", len(missing))
    _download(models_dir)

    still_missing = [f for f in _REQUIRED if not (models_dir / f).exists()]
    if still_missing:
        log.warning("[models] still missing after download: %s", still_missing)
    else:
        log.info("[models] download complete — OK")


def _download(models_dir: Path) -> None:
    import requests

    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("[models] fetching archive from Zenodo (~51 MB) ...")
    with requests.get(_ZENODO_ARCHIVE, stream=True, timeout=300) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1 << 20):
            buf.write(chunk)

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(models_dir)

    log.info("[models] extracted to %s", models_dir)
