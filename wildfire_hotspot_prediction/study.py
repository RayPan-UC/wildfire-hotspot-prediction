"""
study.py
--------
Core dataclass representing a wildfire study area and time period.
Passed to almost every function in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Study:
    """Defines the spatial and temporal scope of a wildfire analysis project.

    Attributes:
        name:         Short identifier used as the project folder name.
        bbox:         (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
        start_date:   Inclusive start date string, e.g. "2016-05-01".
        end_date:     Inclusive end date string, e.g. "2016-05-31".
        project_dir:  Root directory for all project data.
                      Defaults to ./<name>/ relative to the working directory.
    """
    name:        str
    bbox:        tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    start_date:  str
    end_date:    str
    project_dir: Path = field(default=None)

    def __post_init__(self):
        if self.project_dir is None:
            self.project_dir = Path(self.name)
        self.project_dir = Path(self.project_dir)

    # ── data_raw sub-dirs ─────────────────────────────────────────────────────

    @property
    def firms_raw_dir(self) -> Path:
        return self.project_dir / "data_raw" / "firms"

    @property
    def clouds_raw_dir(self) -> Path:
        return self.project_dir / "data_raw" / "clouds"

    @property
    def weather_raw_dir(self) -> Path:
        return self.project_dir / "data_raw" / "weather"

    @property
    def terrain_raw_dir(self) -> Path:
        return self.project_dir / "data_raw" / "terrain"

    @property
    def landcover_raw_dir(self) -> Path:
        return self.project_dir / "data_raw" / "landcover"

    # ── data_processed root ───────────────────────────────────────────────────

    @property
    def data_processed_dir(self) -> Path:
        return self.project_dir / "data_processed"

    @property
    def data_render_dir(self) -> Path:
        return self.project_dir / "data_render"

    # ── data_processed sub-dirs ───────────────────────────────────────────────

    @property
    def firms_dir(self) -> Path:
        return self.project_dir / "data_processed" / "firms"

    @property
    def clouds_dir(self) -> Path:
        return self.project_dir / "data_processed" / "clouds"

    @property
    def weather_dir(self) -> Path:
        return self.project_dir / "data_processed" / "weather"

    @property
    def terrain_dir(self) -> Path:
        return self.project_dir / "data_processed" / "terrain"

    @property
    def landcover_dir(self) -> Path:
        return self.project_dir / "data_processed" / "landcover"

    @property
    def training_dir(self) -> Path:
        return self.project_dir / "data_processed" / "training"

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.project_dir / "predictions"

    def makedirs(self):
        """Create all project subdirectories on disk."""
        for d in [
            self.firms_raw_dir, self.clouds_raw_dir, self.weather_raw_dir,
            self.terrain_raw_dir, self.landcover_raw_dir,
            self.firms_dir, self.clouds_dir, self.weather_dir,
            self.terrain_dir, self.landcover_dir,
            self.training_dir, self.models_dir, self.predictions_dir,
            self.data_render_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def define_study(
    name:        str,
    bbox:        tuple[float, float, float, float],
    start_date:  str,
    end_date:    str,
    project_dir: str | Path = None,
) -> Study:
    """Create a Study and initialise its directory structure on disk.

    Args:
        name:        Short project name used as the root folder name.
        bbox:        (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
        start_date:  Inclusive start date, e.g. "2016-05-01".
        end_date:    Inclusive end date,   e.g. "2016-05-31".
        project_dir: Override root directory. Defaults to ./<name>/.

    Returns:
        Study instance with all subdirectories created on disk.

    Example::

        study = define_study(
            name       = "fort_mcmurray_2016",
            bbox       = (-113.2, 55.8, -109.3, 57.6),
            start_date = "2016-05-01",
            end_date   = "2016-05-31",
        )
    """
    study = Study(
        name=name,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        project_dir=project_dir,
    )
    study.makedirs()
    return study
