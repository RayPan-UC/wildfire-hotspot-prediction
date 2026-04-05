"""
utils/raster.py
---------------
Lightweight rasterio-based raster sampler.
Loads a single-band raster into memory for fast point sampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


class RasterSampler:
    """Load a single-band raster into memory for fast point sampling.

    Pixels are read once at construction time; subsequent sample() calls
    are pure numpy array indexing with no I/O.

    Usage::

        rs = RasterSampler("mrdem_dtm.tif")
        values = rs.sample(xy)   # xy: (n, 2) array of projected coordinates
    """

    def __init__(self, path: str | Path):
        """Load raster into memory.

        Args:
            path: Path to a single-band GeoTIFF (or any rasterio-readable file).
        """
        with rasterio.open(path) as src:
            self._data      = src.read(1).astype(np.float32)
            self._nodata    = src.nodata
            self._transform = src.transform
            self._height    = src.height
            self._width     = src.width

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Sample raster values at projected (x, y) coordinates.

        Out-of-bounds coordinates are clipped to the raster extent.
        NoData pixels are returned as np.nan.

        Args:
            xy: Array of shape (n, 2) with [x, y] columns in the raster CRS.

        Returns:
            1-D float32 array of shape (n,) with sampled values.
        """
        xs, ys = xy[:, 0], xy[:, 1]
        # Affine inverse: (col, row) from (x, y)
        inv = ~self._transform
        cols, rows = inv * (xs, ys)
        rows = np.clip(rows.astype(int), 0, self._height - 1)
        cols = np.clip(cols.astype(int), 0, self._width  - 1)

        values = self._data[rows, cols]
        if self._nodata is not None:
            values = np.where(values == self._nodata, np.nan, values)
        return values.astype(np.float32)
