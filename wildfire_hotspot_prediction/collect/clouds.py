"""
collect/clouds.py
-----------------
On-demand VIIRS cloud mask downloader with file-based cache.
Wraps CloudMaskCache for use in the pipeline.

For each unique T2 timestamp in the hotspot data, finds the nearest
CLDMSK_L2_VIIRS_SNPP granule via NASA CMR and downloads it.

Output:
    <project>/data_raw/clouds/CLDMSK_<YYYYDOY>_<HHMM>.npy   — cloudy pixel (x, y) array
    <project>/data_raw/clouds/CLDMSK_<YYYYDOY>_<HHMM>.none  — sentinel: no granule found

Credentials:
    Set EARTHDATA_TOKEN environment variable (NASA Earthdata Bearer token).
    If absent, cloud masking is silently disabled.
    Get a token at: https://urs.earthdata.nasa.gov/profile
"""

from __future__ import annotations

import os
import re
import time
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

from wildfire_hotspot_prediction.study import Study

warnings.filterwarnings("ignore")

_CLOUDY_THRESH       = 2      # Integer_Cloud_Mask bits 1-0 ≥ 2  →  cloudy
_MATCH_WINDOW_H      = 6.0    # max hours between T2 and granule acquisition time
_CMR_SEARCH_WINDOW_H = 0.5    # ±minutes when querying CMR (orbit-precise)
_CMR_URL             = "https://cmr.earthdata.nasa.gov/search/granules.json"
_FNAME_RE            = re.compile(r"CLDMSK_L2_VIIRS_SNPP\.A(\d{7})\.(\d{4})\.001\.")


def collect_clouds(
    study:            Study,
    timestamps:       list[pd.Timestamp],
    temporal_window:  float = 10.0,
    earthdata_token:  str = None,
) -> Path:
    """Download VIIRS cloud mask granules for the given hotspot timestamps.

    For each timestamp, queries NASA CMR for the nearest
    CLDMSK_L2_VIIRS_SNPP granule within the temporal window,
    downloads and caches cloudy pixel coordinates as .npy arrays.

    Args:
        study:            Study instance.
        timestamps:       List of T2 representative timestamps from hotspot data.
                          Typically the median detection time per revisit group.
        temporal_window:  Max time difference (minutes) between timestamp and
                          granule acquisition time. Defaults to 10 minutes.
        earthdata_token:  NASA Earthdata Bearer token. Falls back to the
                          EARTHDATA_TOKEN environment variable if not provided.

    Returns:
        Path to the clouds raw directory.
    """
    cache = CloudMaskCache(study, temporal_window=temporal_window, earthdata_token=earthdata_token)
    if not cache.enabled:
        print("[clouds] EARTHDATA_TOKEN not set — skipping cloud mask download")
        return study.clouds_raw_dir

    print(f"[clouds] pre-fetching {len(timestamps)} timestamps...")
    for ts in timestamps:
        cache.get_tree(ts)   # triggers download + cache

    return study.clouds_raw_dir


class CloudMaskCache:
    """Lazy, file-cached VIIRS cloud mask lookup.

    Queries NASA CMR API for CLDMSK_L2_VIIRS_SNPP granules,
    downloads HDF5 files, extracts cloudy pixel coordinates,
    and caches them as .npy arrays for fast subsequent access.

    Usage::

        cache = CloudMaskCache(study)
        tree = cache.get_tree(t2_timestamp)   # returns cKDTree | None
        if tree is not None:
            dist, _ = tree.query(candidate_xy, k=1)
    """

    def __init__(self, study: Study, temporal_window: float = 10.0, earthdata_token: str = None):
        """
        Args:
            study:            Study instance (provides cache directory and AOI bbox).
            temporal_window:  Max time difference in minutes between T2 and granule.
            earthdata_token:  NASA Earthdata Bearer token. Falls back to the
                              EARTHDATA_TOKEN environment variable if not provided.
        """
        self._cache_dir = study.clouds_raw_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        lon_min, lat_min, lon_max, lat_max = study.bbox
        self._bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
        self._lat_min, self._lat_max = lat_min, lat_max
        self._lon_min, self._lon_max = lon_min, lon_max
        self._match_window_h = temporal_window / 60.0  # convert minutes to hours

        self._transformer = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)

        token = (earthdata_token or os.environ.get("EARTHDATA_TOKEN", "")).strip()
        if token:
            self._session = requests.Session()
            self._session.headers["Authorization"] = f"Bearer {token}"
            self.enabled = True
        else:
            self._session = None
            self.enabled  = False
            print("  [CloudMaskCache] EARTHDATA_TOKEN not set – cloud masking disabled.")

        self._tree_cache: dict = {}    # cache_key → np.ndarray | None
        self._t2_key_cache: dict = {}  # t2_ns (int) → cache_key | ""

    # ── Public API ────────────────────────────────────────────────────────────

    def reinit_session(self):
        """Recreate the HTTP session — call in forked worker processes to avoid
        inheriting stale TCP connections from the parent process."""
        token = os.environ.get("EARTHDATA_TOKEN", "").strip()
        if token:
            self._session = requests.Session()
            self._session.headers["Authorization"] = f"Bearer {token}"

    def get_tree(self, t2: pd.Timestamp) -> "cKDTree | None":
        """Return a cKDTree of cloudy pixel (x, y) coordinates for the granule
        nearest to t2, or None if no granule found within the temporal window.

        Args:
            t2: Representative acquisition timestamp for a T2 revisit group.

        Returns:
            scipy.spatial.cKDTree | None
        """
        if not self.enabled:
            return None

        t2_ns = t2.value  # int64 nanoseconds — hashable
        if t2_ns in self._t2_key_cache:
            key = self._t2_key_cache[t2_ns]
            if not key:
                return None
            xy = self._tree_cache.get(key)
            if xy is None or not isinstance(xy, np.ndarray):
                return None
            return cKDTree(xy)

        xy = self._get_xy(t2)
        if xy is None or len(xy) == 0:
            return None
        return cKDTree(xy)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_key(self, granule_dt: pd.Timestamp) -> str:
        doy = granule_dt.timetuple().tm_yday
        return f"CLDMSK_{granule_dt.year}{doy:03d}_{granule_dt.hour:02d}{granule_dt.minute:02d}"

    def _npy_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    def _none_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.none"

    def _get_xy(self, t2: pd.Timestamp) -> "np.ndarray | None":
        t2_ns = t2.value

        granule_dt, url = self._find_granule(t2)
        if granule_dt is None:
            self._t2_key_cache[t2_ns] = ""
            return None

        key = self._cache_key(granule_dt)
        self._t2_key_cache[t2_ns] = key

        # In-memory cache hit
        if key in self._tree_cache:
            return self._tree_cache[key]

        # File cache hit
        npy_path  = self._npy_path(key)
        none_path = self._none_path(key)

        if npy_path.exists():
            xy = np.load(npy_path)
            self._tree_cache[key] = xy
            return xy

        if none_path.exists():
            self._tree_cache[key] = None
            return None

        # Download, process, cache
        xy = self._download_and_process(url, key)
        self._tree_cache[key] = xy
        return xy

    def _find_granule(self, t2: pd.Timestamp):
        """Query CMR for granules within ±_CMR_SEARCH_WINDOW_H of t2.
        Returns (granule_dt, download_url) or (None, None)."""
        t_start  = t2 - pd.Timedelta(hours=_CMR_SEARCH_WINDOW_H)
        t_end    = t2 + pd.Timedelta(hours=_CMR_SEARCH_WINDOW_H)
        temporal = (f"{t_start.strftime('%Y-%m-%dT%H:%M:%SZ')},"
                    f"{t_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        try:
            resp = self._session.get(_CMR_URL, params={
                "short_name"   : "CLDMSK_L2_VIIRS_SNPP",
                "temporal"     : temporal,
                "bounding_box" : self._bbox_str,
                "page_size"    : 20,
            }, timeout=30)
            resp.raise_for_status()
            entries = resp.json()["feed"]["entry"]
        except Exception:
            return None, None

        if not entries:
            return None, None

        best_dt   = None
        best_url  = None
        best_diff = pd.Timedelta(hours=_MATCH_WINDOW_H + 1)

        for entry in entries:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if (link.get("type") == "application/x-netcdf"
                        and href.startswith("https://")
                        and href.endswith(".nc")):
                    m = _FNAME_RE.search(href.split("/")[-1])
                    if not m:
                        continue
                    g_dt = (pd.to_datetime(m.group(1), format="%Y%j")
                            + pd.Timedelta(hours=int(m.group(2)[:2]),
                                           minutes=int(m.group(2)[2:])))
                    diff = abs(g_dt - t2)
                    if diff < best_diff:
                        best_diff = diff
                        best_dt   = g_dt
                        best_url  = href
                    break

        if best_dt is None or best_diff > pd.Timedelta(hours=_MATCH_WINDOW_H):
            return None, None

        return best_dt, best_url

    def _download_and_process(self, url: str, key: str) -> "np.ndarray | None":
        fname    = url.split("/")[-1]
        tmp_path = Path(tempfile.gettempdir()) / fname
        try:
            r = self._session.get(url, stream=True, timeout=180, allow_redirects=True)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"  [CloudMaskCache] Download failed {fname}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            self._none_path(key).touch()
            return None

        try:
            xy = self._extract_cloudy_pixels(str(tmp_path))
        except Exception as e:
            print(f"  [CloudMaskCache] Process failed {fname}: {e}")
            xy = None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if xy is not None and len(xy) > 0:
            np.save(self._npy_path(key), xy)
            print(f"  [CloudMask] {key}  {len(xy):,} cloudy px  (cached)")
        else:
            self._none_path(key).touch()
            print(f"  [CloudMask] {key}  no cloudy px in AOI")
            xy = None

        time.sleep(0.1)
        return xy

    def _extract_cloudy_pixels(self, fpath: str) -> "np.ndarray | None":
        """Read one CLDMSK granule, return (n, 2) float32 (x_3978, y_3978) of
        cloudy pixels clipped to the study AOI, or None if empty."""
        with h5py.File(fpath, "r") as f:
            lat  = f["geolocation_data/latitude"][:]
            lon  = f["geolocation_data/longitude"][:]
            mask = f["geophysical_data/Integer_Cloud_Mask"][:]

        lat = np.where(np.abs(lat) > 90,  np.nan, lat.astype(np.float32))
        lon = np.where(np.abs(lon) > 180, np.nan, lon.astype(np.float32))

        cloud_flag = np.array(mask) & 0b11
        in_aoi = (
            (lat >= self._lat_min) & (lat <= self._lat_max) &
            (lon >= self._lon_min) & (lon <= self._lon_max)
        )
        cloudy = in_aoi & (cloud_flag >= _CLOUDY_THRESH)

        lat_c = lat[cloudy].ravel()
        lon_c = lon[cloudy].ravel()
        valid = np.isfinite(lat_c) & np.isfinite(lon_c)
        lat_c, lon_c = lat_c[valid], lon_c[valid]

        if len(lat_c) == 0:
            return None

        x, y = self._transformer.transform(lon_c, lat_c)
        return np.column_stack([x, y]).astype(np.float32)