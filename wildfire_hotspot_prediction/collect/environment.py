"""
collect/environment.py
----------------------
Download static and dynamic environmental datasets:
  - ERA5-Land hourly weather (wind, temperature, humidity, precipitation)
  - MRDEM terrain (DTM, slope, aspect)
  - Land cover / fuel type raster (Canadian FBP system)

Outputs:
    <project>/data_raw/weather/era5.grib
    <project>/data_raw/terrain/dtm.tif
    <project>/data_raw/terrain/slope.tif
    <project>/data_raw/terrain/aspect.tif
    <project>/data_raw/landcover/fuel_type.tif

Credentials (ERA5 only):
    Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-key>
"""

from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests
import rasterio
import rasterio.windows
from rasterio.crs import CRS
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box

from wildfire_hotspot_prediction.study import Study

# MRDEM-30 Cloud-Optimized GeoTIFF (EPSG:3979)
_MRDEM_DTM_URL = (
    "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.tif"
)

# NRCan FBP fuel type archives
_LANDCOVER_URLS = {
    2014: "https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/archive/National_FBP_Fueltypes_version2014b.zip",
    2024: "https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/current/FBP_fueltypes_Canada_100m_EPSG3978_20240527.tif",
}

_ERA5_VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]


def collect_environment(
    study:    Study,
    sources:  list[str] = None,
    cds_key:  str = None,
) -> Path:
    """Download environmental datasets for the study area.

    Args:
        study:    Study instance defining the AOI and time range.
        sources:  Datasets to download. Any subset of ["era5", "terrain", "landcover"].
                  Defaults to all three.
        cds_key:  Copernicus CDS personal access token for ERA5 download.
                  If not provided, falls back to ~/.cdsapirc.

    Returns:
        Path to data_raw directory.
    """
    if sources is None:
        sources = ["era5", "terrain", "landcover"]

    if "era5" in sources:
        _collect_era5(study, cds_key=cds_key)
    if "terrain" in sources:
        _collect_terrain(study)
    if "landcover" in sources:
        _collect_landcover(study)

    return study.project_dir / "data_raw"


def _collect_era5(study: Study, cds_key: str = None) -> Path:
    """Download ERA5-Land hourly reanalysis data for the study AOI and period.

    Variables: 10m_u_component_of_wind, 10m_v_component_of_wind,
               2m_temperature, 2m_dewpoint_temperature, total_precipitation.

    Requires a Copernicus CDS personal access token, passed via cds_key,
    the CDS_KEY environment variable, or stored in ~/.cdsapirc.

    Returns:
        Path to the downloaded ERA5 .grib file.
    """
    import cdsapi

    out_path = study.weather_raw_dir / "era5.nc"
    if out_path.exists():
        print(f"[era5] already exists, skipping → {out_path}")
        return out_path

    print("[era5] NOTE: ERA5 download may take 10–20 minutes depending on "
          "Copernicus server queue. Please be patient.")

    study.weather_raw_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = study.bbox
    # ERA5 area: [North, West, South, East]
    area = [lat_max, lon_min, lat_min, lon_max]

    start = date.fromisoformat(study.start_date)
    end   = date.fromisoformat(study.end_date)

    print(f"[era5] {study.start_date} → {study.end_date}")
    print(f"[era5] area (N,W,S,E) = {area}")
    print(f"[era5] output → {out_path}")

    # Collect all days in range (grouped into one request per month)
    # For simplicity, do a single request covering all months
    months_days: dict[tuple[int, int], list[str]] = {}
    current = start
    while current <= end:
        key = (current.year, current.month)
        months_days.setdefault(key, []).append(f"{current.day:02d}")
        current += timedelta(days=1)

    key = cds_key or os.environ.get("CDS_KEY")
    client = (
        cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key)
        if key else cdsapi.Client()
    )

    parts = []
    for (yr, mo), days in months_days.items():
        part_path = study.weather_raw_dir / f"era5_{yr}_{mo:02d}.nc"
        if not part_path.exists():
            request = {
                "variable":        _ERA5_VARIABLES,
                "year":            str(yr),
                "month":           f"{mo:02d}",
                "day":             days,
                "time":            [f"{h:02d}:00" for h in range(24)],
                "data_format":     "netcdf",
                "download_format": "unarchived",
                "area":            area,
            }
            print(f"[era5] downloading {yr}-{mo:02d} → {part_path}")
            client.retrieve("reanalysis-era5-land", request).download(str(part_path))
        parts.append(part_path)

    if len(parts) == 1:
        parts[0].rename(out_path)
    else:
        import xarray as xr
        ds = xr.open_mfdataset(parts, combine="by_coords")
        ds.to_netcdf(out_path)
        ds.close()
        for p in parts:
            p.unlink(missing_ok=True)

    print(f"[era5] done → {out_path}")
    return out_path


def _collect_terrain(study: Study) -> Path:
    """Download MRDEM DTM and compute slope and aspect rasters for the study AOI.

    Streams the DTM from the MRDEM-30 Cloud-Optimized GeoTIFF (no full download),
    then computes slope and aspect using numpy gradient.

    Slope:  arctan(sqrt((dz/dx)² + (dz/dy)²))  in degrees
    Aspect: clockwise from north                 in degrees [0, 360)

    Returns:
        Path to the terrain raw directory.
    """
    dtm_path    = study.terrain_raw_dir / "dtm.tif"
    slope_path  = study.terrain_raw_dir / "slope.tif"
    aspect_path = study.terrain_raw_dir / "aspect.tif"

    if dtm_path.exists() and slope_path.exists() and aspect_path.exists():
        print(f"[terrain] already exists, skipping → {study.terrain_raw_dir}")
        return study.terrain_raw_dir

    print("[terrain] NOTE: streaming DTM from MRDEM COG — may take a few minutes "
          "depending on bbox size and network speed.")

    study.terrain_raw_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = study.bbox

    # Reproject bbox to EPSG:3979 (MRDEM native CRS)
    minx, miny, maxx, maxy = transform_bounds(
        CRS.from_epsg(4326), CRS.from_epsg(3979),
        lon_min, lat_min, lon_max, lat_max,
    )
    print(f"[terrain] streaming DTM window from MRDEM COG...")

    with rasterio.open(_MRDEM_DTM_URL) as src:
        window      = src.window(minx, miny, maxx, maxy)
        raster_data = src.read(window=window)
        meta        = src.meta.copy()
        meta.update({
            "height":    raster_data.shape[1],
            "width":     raster_data.shape[2],
            "count":     1,
            "transform": rasterio.windows.transform(window, src.transform),
            "compress":  "lzw",
        })
        nodata = src.nodata

    with rasterio.open(dtm_path, "w", **meta) as dst:
        dst.write(raster_data)
    print(f"[terrain] saved DTM → {dtm_path}")

    # Compute slope and aspect from DTM
    dem = raster_data[0].astype("float32")
    if nodata is not None:
        dem[dem == nodata] = np.nan

    xres = abs(meta["transform"].a)
    yres = abs(meta["transform"].e)
    dz_dy, dz_dx = np.gradient(dem, yres, xres)

    slope  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (90.0 - aspect) % 360.0

    out_nodata = -9999.0
    slope  = np.where(np.isnan(dem), out_nodata, slope).astype("float32")
    aspect = np.where(np.isnan(dem), out_nodata, aspect).astype("float32")

    deriv_meta = meta.copy()
    deriv_meta.update(dtype="float32", count=1, nodata=out_nodata)

    with rasterio.open(slope_path, "w", **deriv_meta) as dst:
        dst.write(slope, 1)
    with rasterio.open(aspect_path, "w", **deriv_meta) as dst:
        dst.write(aspect, 1)

    print(f"[terrain] saved slope  → {slope_path}")
    print(f"[terrain] saved aspect → {aspect_path}")
    return study.terrain_raw_dir


def _collect_landcover(study: Study) -> Path:
    """Download the Canadian FBP fuel type raster and clip it to the study AOI.

    Uses the 2014b archive (best coverage for historical events up to ~2020).

    Returns:
        Path to the landcover raw directory.
    """
    out_path = study.landcover_raw_dir / "fuel_type.tif"
    if out_path.exists():
        print(f"[landcover] already exists, skipping → {out_path}")
        return study.landcover_raw_dir

    study.landcover_raw_dir.mkdir(parents=True, exist_ok=True)

    # Choose landcover year: closest year ≤ event start year
    event_year = date.fromisoformat(study.start_date).year
    available  = sorted(y for y in _LANDCOVER_URLS if y <= event_year)
    lc_year    = available[-1] if available else min(_LANDCOVER_URLS)
    url        = _LANDCOVER_URLS[lc_year]

    print(f"[landcover] downloading FBP fuel type ({lc_year}) from {url}")

    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    # Find the .tif file (may be directly a TIF or inside a zip)
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise FileNotFoundError("No .tif found in landcover zip")
            # Prefer the main named file (e.g. nat_fbpfuels_2014b.tif)
            tif_name = tif_names[0]
            raw_tif  = zf.read(tif_name)
        fuel_bytes = io.BytesIO(raw_tif)
    else:
        fuel_bytes = io.BytesIO(data)

    # Clip to study bbox
    lon_min, lat_min, lon_max, lat_max = study.bbox
    with rasterio.open(fuel_bytes) as src:
        # Reproject AOI bbox to raster CRS using rasterio (avoids pyproj PROJ conflict)
        from rasterio.warp import transform_bounds as _tb
        from rasterio.crs import CRS as _CRS
        from shapely.geometry import mapping
        minx, miny, maxx, maxy = _tb(
            _CRS.from_epsg(4326), src.crs,
            lon_min, lat_min, lon_max, lat_max,
        )
        aoi_proj = [mapping(box(minx, miny, maxx, maxy))]

        out_image, out_transform = rasterio_mask(src, aoi_proj, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",
        })

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"[landcover] saved → {out_path}")
    return study.landcover_raw_dir