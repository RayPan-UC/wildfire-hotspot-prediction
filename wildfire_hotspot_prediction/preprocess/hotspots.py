"""
preprocess/hotspots.py
----------------------
Clean and index raw VIIRS hotspot data.

Reads:
    <project>/data_raw/firms/hotspots_raw.csv

Produces:
    <project>/data_processed/firms/hotspots.parquet
        cols: datetime, frp, confidence, x_proj, y_proj, overpass_time
"""

from __future__ import annotations

from dataclasses import dataclass
import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from wildfire_hotspot_prediction.study import Study

_PROJ_CRS = "EPSG:3978"   # NAD83 / Canada Atlas Lambert


@dataclass
class HotspotData:
    """Preprocessed hotspot dataset with derived overpass times.

    Attributes:
        gdf:           GeoDataFrame with cols: datetime, frp, confidence,
                       x_proj, y_proj, overpass_time, geometry.
        overpass_times: Sorted list of unique satellite overpass timestamps.
    """
    gdf:            gpd.GeoDataFrame
    overpass_times: list[pd.Timestamp]


def preprocess_hotspots(
    study:          Study,
    confidence:     list[str] = None,
    time_tolerance: float = 10.0,
) -> HotspotData:
    """Filter, reproject and group raw hotspot detections into overpass events.

    Groups detections that fall within time_tolerance of each other into
    a single overpass (representative timestamp = median detection time).

    Args:
        study:          Study instance.
        confidence:     FIRMS confidence levels to keep.
                        Options: "low", "nominal", "high". Defaults to ["nominal", "high"].
        time_tolerance: Max minutes between detections to be grouped into the
                        same overpass. Defaults to 10 minutes.

    Returns:
        HotspotData with cleaned GeoDataFrame and sorted overpass_times list.
    """
    if confidence is None:
        confidence = ["nominal", "high", "n", "h"]

    out_path = study.firms_dir / "hotspots.parquet"

    # ── Read raw CSV ──────────────────────────────────────────────────────────
    raw_path = study.firms_raw_dir / "hotspots_raw.csv"
    df = pd.read_csv(raw_path)

    if df.empty:
        gdf = gpd.GeoDataFrame(columns=["datetime", "frp", "confidence", "x_proj", "y_proj", "overpass_time"])
        gdf.set_crs(_PROJ_CRS, inplace=True)
        return HotspotData(gdf=gdf, overpass_times=[])

    # ── Parse datetime ────────────────────────────────────────────────────────
    df["datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " +
        df["acq_time"].astype(str).str.zfill(4).str.replace(r"(\d{2})(\d{2})", r"\1:\2", regex=True)
    )

    # ── Filter confidence ─────────────────────────────────────────────────────
    if "confidence" in df.columns:
        df = df[df["confidence"].str.lower().isin(confidence)]

    # ── Reproject to projected CRS ────────────────────────────────────────────
    transformer = Transformer.from_crs("EPSG:4326", _PROJ_CRS, always_xy=True)
    x_proj, y_proj = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_proj"] = x_proj
    df["y_proj"] = y_proj

    # ── Group into overpass events ────────────────────────────────────────────
    # Sort by time, then assign group IDs where gap > time_tolerance
    df = df.sort_values("datetime").reset_index(drop=True)
    tolerance = pd.Timedelta(minutes=time_tolerance)
    gaps = df["datetime"].diff() > tolerance
    df["_group"] = gaps.cumsum()

    # Representative timestamp per group = median
    group_ts = df.groupby("_group")["datetime"].median().rename("overpass_time")
    df = df.join(group_ts, on="_group").drop(columns="_group")
    df["overpass_time"] = pd.to_datetime(df["overpass_time"]).dt.floor("min")

    # ── Keep relevant columns ─────────────────────────────────────────────────
    keep = ["datetime", "frp", "confidence", "x_proj", "y_proj", "overpass_time"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # ── Build GeoDataFrame ────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_proj"], df["y_proj"]),
        crs=_PROJ_CRS,
    )

    # ── Save to parquet ───────────────────────────────────────────────────────
    study.firms_dir.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_parquet(out_path, index=False)
    print(f"[preprocess] hotspots → {len(gdf):,} detections, "
          f"{gdf['overpass_time'].nunique()} overpasses → {out_path}")

    overpass_times = sorted(gdf["overpass_time"].unique().tolist())
    return HotspotData(gdf=gdf, overpass_times=overpass_times)
