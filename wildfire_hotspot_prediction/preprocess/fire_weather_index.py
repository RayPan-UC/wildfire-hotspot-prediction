"""
preprocess/fire_weather_index.py
---------------------------------
Pre-compute daily Canadian FBP fire weather indices from ERA5-Land data.

Computes for every ERA5 grid cell and every day in the study period:
  - FFMC  (Fine Fuel Moisture Code)
  - ISI   (Initial Spread Index)  — hourly, from FFMC + wind speed
  - ROS   (Rate of Spread)        — hourly, from ISI + fuel type

These are expensive to compute per-pair; pre-computing and caching them
makes feature extraction fast during training data construction.

Reads:
    <project>/data_processed/weather/era5.parquet
    <project>/data_processed/landcover/fuel_type.tif

Produces:
    <project>/data_processed/weather/ffmc_daily.parquet
        cols: date, grid_id, ffmc
    <project>/data_processed/weather/isi_hourly.parquet
        cols: valid_time, grid_id, isi
    <project>/data_processed/weather/ros_hourly.parquet
        cols: valid_time, grid_id, ros
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_hotspot_prediction.study import Study
from wildfire_hotspot_prediction.utils.raster import RasterSampler

# ── Canadian FBP fuel type parameters (code → (a, b, c)) ─────────────────────
# RSI = a * (1 - exp(-b * ISI))^c   [m/min]
# Codes 101–122 map to FBP types C-1…S-3 + non-fuel
_FBP_PARAMS: dict[int, tuple[float, float, float]] = {
    101: (90,  0.0649, 4.5),   # C-1 Spruce-Lichen Woodland
    102: (110, 0.0282, 1.5),   # C-2 Boreal Spruce
    103: (110, 0.0444, 3.0),   # C-3 Mature Jack/Lodgepole Pine
    104: (110, 0.0293, 1.5),   # C-4 Immature Jack/Lodgepole Pine
    105: (30,  0.0697, 4.0),   # C-5 Red and White Pine
    106: (30,  0.0800, 3.0),   # C-6 Conifer Plantation
    107: (45,  0.0305, 2.0),   # C-7 Ponderosa Pine - Douglas-Fir
    108: (30,  0.0232, 1.6),   # D-1 Leafless Aspen
    109: (30,  0.0232, 1.6),   # D-2 Green Aspen (approx D-1)
    110: None,                  # M-1 Boreal Mixedwood Leafless (blended)
    111: None,                  # M-2 Boreal Mixedwood Green (blended)
    112: (120, 0.0572, 1.4),   # M-3 Dead Balsam Fir Mixedwood Leafless
    113: (100, 0.0404, 1.48),  # M-4 Dead Balsam Fir Mixedwood Green
    114: (190, 0.0310, 1.4),   # O-1a Matted Grass
    115: (250, 0.0350, 1.7),   # O-1b Standing Grass
    116: (75,  0.0297, 1.3),   # S-1 Jack/Lodgepole Pine Slash
    117: (40,  0.0438, 1.7),   # S-2 White Spruce - Balsam Fir Slash
    118: (55,  0.0829, 3.2),   # S-3 Coastal Cedar - Hemlock - Douglas Fir Slash
    119: None,                  # Non-fuel
    120: None,                  # Water
    121: None,                  # Urban
    122: None,                  # Unknown
}
_PC_DEFAULT = 50.0  # % conifer for M-1/M-2 when not available


def _solar_noon_utc(study) -> int:
    """Estimate solar noon as UTC hour from the study bbox centroid longitude."""
    lon_centre = (study.bbox[0] + study.bbox[2]) / 2
    return round(12 - lon_centre / 15) % 24


def build_fire_weather_index(study: Study) -> Path:
    """Pre-compute FFMC, ISI, and ROS for all ERA5 grid cells and timesteps.

    Args:
        study: Study instance.

    Returns:
        Path to data_processed/weather/ directory.
    """
    ffmc_path = study.weather_dir / "ffmc_daily.parquet"
    isi_path  = study.weather_dir / "isi_hourly.parquet"
    ros_path  = study.weather_dir / "ros_hourly.parquet"

    if ffmc_path.exists() and isi_path.exists() and ros_path.exists():
        print("[preprocess] FWI already exists, skipping")
        return study.weather_dir

    # ── Load ERA5 parquet ─────────────────────────────────────────────────────
    era5 = pd.read_parquet(study.weather_dir / "era5.parquet")
    era5["valid_time"] = pd.to_datetime(era5["valid_time"])
    era5 = era5.sort_values("valid_time").reset_index(drop=True)

    grid_ids = sorted(era5["grid_id"].unique())
    n_grid   = len(grid_ids)

    # Build (n_times × n_grid) arrays
    pivot = era5.pivot_table(index="valid_time", columns="grid_id", values=[
        "temp_c", "rh", "wind_speed", "precip_mm"
    ], sort=True)
    pivot = pivot.reindex(sorted(pivot.index))

    temp_c     = pivot["temp_c"].values      # (n_times, n_grid)
    rh         = pivot["rh"].values
    wind_speed = pivot["wind_speed"].values
    precip_mm  = pivot["precip_mm"].values

    era5_times = pd.to_datetime(pivot.index)

    # ── FFMC daily ────────────────────────────────────────────────────────────
    noon_utc = _solar_noon_utc(study)
    print(f"[preprocess] solar noon UTC = {noon_utc:02d}:00 (lon centre = {(study.bbox[0]+study.bbox[2])/2:.1f}°)")
    print("[preprocess] computing FFMC daily...")
    ffmc_dict = compute_ffmc_daily(era5_times, temp_c, rh, wind_speed, precip_mm,
                                   noon_utc=noon_utc)

    ffmc_rows = []
    for d, arr in ffmc_dict.items():
        for gidx, val in enumerate(arr):
            ffmc_rows.append({"date": d, "grid_id": grid_ids[gidx], "ffmc": float(val)})
    ffmc_df = pd.DataFrame(ffmc_rows)
    ffmc_df.to_parquet(ffmc_path, index=False)
    print(f"[preprocess] FFMC → {ffmc_path}")

    # ── ISI hourly ────────────────────────────────────────────────────────────
    print("[preprocess] computing ISI hourly...")
    # Map each hourly timestamp to its day's FFMC
    ffmc_arr = np.zeros_like(wind_speed)
    for i, ts in enumerate(era5_times):
        d = ts.date()
        if d in ffmc_dict:
            ffmc_arr[i] = ffmc_dict[d]

    isi_2d = compute_isi(ffmc_arr, wind_speed)   # (n_times, n_grid)

    isi_rows = []
    for i, ts in enumerate(era5_times):
        for gidx in range(n_grid):
            isi_rows.append({"valid_time": ts, "grid_id": grid_ids[gidx],
                             "isi": float(isi_2d[i, gidx])})
    isi_df = pd.DataFrame(isi_rows)
    isi_df.to_parquet(isi_path, index=False)
    print(f"[preprocess] ISI → {isi_path}")

    # ── Fuel type per ERA5 grid_id ────────────────────────────────────────────
    # grid_ids are already EPSG:3978 "x_y" strings — decode directly
    fuel_tif = study.landcover_dir / "fuel_type.tif"
    if fuel_tif.exists():
        xy = np.array([list(map(float, g.split("_"))) for g in grid_ids])
        fuel_grid = RasterSampler(fuel_tif).sample(xy).astype(int)
    else:
        print("[preprocess] fuel_type.tif not found — ROS will be zero")
        fuel_grid = np.zeros(n_grid, dtype=int)

    # ── ROS hourly ────────────────────────────────────────────────────────────
    print("[preprocess] computing ROS hourly...")
    ros_rows = []
    for i, ts in enumerate(era5_times):
        ros_row = compute_ros(isi_2d[i], fuel_grid)
        for gidx in range(n_grid):
            ros_rows.append({"valid_time": ts, "grid_id": grid_ids[gidx],
                             "ros": float(ros_row[gidx])})
    ros_df = pd.DataFrame(ros_rows)
    ros_df.to_parquet(ros_path, index=False)
    print(f"[preprocess] ROS → {ros_path}")

    return study.weather_dir


def compute_ffmc_daily(
    era5_times: np.ndarray,
    temp_c:     np.ndarray,
    rh:         np.ndarray,
    wind_speed: np.ndarray,
    precip_mm:  np.ndarray,
    ffmc0:      float = 85.0,
    noon_utc:   int   = 18,
) -> dict:
    """Compute daily FFMC for every ERA5 grid cell over the study period.

    Uses 18:00 UTC as proxy for local solar noon.
    Precipitation is the 24-hour accumulated total ending at 18:00 UTC.

    Args:
        era5_times:  Array of ERA5 valid_time values, shape (n_times,).
        temp_c:      Temperature [°C],        shape (n_times, n_grid).
        rh:          Relative humidity [%],   shape (n_times, n_grid).
        wind_speed:  Wind speed [m/s],        shape (n_times, n_grid).
        precip_mm:   Precipitation [mm],      shape (n_times, n_grid).
        ffmc0:       Initial FFMC value. Defaults to 85.0 (standard dry condition).

    Returns:
        Dict mapping datetime.date → np.ndarray of shape (n_grid,) float32.
    """
    times  = pd.to_datetime(era5_times)
    dates  = sorted({t.date() for t in times})
    n_grid = temp_c.shape[1]

    ffmc_prev = np.full(n_grid, ffmc0, dtype=np.float32)
    result: dict = {}

    for d in dates:
        noon_idx = [i for i, t in enumerate(times) if t.date() == d and t.hour == noon_utc]
        if not noon_idx:
            result[d] = ffmc_prev.copy()
            continue
        i = noon_idx[0]

        T = np.maximum(temp_c[i].astype(np.float32), -1.1)   # FFMC temp clamp
        H = np.clip(rh[i].astype(np.float32), 1, 99)
        W = wind_speed[i].astype(np.float32) * 3.6    # m/s → km/h

        # 24-hour accumulated precipitation ending at noon_utc
        prev_day = d - timedelta(days=1)
        precip_mask = [
            (t.date() == prev_day and t.hour >= noon_utc) or
            (t.date() == d        and t.hour <  noon_utc)
            for t in times
        ]
        if any(precip_mask):
            P = precip_mm[np.array(precip_mask)].sum(axis=0).astype(np.float32)
        else:
            P = np.zeros(n_grid, dtype=np.float32)

        ffmc_prev = _ffmc_step(ffmc_prev, T, H, W, P)
        result[d] = ffmc_prev.copy()

    return result


def _ffmc_step(
    ffmc_prev: np.ndarray,
    T: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """One-day FFMC update. All arrays shape (n_grid,)."""
    mo = 147.2 * (101 - ffmc_prev) / (59.5 + ffmc_prev)

    # Precipitation
    wet = P > 0.5
    if np.any(wet):
        rf = np.where(wet, P - 0.5, 0.0)
        exp_term = np.exp(-100 / np.maximum(251 - mo, 1e-6))
        mr = mo + 42.5 * rf * exp_term * (1 - np.exp(-6.93 / np.maximum(rf, 1e-6)))
        # Extra term for high moisture
        mr = np.where(wet & (mo > 150),
                      mr + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf), mr)
        mr = np.minimum(mr, 250)
        mo = np.where(wet, mr, mo)

    # Equilibrium moisture
    ed = (0.942 * H ** 0.679
          + 11 * np.exp((H - 100) / 10)
          + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H)))
    ew = (0.618 * H ** 0.753
          + 10 * np.exp((H - 100) / 10)
          + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H)))

    # Drying
    kd  = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - (H / 100) ** 8)
    kwd = kd * 0.581 * np.exp(0.0365 * T)
    m_dry = ed + (mo - ed) * 10 ** (-kwd)

    # Wetting
    kw  = (0.424 * (1 - ((100 - H) / 100) ** 1.7)
           + 0.0694 * np.sqrt(W) * (1 - ((100 - H) / 100) ** 8))
    kww = kw * 0.581 * np.exp(0.0365 * T)
    m_wet = ew - (ew - mo) * 10 ** (-kww)

    m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))
    ffmc = 59.5 * (250 - m) / (147.2 + m)
    return np.clip(ffmc, 0, 101).astype(np.float32)


def compute_isi(ffmc: np.ndarray, wind_speed_ms: np.ndarray) -> np.ndarray:
    """Compute Initial Spread Index from FFMC and wind speed.

    Args:
        ffmc:          FFMC values, shape (n,) or (n_times, n_grid).
        wind_speed_ms: Wind speed [m/s], same shape as ffmc.

    Returns:
        ISI array, same shape, float32.
    """
    W  = wind_speed_ms * 3.6                         # m/s → km/h
    fW = np.exp(0.05039 * W)
    fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    fF = 91.9 * np.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
    return (0.208 * fW * fF).astype(np.float32)


def compute_ros(isi: np.ndarray, fuel_type: np.ndarray) -> np.ndarray:
    """Compute Rate of Spread [m/min] from ISI and Canadian FBP fuel type codes.

    Args:
        isi:       ISI values, shape (n,).
        fuel_type: FBP fuel type codes (101–122), shape (n,).

    Returns:
        ROS array [m/min], shape (n,), float32.
    """
    ros = np.zeros(len(isi), dtype=np.float32)

    for code, params in _FBP_PARAMS.items():
        mask = fuel_type == code
        if not np.any(mask):
            continue

        if code in (110, 111):
            # M-1/M-2: blend C-2 and D-1 components
            a_c2, b_c2, c_c2 = _FBP_PARAMS[102]
            a_d1, b_d1, c_d1 = _FBP_PARAMS[108]
            pc = _PC_DEFAULT / 100
            ph = 1 - pc
            rsi_c2 = a_c2 * (1 - np.exp(-b_c2 * isi[mask])) ** c_c2
            rsi_d1 = a_d1 * (1 - np.exp(-b_d1 * isi[mask])) ** c_d1
            if code == 110:   # M-1: RSI = PC*C2 + PH*D1
                ros[mask] = pc * rsi_c2 + ph * rsi_d1
            else:             # M-2: RSI = PC*C2 + 0.2*PH*D1
                ros[mask] = pc * rsi_c2 + 0.2 * ph * rsi_d1
        elif params is not None:
            a, b, c = params
            ros[mask] = a * (1 - np.exp(-b * isi[mask])) ** c

    return ros
