from wildfire_hotspot_prediction.collect.hotspots    import collect_hotspots
from wildfire_hotspot_prediction.collect.environment import collect_environment
from wildfire_hotspot_prediction.collect.clouds      import collect_clouds


def collect(
    study,
    firms_api_key:   str = None,
    cds_key:         str = None,
    earthdata_token: str = None,
    sources:         list[str] = None,
):
    """Convenience wrapper — run all collect stages in order.

    Derives cloud mask timestamps directly from the raw hotspot CSV,
    so no preprocessing step is needed.

    Args:
        study:           Study instance.
        firms_api_key:   FIRMS MAP_KEY for hotspot download.
        cds_key:         Copernicus CDS token for ERA5 download.
        earthdata_token: NASA Earthdata token for cloud mask download.
        sources:         Environment sources subset. Defaults to all
                         ["era5", "terrain", "landcover"].
    """
    import pandas as pd

    print("[collect] starting data collection ...")
    print("[collect] stage 1/3 — hotspots")
    collect_hotspots(study, api_key=firms_api_key)

    df = pd.read_csv(study.firms_raw_dir / "hotspots_raw.csv")
    df["datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " +
        df["acq_time"].astype(str).str.zfill(4).str.replace(r"(\d{2})(\d{2})", r"\1:\2", regex=True)
    )
    timestamps = sorted(pd.Timestamp(t) for t in df["datetime"].dropna().unique())
    print(f"[collect] {len(timestamps)} unique timestamps derived from hotspot CSV")

    print("[collect] stage 2/3 — environment")
    collect_environment(study, sources=sources, cds_key=cds_key)

    print("[collect] stage 3/3 — clouds")
    collect_clouds(study, timestamps=timestamps, earthdata_token=earthdata_token)

    print("[collect] done")
