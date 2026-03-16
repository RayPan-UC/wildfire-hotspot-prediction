from pathlib import Path
import zipfile
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def download_and_clip_fuel_type(
    aoi_path=Path(r"data_raw\AOI_boundary\AOI.geojson"),
    fuel_dir=Path(r"data_raw\fuel_type"),
    archive=True
):
    # -----------------------------
    # Input paths
    # -----------------------------
    aoi_path = Path(aoi_path)
    fuel_dir = Path(fuel_dir)

    # -----------------------------
    # Source URL and output paths
    # -----------------------------
    if archive:
        fuel_url = (
            "https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/"
            "archive/National_FBP_Fueltypes_version2014b.zip"
        )
        version_dir = fuel_dir / "archive"
        zip_path = version_dir / "National_FBP_Fueltypes_version2014b.zip"
        extract_dir = version_dir / "National_FBP_Fueltypes_version2014b"
    else:
        fuel_url = (
            "https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/"
            "current/FBP_fueltypes_Canada_30m_EPSG3978_20240522.zip"
        )
        version_dir = fuel_dir / "current"
        zip_path = version_dir / "FBP_fueltypes_Canada_30m_EPSG3978_20240522.zip"
        extract_dir = version_dir / "FBP_fueltypes_Canada_30m_EPSG3978_20240522"

    clipped_fuel_path = version_dir / "fuel_type_aoi.tif"
    version_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. Download zip
    # -----------------------------
    if not zip_path.exists():
        print("Downloading fuel type zip...")
        with requests.get(fuel_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Saved: {zip_path}")
    else:
        print(f"Zip already exists: {zip_path}")

    # -----------------------------
    # 2. Extract zip
    # -----------------------------
    if not extract_dir.exists():
        print("Extracting zip...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")
    else:
        print(f"Extract folder already exists: {extract_dir}")

    # -----------------------------
    # 3. Find raster file
    # -----------------------------
    tif_files = list(extract_dir.rglob("*.tif"))
    if not tif_files:
        raise FileNotFoundError("No .tif file found in extracted fuel type folder.")

    if archive:
        # archive 版通常直接選 nat_fbpfuels_2014b.tif
        target_files = [p for p in tif_files if p.name.lower() == "nat_fbpfuels_2014b.tif"]
        fuel_raster_path = target_files[0] if target_files else tif_files[0]
    else:
        fuel_raster_path = tif_files[0]

    print(f"Fuel raster found: {fuel_raster_path}")

    # -----------------------------
    # 4. Read AOI
    # -----------------------------
    aoi = gpd.read_file(aoi_path)

    # -----------------------------
    # 5. Clip raster by AOI
    # -----------------------------
    with rasterio.open(fuel_raster_path) as src:
        aoi_proj = aoi.to_crs(src.crs)

        out_image, out_transform = mask(
            src,
            aoi_proj.geometry,
            crop=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # -----------------------------
    # 6. Save clipped raster
    # -----------------------------
    with rasterio.open(clipped_fuel_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"Clipped fuel type saved to: {clipped_fuel_path}")



if __name__ == "__main__":
    download_and_clip_fuel_type()