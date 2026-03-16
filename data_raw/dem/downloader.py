from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np

from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np


MRDEM_DTM_URL = (
    "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com/"
    "store/elevation/mrdem/mrdem-30/mrdem-30-dtm.tif"
)


def prepare_terrain_layers(
    aoi_path=r"data_raw\AOI_boundary\AOI.geojson",
    output_dir=r"data_raw\dem\MRDEM"
) -> None:
    aoi_path = Path(aoi_path)
    output_dir = Path(output_dir)

    dtm_path = output_dir / "mrdem_aoi_dtm.tif"
    slope_path = output_dir / "mrdem_aoi_slope.tif"
    aspect_path = output_dir / "mrdem_aoi_aspect.tif"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read AOI
    aoi = gpd.read_file(aoi_path)
    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")

    # Reproject AOI to MRDEM CRS
    aoi_3979 = aoi.to_crs("EPSG:3979")

    # Clip DTM from remote COG
    print("Clip DTM from remote COG... (May take 1 minutes)")
    with rasterio.open(MRDEM_DTM_URL) as src:
        out_image, out_transform = mask(
            src,
            aoi_3979.geometry,
            crop=True,
            filled=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            "count": out_image.shape[0],
            "dtype": out_image.dtype,
            "compress": "lzw",
            "nodata": src.nodata
        })

        with rasterio.open(dtm_path, "w", **out_meta) as dst:
            dst.write(out_image)

    print(f"Saved DTM: {dtm_path}")

    # Read clipped DTM
    with rasterio.open(dtm_path) as src:
        dem = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        transform = src.transform

    # Convert nodata to NaN
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # Pixel size in metres
    xres = transform.a
    yres = abs(transform.e)

    # Terrain gradients
    dz_dy, dz_dx = np.gradient(dem, yres, xres)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Aspect in degrees, clockwise from north
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (90.0 - aspect) % 360.0

    output_nodata = -9999.0
    slope = np.where(np.isnan(dem), output_nodata, slope).astype("float32")
    aspect = np.where(np.isnan(dem), output_nodata, aspect).astype("float32")

    profile.update(
        dtype="float32",
        count=1,
        nodata=output_nodata,
        compress="lzw"
    )

    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)

    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)

    print(f"Saved slope: {slope_path}")
    print(f"Saved aspect: {aspect_path}")


if __name__ == "__main__":
    prepare_terrain_layers()