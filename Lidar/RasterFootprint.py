"""Create footprint polygons for a folder of GeoTIFFs and write them to a GeoPackage."""

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box, shape
from shapely.ops import unary_union

Region = "whidbey"  # 'hood_canal' 'north_sound' 'south_sound' strait_of_juan_de_fuca' 'whidbey'


IN_DIR = Path(r"D:\Data\Lidar")

IN_DIR = IN_DIR / Region

OUT_DIR = IN_DIR / "footprints"
OUT_GPKG = OUT_DIR / f"{Region}_footprints.gpkg"
LAYER_NAME = "footprints"
BOUNDING_LAYER_NAME = f"{Region}_bounding_footprint"

# True = bounding layer is the rectangular extent, False = dissolved union of all footprints
BOUNDING_AS_BBOX = False

# True = trace the valid-data edge, False = simple bounding box of each raster
VALID_DATA_ONLY = True
# Downsample factor used when tracing valid data (1 = full resolution, higher = faster/coarser)
DECIMATION = 1


def valid_data_footprint(src, decimation=DECIMATION):
    """Polygon covering pixels that are not nodata, read at reduced resolution."""
    height = max(1, src.height // decimation)
    width = max(1, src.width // decimation)
    mask = src.read_masks(1, out_shape=(height, width))
    transform = src.transform * src.transform.scale(
        src.width / width, src.height / height
    )

    geoms = [
        shape(geom)
        for geom, value in features.shapes(mask, mask=mask > 0, transform=transform)
        if value > 0
    ]
    if not geoms:
        return None
    return unary_union(geoms)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rasters = sorted(
        p
        for p in IN_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not rasters:
        raise SystemExit(f"No GeoTIFFs found in {IN_DIR}")

    records = []
    crs = None
    for i, path in enumerate(rasters, 1):
        print(f"[{i}/{len(rasters)}] {path.name}")
        try:
            with rasterio.open(path) as src:
                if crs is None:
                    crs = src.crs
                elif src.crs != crs:
                    print(f"  skipped: CRS {src.crs} differs from {crs}")
                    continue

                geom = (
                    valid_data_footprint(src) if VALID_DATA_ONLY else box(*src.bounds)
                )
                if geom is None or geom.is_empty:
                    print("  skipped: no valid data")
                    continue

                records.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "width": src.width,
                        "height": src.height,
                        "pixel_size_x": src.res[0],
                        "pixel_size_y": src.res[1],
                        "geometry": geom,
                    }
                )
        except rasterio.errors.RasterioIOError as exc:
            print(f"  skipped: {exc}")

    if not records:
        raise SystemExit("No footprints produced.")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    gdf.to_file(OUT_GPKG, layer=LAYER_NAME, driver="GPKG")
    print(f"Wrote {len(gdf)} footprints to {OUT_GPKG} (layer: {LAYER_NAME})")

    bounding = box(*gdf.total_bounds) if BOUNDING_AS_BBOX else unary_union(gdf.geometry)
    bounding_gdf = gpd.GeoDataFrame(
        [{"region": Region, "raster_count": len(gdf), "geometry": bounding}],
        geometry="geometry",
        crs=crs,
    )
    bounding_gdf.to_file(OUT_GPKG, layer=BOUNDING_LAYER_NAME, driver="GPKG")
    print(f"Wrote bounding footprint to {OUT_GPKG} (layer: {BOUNDING_LAYER_NAME})")


if __name__ == "__main__":
    main()
