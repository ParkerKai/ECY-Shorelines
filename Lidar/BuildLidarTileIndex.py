"""Build a spatial index of all lidar GeoTIFF tiles under a root directory.

Each immediate subfolder of ROOT_DIR is treated as one lidar scan (acquisition).
Only rasters beneath a scan's DTM or DSM subfolder are indexed, and each product
is written to its own GeoPackage so the two surfaces stay separate. Footprints
are reprojected to a single INDEX_CRS so all scans live in one queryable index.

Each output GeoPackage has two layers:
  tiles  - one polygon per GeoTIFF, with the path needed to sample it later
  scans  - one dissolved polygon per scan folder
"""

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import box, shape
from shapely.ops import unary_union

ROOT_DIR = Path(r"D:\Data\Lidar")

# Product subfolder name (matched case-insensitively) -> output GeoPackage.
PRODUCTS = {
    "DTM": ROOT_DIR / "lidar_tile_index_dtm.gpkg",
    "DSM": ROOT_DIR / "lidar_tile_index_dsm.gpkg",
}

TILE_LAYER = "tiles"
SCAN_LAYER = "scans"

# Common CRS for the index. Transects are EPSG:26910 (UTM 10N, metres).
INDEX_CRS = "EPSG:26910"

RASTER_SUFFIXES = {".tif", ".tiff"}

# True = trace the valid-data edge, False = simple bounding box of each raster.
# Bounding boxes are far faster and are usually enough for tile lookup.
VALID_DATA_ONLY = False
# Downsample factor used when tracing valid data (1 = full resolution).
DECIMATION = 8


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


def validate_raster_data(src):
    """Read every compressed block so truncated or corrupt tiles are rejected."""
    for band in src.indexes:
        for _, window in src.block_windows(band):
            src.read(band, window=window)


def scan_year(scan_name):
    """Acquisition year taken from the trailing 4 digits of the scan folder name."""
    match = re.search(r"(\d{4})\D*$", scan_name)
    return int(match.group(1)) if match else None


def find_scans(root, product):
    """Map scan name -> sorted raster paths inside that scan's product subfolder."""
    scans = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        product_dirs = [
            p
            for p in folder.iterdir()
            if p.is_dir() and p.name.lower() == product.lower()
        ]
        rasters = sorted(
            p
            for product_dir in product_dirs
            for p in product_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES
        )
        if rasters:
            scans[folder.name] = rasters
    return scans


def index_scan(scan_name, rasters, product):
    """Footprint records for one scan, grouped by native CRS."""
    year = scan_year(scan_name)
    by_crs = {}
    corrupt = []
    for path in rasters:
        try:
            with rasterio.open(path) as src:
                validate_raster_data(src)

                if src.crs is None:
                    print(f"  skipped (no CRS): {path.name}")
                    continue

                geom = (
                    valid_data_footprint(src) if VALID_DATA_ONLY else box(*src.bounds)
                )
                if geom is None or geom.is_empty:
                    print(f"  skipped (no valid data): {path.name}")
                    continue

                by_crs.setdefault(src.crs, []).append(
                    {
                        "scan": scan_name,
                        "product": product,
                        "year": year,
                        "name": path.name,
                        "path": str(path),
                        "native_crs": src.crs.to_string(),
                        "width": src.width,
                        "height": src.height,
                        "pixel_size_x": abs(src.res[0]),
                        "pixel_size_y": abs(src.res[1]),
                        "nodata": src.nodata,
                        "geometry": geom,
                    }
                )
        except (rasterio.errors.RasterioError, OSError) as exc:
            corrupt.append((path, str(exc)))
            print(f"  CORRUPT - skipped: {path}\n    {exc}")
            continue

    reprojected = []
    for crs, records in by_crs.items():
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        reprojected.append(gdf.to_crs(INDEX_CRS))
    return reprojected, corrupt


def build_index(product, out_gpkg):
    """Index every scan's rasters for one product and write them to out_gpkg."""
    scans = find_scans(ROOT_DIR, product)
    if not scans:
        print(f"{product}: no GeoTIFFs found under {ROOT_DIR}")
        return

    frames = []
    corrupt_tiles = []
    for i, (scan_name, rasters) in enumerate(scans.items(), 1):
        print(f"[{product} {i}/{len(scans)}] {scan_name} ({len(rasters)} rasters)")
        scan_frames, scan_corrupt = index_scan(scan_name, rasters, product)
        frames.extend(scan_frames)
        corrupt_tiles.extend(scan_corrupt)

    if not frames:
        print(f"{product}: no footprints produced.")
        return

    tiles = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=INDEX_CRS
    )

    if out_gpkg.exists():
        out_gpkg.unlink()
    tiles.to_file(out_gpkg, layer=TILE_LAYER, driver="GPKG")
    print(f"Wrote {len(tiles)} tile footprints to {out_gpkg} (layer: {TILE_LAYER})")

    scan_records = [
        {
            "scan": scan_name,
            "product": product,
            "year": scan_year(scan_name),
            "tile_count": len(group),
            "native_crs": group["native_crs"].iloc[0],
            "pixel_size_x": group["pixel_size_x"].iloc[0],
            "geometry": unary_union(group.geometry.values),
        }
        for scan_name, group in tiles.groupby("scan", sort=True)
    ]
    scan_gdf = gpd.GeoDataFrame(scan_records, geometry="geometry", crs=INDEX_CRS)
    scan_gdf.to_file(out_gpkg, layer=SCAN_LAYER, driver="GPKG")
    print(f"Wrote {len(scan_gdf)} scan footprints to {out_gpkg} (layer: {SCAN_LAYER})")

    if corrupt_tiles:
        print(f"WARNING: {len(corrupt_tiles)} corrupt tile(s) were excluded:")
        for path, error in corrupt_tiles:
            print(f"  {path}: {error}")


def main():
    for product, out_gpkg in PRODUCTS.items():
        build_index(product, out_gpkg)


if __name__ == "__main__":
    main()
