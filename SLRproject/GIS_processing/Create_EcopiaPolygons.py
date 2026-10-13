#!/usr/bin/env python3
"""
Polygonize impervious (2,3) and railroad (12) classes by county from tiled GeoTIFFs,
with automatic orientation normalization so rasters can be merged even if some tiles
have positive pixel height (e > 0) or negative pixel width (a < 0).

Usage:
  1) Edit the CONFIG section.
  2) Run: python county_polygonize_cli_fixed.py
"""

# -------------------------
# CONFIG — EDIT THESE VALUES
# -------------------------
ROOT_DIR = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Ecopia_Statewide_3ft_Raster_Tiles\raster"  # Root folder containing GeoTIFF tiles
COUNTY_LIST = [
    "Clallam",
    "Grays_Harbor",
    "Island",
    "Jefferson",
    "King",
    "Kitsap",
    "Mason",
    "Pacific",
    "Pierce",
    "San_Juan",
    "Skagit",
    "Snohomish",
    "Thurston",
    "Wahkiakum",
    "Whatcom",
]
OUTDIR = r"N:\SEA\Kai\Data\CCAP\Ecopia_ImperviousPolygons"  # Output directory

WRITE_SINGLE_GPKG = True  # True: one .gpkg with two layers; False: two .gpkg files

NODATA_OVERRIDE = None  # e.g., 0 if rasters lack NoData; otherwise None
MIN_PIXELS = 4  # e.g., 4 (requires pixel size below)
PIXEL_WIDTH = None  # e.g., 1.0 (map units)
PIXEL_HEIGHT = None  # e.g., 1.0
FIX_INVALID_GEOMETRIES = True

# -------------------------
# END CONFIG
# -------------------------

import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from affine import Affine


def log(msg: str):
    print(f"[landuse] {msg}")


def find_county_tiles(root, county):
    p = Path(root)
    strict = re.compile(rf"WA_land_\d+_{re.escape(county)}_.+\.tif$", re.IGNORECASE)
    files = [str(fp) for fp in p.rglob("*.tif") if strict.search(fp.name)]
    if not files:
        files = [
            str(fp)
            for fp in p.rglob("*.tif")
            if re.search(rf"_{re.escape(county)}_", fp.name, re.IGNORECASE)
        ]
    return sorted(files)


def normalize_orientation_and_wrap(src):
    """
    Ensure src has a north-up, left-to-right transform: a>0, e<0.
    If needed, flip pixels and adjust transform, then return a MemoryFile dataset.
    If already normalized, return None to indicate the original src can be used.
    """
    T: Affine = src.transform
    a, b, c, d, e, f = T.a, T.b, T.c, T.d, T.e, T.f
    need_vert_flip = e > 0
    need_horiz_flip = a < 0

    if not need_vert_flip and not need_horiz_flip:
        return None  # already OK

    data = src.read(1)
    # Apply flips
    if need_vert_flip:
        data = np.flipud(data)
        e = -e
        f = f + src.transform.e * (src.height - 1)  # use original e for shift
    if need_horiz_flip:
        data = np.fliplr(data)
        a = -a
        c = c + src.transform.a * (src.width - 1)  # use original a for shift

    new_T = Affine(a, b, c, d, e, f)

    profile = src.profile.copy()
    profile.update(transform=new_T, height=src.height, width=src.width)
    # Keep dtype, nodata, count, crs as-is
    memfile = MemoryFile()
    mem_ds = memfile.open(**profile)
    mem_ds.write(data, 1)
    # Return both so we can close them later
    return memfile, mem_ds


def mosaic(files):
    """
    Mosaic single-band tiles into one array; returns (array, transform, crs, nodata).
    This version normalizes per-tile orientation before merging.
    """
    if not files:
        raise FileNotFoundError("No input tiles provided to mosaic().")

    srcs = [rasterio.open(f) for f in files]
    crs = srcs[0].crs
    dtype = srcs[0].dtypes[0]

    # Prepare list of datasets to merge: either original srcs or normalized memory datasets
    to_merge = []
    mem_handles = []  # [(memfile, mem_ds), ...] so we can close them

    try:
        for s in srcs:
            if s.crs != crs:
                raise ValueError(
                    "Tiles have mixed CRS. Reproject to a common CRS before mosaicking."
                )
            if s.count != 1:
                raise ValueError(f"Tile {s.name} has {s.count} bands; expected 1.")
            if s.dtypes[0] != dtype:
                log(
                    f"Warning: mixed dtypes ({dtype} vs {s.dtypes[0]}). Proceeding; verify results."
                )

            normalized = normalize_orientation_and_wrap(s)
            if normalized is None:
                to_merge.append(s)  # use original
            else:
                memfile, mem_ds = normalized
                mem_handles.append((memfile, mem_ds))
                to_merge.append(mem_ds)

        # Use nodata from first dataset (or override)
        nodata = srcs[0].nodata

        mosaic_arr, mosaic_transform = merge(
            to_merge, nodata=nodata
        )  # shape: (bands, rows, cols)
        arr = mosaic_arr[0]
        return arr, mosaic_transform, crs, nodata
    finally:
        # Close everything
        for s in srcs:
            s.close()
        for memfile, mem_ds in mem_handles:
            mem_ds.close()
            memfile.close()


def polygonize_mask(mask, transform, crs, class_name, county):
    mask = mask.astype(np.uint8)
    geoms = []
    for geom, value in shapes(mask, transform=transform):
        if value == 1:
            geoms.append(shape(geom))
    if not geoms:
        return gpd.GeoDataFrame(columns=["class", "county", "geometry"], crs=crs)
    gdf = gpd.GeoDataFrame(
        {"class": [class_name] * len(geoms), "county": [county] * len(geoms)},
        geometry=geoms,
        crs=crs,
    )
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf


def fix_invalid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def _fix(geom: BaseGeometry):
        if geom is None:
            return None
        try:
            return geom if geom.is_valid else geom.buffer(0)
        except Exception:
            return geom

    gdf["geometry"] = gdf["geometry"].apply(_fix)
    return gdf[~gdf.geometry.is_empty]


def run(
    root,
    county,
    outdir,
    single_gpkg=False,
    gpkg_name=None,
    min_pixels=None,
    pixel_size=None,
    nodata_override=None,
    fix_invalid_geoms=True,
):
    files = find_county_tiles(root, county)
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs found for county '{county}' under {root}")
    log(f"Found {len(files)} tiles for {county}. Mosaicking...")
    arr, transform, crs, nodata = mosaic(files)

    if nodata_override is not None:
        nodata = nodata_override
        log(f"Using NODATA_OVERRIDE={nodata_override}")

    log("Building masks...")
    impervious_mask = np.isin(arr, [2, 3])
    railroad_mask = arr == 12
    if nodata is not None:
        impervious_mask &= arr != nodata
        railroad_mask &= arr != nodata

    log("Polygonizing impervious...")
    gdf_imp = polygonize_mask(impervious_mask, transform, crs, "impervious", county)
    log(f"Impervious polygons: {len(gdf_imp)}")

    log("Polygonizing railroad...")
    gdf_rail = polygonize_mask(railroad_mask, transform, crs, "railroad", county)
    log(f"Railroad polygons: {len(gdf_rail)}")

    if min_pixels and pixel_size:
        px_area = pixel_size[0] * pixel_size[1]
        thr_area = min_pixels * px_area
        before_imp, before_rail = len(gdf_imp), len(gdf_rail)
        gdf_imp = gdf_imp[gdf_imp.area >= thr_area]
        gdf_rail = gdf_rail[gdf_rail.area >= thr_area]
        log(
            f"Size filter (>= {min_pixels} px => {thr_area} map-units^2): "
            f"impervious {before_imp} -> {len(gdf_imp)}, railroad {before_rail} -> {len(gdf_rail)}"
        )
    elif min_pixels and not pixel_size:
        log(
            "Warning: MIN_PIXELS set but PIXEL_WIDTH/PIXEL_HEIGHT not provided; skipping size filter."
        )

    if fix_invalid_geoms:
        log("Fixing invalid geometries (buffer(0))...")
        gdf_imp = fix_invalid(gdf_imp)
        gdf_rail = fix_invalid(gdf_rail)
        log(f"After fix: impervious={len(gdf_imp)}, railroad={len(gdf_rail)}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if single_gpkg:
        gpkg = outdir / (gpkg_name or f"{county}_landuse.gpkg")
        if gpkg.exists():
            gpkg.unlink()
        gdf_imp.to_file(gpkg, driver="GPKG", layer="impervious")
        gdf_rail.to_file(gpkg, driver="GPKG", layer="railroad")
        log(f"Wrote two layers to {gpkg}")
    else:
        gpkg_imp = outdir / f"{county}_impervious.gpkg"
        gpkg_rail = outdir / f"{county}_railroad.gpkg"
        if gpkg_imp.exists():
            gpkg_imp.unlink()
        if gpkg_rail.exists():
            gpkg_rail.unlink()
        gdf_imp.to_file(gpkg_imp, driver="GPKG")
        gdf_rail.to_file(gpkg_rail, driver="GPKG")
        log(f"Wrote {gpkg_imp} and {gpkg_rail}")


#################### MAIN ##########################
px = (PIXEL_WIDTH, PIXEL_HEIGHT) if (PIXEL_WIDTH and PIXEL_HEIGHT) else None

for COUNTY in COUNTY_LIST:
    run(
        ROOT_DIR,
        COUNTY,
        OUTDIR,
        single_gpkg=WRITE_SINGLE_GPKG,
        gpkg_name=f"{COUNTY}_landuse.gpkg",
        min_pixels=MIN_PIXELS,
        pixel_size=px,
        nodata_override=NODATA_OVERRIDE,
        fix_invalid_geoms=FIX_INVALID_GEOMETRIES,
    )
