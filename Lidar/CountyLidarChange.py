"""Difference the oldest and newest lidar scans over a region, pixel by pixel.

Follows the same index-driven workflow as TransectLidarProfiles.py, but works
on a raster grid instead of transect samples. The tile index is queried for
tiles overlapping the trimming polygon layer, which are mosaicked into an
oldest-available and a newest-available surface, and the two are differenced on
a common grid.

Both surfaces are built per pixel rather than per acquisition: each pixel takes
its value from the earliest (or latest) scan that covers it, so an area served
by several partially overlapping scans still gets full coverage instead of
being limited to the two extreme years.

The analysis region is the trimming polygon clipped to the lidar footprint, so
a coastal-only lidar strip produces a coastal-only grid rather than one that is
mostly nodata. Pixels whose oldest and newest scans are the same acquisition
are excluded.

Outputs:
  <region>_change.tif   - newest minus oldest elevation, in metres
  <region>_change.gpkg  - the difference classified into negative / no
                          significant / positive change and polygonized
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio import features
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

INDEX_GPKG = Path(r"D:\Data\Lidar\lidar_tile_index_dtm.gpkg")
TILE_LAYER = "tiles"

# Surface-water returns tagged by BuildWaterMaskIndex.py. Lidar flown at high
# tide images the sea surface as terrain, which would otherwise difference
# against a low-tide scan as metres of fake elevation change.
WATER_MASK_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
WATER_TILE_LAYER = "water_tiles"
APPLY_WATER_MASK = True

REGION_LAYER = "barrier_region"

# Select "county" for the standard county boundaries or "outer" for the
# expanded coastal boundaries.
MODE = "PugetSound"


MODE_CONFIGS = {
    "PugetSound": {
        "region_name": "thurston",
        "region_root": Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4"),
        "output_root": Path(r"D:\Data\Lidar\county_change"),
        "max_pixels": 2_000_000_000,
    },
    "outer": {
        "region_name": "pacific",
        "region_root": Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_Outer"),
        "output_root": Path(r"D:\Data\Lidar\county_change_outer"),
        "max_pixels": 10_000_000_000,
    },
}

if MODE not in MODE_CONFIGS:
    raise SystemExit(f"Unknown MODE {MODE!r}; choose from {sorted(MODE_CONFIGS)}")

MODE_CONFIG = MODE_CONFIGS[MODE]
# Label used for output file names and the exported attributes.
REGION_NAME = MODE_CONFIG["region_name"]
OUT_DIR = MODE_CONFIG["output_root"] / REGION_NAME

# Polygon layer the analysis is trimmed to; all its features are dissolved.
REGION_GPKG = MODE_CONFIG["region_root"] / REGION_NAME / "county_data.gpkg"


# Grid the composites are built on. Metres, in the index CRS.
OUTPUT_CRS = "EPSG:26910"
TARGET_RES = 1

# Refuse to build grids larger than this; raise TARGET_RES for big counties.
MAX_PIXELS = MODE_CONFIG["max_pixels"]

# Elevation change smaller than this (metres) is treated as noise.
ERROR_TOLERANCE = 0.5

# A pixel needs two scans at least this many years apart to be differenced.
MIN_YEAR_GAP = 1

# Convert sampled elevations to metres using the tile's horizontal linear unit.
CONVERT_Z_TO_METRES = True

# Elevations outside this range (metres) are undeclared nodata sentinels such
# as -9999 or -3.4e38, which would otherwise resample into huge fake changes.
# Applied to the source pixels before resampling, so sentinels can never be
# blended into a plausible-looking elevation.
VALID_Z_RANGE = (-200.0, 5000.0)

# Fraction of a destination pixel's resampling kernel that must land on valid
# source pixels. 1.0 means a pixel is dropped as soon as the kernel touches
# nodata or the tile edge, which is what stops the seaward edge of a tile from
# being extrapolated outward. Lower it only to trade edge purity for coverage.
MIN_SOURCE_WEIGHT = 1.0 - 1e-6

# Also write the two epoch composites, which makes bad mosaics obvious in GIS.
WRITE_COMPOSITES = True

# Drop classified polygons smaller than this before export. 0 = keep all.
MIN_POLYGON_AREA_M2 = 0.0

RESAMPLING = Resampling.bilinear

CLASS_LABELS = {
    -1: "negative change",
    0: "no significant change",
    1: "positive change",
}

YEAR_PATTERN = re.compile(r"(\d{4})")


def scan_year(scan_name):
    years = YEAR_PATTERN.findall(str(scan_name))
    return int(years[-1]) if years else None


def z_scale(crs):
    """Factor converting the tile's vertical units to metres."""
    if not CONVERT_Z_TO_METRES:
        return 1.0
    axis = CRS.from_user_input(crs).axis_info
    return axis[0].unit_conversion_factor if axis else 1.0


def target_grid(bounds):
    """Pixel-aligned transform and shape covering the given bounds."""
    left, bottom, right, top = bounds
    left = np.floor(left / TARGET_RES) * TARGET_RES
    bottom = np.floor(bottom / TARGET_RES) * TARGET_RES
    right = np.ceil(right / TARGET_RES) * TARGET_RES
    top = np.ceil(top / TARGET_RES) * TARGET_RES

    width = int(np.rint((right - left) / TARGET_RES))
    height = int(np.rint((top - bottom) / TARGET_RES))
    transform = rasterio.transform.from_origin(left, top, TARGET_RES, TARGET_RES)
    return transform, (height, width)


def read_tile(src):
    """Band 1 in metres, with declared nodata, masks and sentinels set to NaN."""
    band = src.read(1, masked=True).astype("float32").filled(np.nan)
    band *= z_scale(src.crs)
    low, high = VALID_Z_RANGE
    with np.errstate(invalid="ignore"):
        band[~np.isfinite(band) | (band < low) | (band > high)] = np.nan
    return band


def select_tiles(tiles, geometry):
    """Tiles whose footprint intersects the geometry, via the spatial index."""
    hits = np.unique(tiles.sindex.query(geometry, predicate="intersects"))
    return tiles.iloc[hits]


def water_geometries(water, scan_name, bounds):
    """This scan's water polygons that fall inside the given grid bounds."""
    group = water.get(scan_name)
    if group is None or group.empty:
        return []
    hits = np.unique(group.sindex.query(box(*bounds), predicate="intersects"))
    return list(group.geometry.iloc[hits])


def build_composite(epoch_tiles, transform, shape_hw, water):
    """Mosaic tiles onto the target grid, first tile to cover a pixel wins.

    Tiles must already be in priority order, so passing them oldest-first
    builds the oldest-available surface and newest-first the newest. Also
    returns the acquisition year behind every filled pixel.
    """
    mosaic = np.full(shape_hw, np.nan, dtype="float32")
    years = np.zeros(shape_hw, dtype="int16")
    height, width = shape_hw
    used = 0
    wet_pixels = 0

    for path, year, scan_name in zip(
        epoch_tiles["path"], epoch_tiles["year"], epoch_tiles["scan"]
    ):
        try:
            src = rasterio.open(path)
        except rasterio.errors.RasterioIOError as exc:
            print(f"    skipped ({exc}): {Path(path).name}")
            continue

        with src:
            if src.crs is None:
                print(f"    skipped (no CRS): {Path(path).name}")
                continue
            dst_bounds = transform_bounds(
                src.crs, OUTPUT_CRS, *src.bounds, densify_pts=21
            )
            window = from_bounds(*dst_bounds, transform=transform)
            window = window.round_offsets().round_lengths()
            # Pad by a pixel so resampling at the tile edge has neighbours.
            col_off = max(0, int(window.col_off) - 1)
            row_off = max(0, int(window.row_off) - 1)
            col_end = min(width, int(window.col_off + window.width) + 1)
            row_end = min(height, int(window.row_off + window.height) + 1)
            if col_end <= col_off or row_end <= row_off:
                continue

            window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
            dst_transform = rasterio.windows.transform(window, transform)
            source = read_tile(src)
            src_crs = src.crs
            src_transform = src.transform

        buffer = np.full((window.height, window.width), np.nan, dtype="float32")
        reproject(
            source=source,
            destination=buffer,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=OUTPUT_CRS,
            dst_nodata=np.nan,
            resampling=RESAMPLING,
        )

        # Resample the validity mask through the same kernel. A destination
        # pixel whose weight is short of 1 drew on nodata or fell outside the
        # tile, so it is an extrapolated edge value rather than a measurement.
        weight = np.full((window.height, window.width), -1.0, dtype="float32")
        reproject(
            source=np.isfinite(source).astype("float32"),
            destination=weight,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=-1.0,
            dst_transform=dst_transform,
            dst_crs=OUTPUT_CRS,
            dst_nodata=-1.0,
            resampling=RESAMPLING,
        )
        buffer[~(weight >= MIN_SOURCE_WEIGHT)] = np.nan

        geoms = water_geometries(
            water, scan_name, rasterio.windows.bounds(window, transform)
        )
        if geoms:
            wet = features.geometry_mask(
                geoms,
                out_shape=(int(window.height), int(window.width)),
                transform=dst_transform,
                invert=True,
            )
            wet &= np.isfinite(buffer)
            wet_pixels += int(wet.sum())
            buffer[wet] = np.nan

        patch = mosaic[row_off:row_end, col_off:col_end]
        fill = np.isnan(patch) & np.isfinite(buffer)
        if not fill.any():
            continue
        patch[fill] = buffer[fill]
        years[row_off:row_end, col_off:col_end][fill] = int(year)
        used += 1

    return mosaic, years, used, wet_pixels


def classify(diff):
    """Change classes stored offset by +1 (0/1/2), with 255 marking nodata."""
    classes = np.full(diff.shape, 255, dtype="uint8")
    valid = np.isfinite(diff)
    classes[valid & (diff < -ERROR_TOLERANCE)] = 0
    classes[valid & (np.abs(diff) <= ERROR_TOLERANCE)] = 1
    classes[valid & (diff > ERROR_TOLERANCE)] = 2
    return classes


def polygonize(classes, transform):
    """Dissolve the classified raster into one multipolygon per class."""
    mask = classes != 255
    records = [
        {"class_id": int(value) - 1, "geometry": shape(geom)}
        for geom, value in features.shapes(classes, mask=mask, transform=transform)
        if value != 255
    ]
    if not records:
        return None

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=OUTPUT_CRS)
    if MIN_POLYGON_AREA_M2 > 0:
        gdf = gdf[gdf.geometry.area >= MIN_POLYGON_AREA_M2]
        if gdf.empty:
            return None
    gdf = gdf.dissolve(by="class_id", as_index=False)
    gdf["change_class"] = gdf["class_id"].map(CLASS_LABELS)
    gdf["area_m2"] = gdf.geometry.area
    return gdf[["class_id", "change_class", "area_m2", "geometry"]]


def write_raster(path, data, transform, nodata=np.nan):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=OUTPUT_CRS,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dst:
        for _, window in dst.block_windows(1):
            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            col_start = int(window.col_off)
            col_end = col_start + int(window.width)
            dst.write(data[row_start:row_end, col_start:col_end], 1, window=window)


def process_region(name, region_geom, tiles, water):
    hits = select_tiles(tiles, region_geom)
    hits = hits[hits["year"].notna()]
    if hits.empty:
        print("  no dated lidar tiles overlap this region")
        return

    years = sorted(int(y) for y in hits["year"].unique())
    if len(years) < 2:
        print(f"  only {len(years)} year(s) available: {years}")
        return
    print(f"  {len(hits)} tiles spanning years {years}")
    crs_list = sorted(hits["native_crs"].dropna().unique())
    print(f"    native CRS in index: {crs_list} -> reprojected to {OUTPUT_CRS}")

    # Analysis region: trimming polygon clipped to wherever any lidar exists.
    region = region_geom.intersection(hits.geometry.union_all())
    if region.is_empty:
        print("  no overlap between the trimming region and the lidar coverage")
        return
    print(f"  analysis region: {region.area / 1e6:.1f} km2")

    transform, shape_hw = target_grid(region.bounds)
    if shape_hw[0] * shape_hw[1] > MAX_PIXELS:
        print(f"  grid {shape_hw} exceeds MAX_PIXELS; raise TARGET_RES")
        return
    print(f"  grid {shape_hw[1]}x{shape_hw[0]} at {TARGET_RES} m")

    hits = select_tiles(hits, region)
    inside = features.geometry_mask(
        [region], out_shape=shape_hw, transform=transform, invert=True
    )

    composites = {}
    year_grids = {}
    # Oldest-first fills each pixel from the earliest scan covering it, and
    # newest-first from the latest, so coverage gaps fall back to other years.
    for label, ascending in (("old", True), ("new", False)):
        ordered = hits.sort_values(["year", "path"], ascending=[ascending, True])
        composite, grid_years, used, wet_pixels = build_composite(
            ordered, transform, shape_hw, water
        )
        composite[~inside] = np.nan
        grid_years[~inside] = 0
        valid = int(np.isfinite(composite).sum())
        present = sorted(int(y) for y in np.unique(grid_years) if y)
        print(
            f"  {label} composite: {used}/{len(ordered)} tiles, "
            f"{valid} valid pixels, {wet_pixels} dropped as water, "
            f"years used {present}"
        )
        if valid == 0:
            print("    composite is empty; check tile paths and CRS")
            return
        composites[label] = composite
        year_grids[label] = grid_years

    gap = year_grids["new"].astype("int32") - year_grids["old"].astype("int32")
    diff = composites["new"] - composites["old"]
    # A pixel covered by only one scan would otherwise difference against itself.
    diff[gap < MIN_YEAR_GAP] = np.nan

    finite = np.isfinite(diff)
    if not finite.any():
        print(f"  no pixel has two scans at least {MIN_YEAR_GAP} year(s) apart")
        return
    lo, mid, hi = np.percentile(diff[finite], [2, 50, 98])
    print(
        f"  diff over {finite.sum()} pixels: median {mid:.2f} m, "
        f"2-98% {lo:.2f} to {hi:.2f} m"
    )

    slug = re.sub(r"\W+", "_", name).strip("_").lower()
    stem = f"{slug}_change_{years[0]}_{years[-1]}"

    if WRITE_COMPOSITES:
        for label, composite in composites.items():
            write_raster(OUT_DIR / f"{slug}_dem_{label}.tif", composite, transform)
            write_raster(
                OUT_DIR / f"{slug}_year_{label}.tif", year_grids[label], transform, 0
            )

    raster_path = OUT_DIR / f"{stem}.tif"
    write_raster(raster_path, diff, transform)
    print(f"  wrote {raster_path.name}")

    polygons = polygonize(classify(diff), transform)
    if polygons is None:
        print("  no classified polygons produced")
        return
    polygons.insert(0, "region", name)
    polygons["old_year"] = years[0]
    polygons["new_year"] = years[-1]
    polygons["tolerance_m"] = ERROR_TOLERANCE

    gpkg_path = OUT_DIR / f"{stem}.gpkg"
    polygons.to_file(gpkg_path, layer="change_classes", driver="GPKG")
    print(f"  wrote {gpkg_path.name} ({len(polygons)} classes)")


def load_water_masks(crs):
    """Water polygons grouped by scan, each group carrying its own index."""
    if not APPLY_WATER_MASK:
        return {}
    if not WATER_MASK_GPKG.exists():
        print(f"WARNING: {WATER_MASK_GPKG} not found; run BuildWaterMaskIndex.py")
        return {}

    water = gpd.read_file(WATER_MASK_GPKG, layer=WATER_TILE_LAYER).to_crs(crs)
    groups = {name: group for name, group in water.groupby("scan", sort=False)}
    print(f"water mask: {len(water)} polygons across {len(groups)} scans")
    return groups


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tiles = gpd.read_file(INDEX_GPKG, layer=TILE_LAYER)
    if tiles.crs is None:
        raise SystemExit(f"{INDEX_GPKG} layer {TILE_LAYER!r} has no CRS")
    print(f"tile index CRS {tiles.crs.to_string()} -> {OUTPUT_CRS}")
    tiles = tiles.to_crs(OUTPUT_CRS)
    if "year" not in tiles.columns:
        tiles["year"] = tiles["scan"].map(scan_year)
    tiles["year"] = pd.to_numeric(tiles["year"], errors="coerce")

    region_gdf = gpd.read_file(REGION_GPKG, layer=REGION_LAYER)
    if region_gdf.empty:
        raise SystemExit(f"{REGION_GPKG} layer {REGION_LAYER!r} is empty")
    if region_gdf.crs is None:
        raise SystemExit(f"{REGION_GPKG} layer {REGION_LAYER!r} has no CRS")
    region_gdf = region_gdf.to_crs(OUTPUT_CRS)

    geometry = region_gdf.geometry.union_all()
    if geometry.is_empty:
        raise SystemExit(f"layer {REGION_LAYER!r} has empty geometry")

    print(
        f"{REGION_NAME}: trimming to {len(region_gdf)} feature(s) from "
        f"{REGION_LAYER!r} ({geometry.area / 1e6:.1f} km2), "
        f"{len(tiles)} tiles in index"
    )
    process_region(REGION_NAME, geometry, tiles, load_water_masks(OUTPUT_CRS))


if __name__ == "__main__":
    main()
