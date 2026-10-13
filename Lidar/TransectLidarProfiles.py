"""Extract per-scan elevation profiles along transects using the lidar tile index.

Reads the tile index written by BuildLidarTileIndex.py plus a transect layer,
then for every transect samples a continuous elevation profile from each lidar
scan that covers it. A transect therefore yields one profile per overlapping
scan, letting the scans be compared against each other over time.

Output is a long-format table: one row per (transect, scan, distance along
transect) with the sampled elevation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from shapely import get_coordinates, line_interpolate_point

INDEX_GPKG = Path(r"D:\Data\Lidar\lidar_tile_index_dtm.gpkg")
TILE_LAYER = "tiles"
SCAN_LAYER = "scans"

# Surface-water returns tagged by BuildWaterMaskIndex.py. Lidar flown at high
# tide images the sea surface as terrain, which would otherwise appear as
# metres of fake elevation change between scans.
WATER_MASK_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
WATER_SCAN_LAYER = "water_scans"
APPLY_WATER_MASK = True
# False keeps the samples and exports a "water" flag column instead, for
# auditing the mask before trusting it.
DROP_WATER = False

REGION_NAME = "jefferson"


TRANSECT_GPKG = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Counties"
)
TRANSECT_GPKG = TRANSECT_GPKG / REGION_NAME / "transects.gpkg"

TRANSECT_LAYER = "transects_all"
# Columns that uniquely identify a transect; carried through to the output.
TRANSECT_ID_COLS = ["path_id", "sample_id"]


OUT_DIR = Path(r"D:\Data\Lidar\county_change")
OUT_DIR = OUT_DIR / REGION_NAME / 'profiles'

# Sample spacing along each transect, in transect CRS units (metres).
SAMPLE_SPACING = 1.0
# Drop a scan's profile for a transect unless at least this fraction of the
# samples returned data. Keeps partially covered edge transects out.
MIN_COVERAGE = 0.5
# Convert sampled elevations to metres using the scan's horizontal linear unit.
# WA State Plane tiles are in US survey feet; set False to keep native units.
CONVERT_Z_TO_METRES = True

# Rows of a raster read at once when sampling. Bounds peak memory.
ROW_BLOCK = 2048


def load_transects():
    gdf = gpd.read_file(TRANSECT_GPKG, layer=TRANSECT_LAYER)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    missing = [c for c in TRANSECT_ID_COLS if c not in gdf.columns]
    if missing:
        raise SystemExit(f"Transect layer is missing id columns: {missing}")
    return gdf.reset_index(drop=True)


def build_sample_points(geometries, rows):
    """Densify each transect into evenly spaced sample points.

    Returns a flat table of sample points carrying the owning transect's row
    label and the distance along that transect.
    """
    lengths = np.asarray([g.length for g in geometries])
    counts = np.maximum(2, np.floor(lengths / SAMPLE_SPACING).astype(int) + 1)

    positions = np.repeat(np.arange(len(lengths)), counts)
    # Distance along each transect: 0, spacing, 2*spacing, ... clipped to length.
    offsets = np.concatenate([np.arange(n) for n in counts]) * SAMPLE_SPACING
    distances = np.minimum(offsets, lengths[positions])

    points = line_interpolate_point(np.asarray(geometries)[positions], distances)
    coords = get_coordinates(points)

    return pd.DataFrame(
        {
            "row": np.asarray(rows)[positions],
            "dist_m": distances,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )


def z_scale(crs):
    """Factor converting the scan's vertical units to metres."""
    if not CONVERT_Z_TO_METRES:
        return 1.0
    axis = CRS.from_user_input(crs).axis_info
    return axis[0].unit_conversion_factor if axis else 1.0


def sample_raster(path, xs, ys):
    """Sample a raster at native-CRS coordinates, returning NaN outside data."""
    out = np.full(xs.shape, np.nan, dtype="float64")
    with rasterio.open(path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, op=np.floor)
        rows = np.asarray(rows, dtype="int64")
        cols = np.asarray(cols, dtype="int64")

        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if not inside.any():
            return out

        order = np.argsort(rows[inside], kind="stable")
        idx = np.flatnonzero(inside)[order]

        for start in range(rows[idx].min(), rows[idx].max() + 1, ROW_BLOCK):
            stop = min(start + ROW_BLOCK, src.height)
            block = idx[(rows[idx] >= start) & (rows[idx] < stop)]
            if block.size == 0:
                continue
            window = rasterio.windows.Window(0, start, src.width, stop - start)
            data = src.read(1, window=window, masked=True).astype("float64")
            values = data[rows[block] - start, cols[block]]
            out[block] = values.filled(np.nan)
    return out


def sample_scan(scan_tiles, samples, transect_crs):
    """Sample every tile of one scan and merge into a single value array."""
    values = np.full(len(samples), np.nan, dtype="float64")
    sx = samples["x"].to_numpy()
    sy = samples["y"].to_numpy()

    # A scan can mix projections between tiles, so reproject per native CRS.
    for native_crs, group in scan_tiles.groupby("native_crs", sort=False):
        transformer = Transformer.from_crs(transect_crs, native_crs, always_xy=True)
        xs, ys = transformer.transform(sx, sy)
        bounds = group.to_crs(native_crs).geometry.bounds.to_numpy()
        scale = z_scale(native_crs)

        for (left, bottom, right, top), path in zip(bounds, group["path"]):
            pending = np.flatnonzero(np.isnan(values))
            if pending.size == 0:
                return values
            in_tile = pending[
                (xs[pending] >= left)
                & (xs[pending] < right)
                & (ys[pending] >= bottom)
                & (ys[pending] < top)
            ]
            if in_tile.size == 0:
                continue
            values[in_tile] = sample_raster(path, xs[in_tile], ys[in_tile]) * scale

    return values


def load_water_masks(crs):
    if not APPLY_WATER_MASK:
        return {}
    if not WATER_MASK_GPKG.exists():
        print(f"WARNING: {WATER_MASK_GPKG} not found; run BuildWaterMaskIndex.py")
        return {}

    water = gpd.read_file(WATER_MASK_GPKG, layer=WATER_SCAN_LAYER).to_crs(crs)
    print(f"water mask: {len(water)} scans")
    return dict(zip(water["scan"], water.geometry))


def water_flags(water, scan_name, samples, crs):
    """Boolean per sample marking points that fall on this scan's water mask."""
    flags = np.zeros(len(samples), dtype=bool)
    geom = water.get(scan_name)
    if geom is None or geom.is_empty:
        return flags
    points = gpd.GeoSeries(gpd.points_from_xy(samples["x"], samples["y"]), crs=crs)
    flags[points.sindex.query(geom, predicate="intersects")] = True
    return flags


OUT_DIR.mkdir(parents=True, exist_ok=True)

tiles = gpd.read_file(INDEX_GPKG, layer=TILE_LAYER)
scans = gpd.read_file(INDEX_GPKG, layer=SCAN_LAYER)
transects = load_transects()
print(f"{len(transects)} transects, {len(tiles)} tiles, {len(scans)} scans")

tiles = tiles.to_crs(transects.crs)
scans = scans.to_crs(transects.crs)
ids = transects[TRANSECT_ID_COLS]
water = load_water_masks(transects.crs)

out_cols = TRANSECT_ID_COLS + ["scan", "dist_m", "x", "y", "z"]
if not DROP_WATER:
    out_cols.append("water")

written = []
for scan_name in scans["scan"]:
    scan_tiles = tiles[tiles["scan"] == scan_name]
    if scan_tiles.empty:
        continue

    # Only transects touching one of this scan's tiles are worth reading.
    hit_rows = np.unique(
        gpd.sjoin(transects[["geometry"]], scan_tiles[["geometry"]]).index.to_numpy()
    )
    if hit_rows.size == 0:
        continue

    hits = transects.loc[hit_rows]
    scan_samples = build_sample_points(hits.geometry.to_numpy(), hit_rows)

    print(
        f"{scan_name}: {hit_rows.size} transects, "
        f"{len(scan_samples)} samples, {len(scan_tiles)} tiles"
    )
    z = sample_scan(scan_tiles, scan_samples, transects.crs)

    wet = water_flags(water, scan_name, scan_samples, transects.crs)
    if wet.any():
        print(f"  {wet.sum()} of {wet.size} samples fall on water")

    # Masking before the coverage test drops transects that are mostly water.
    profile = scan_samples.assign(
        scan=scan_name, z=np.where(wet, np.nan, z) if DROP_WATER else z, water=wet
    )
    coverage = profile.groupby("row")["z"].transform(lambda s: s.notna().mean())
    profile = profile[coverage >= MIN_COVERAGE]
    profile = profile[profile["z"].notna()]
    if profile.empty:
        print("  no covered transects after filtering")
        continue

    profile = profile.join(ids, on="row")
    out_path = OUT_DIR / f"{scan_name}_profiles.parquet"
    profile[out_cols].to_parquet(out_path, index=False)
    written.append(out_path)
    print(f"  wrote {len(profile)} rows to {out_path.name}")

print(f"Wrote {len(written)} scan profile files to {OUT_DIR}")
