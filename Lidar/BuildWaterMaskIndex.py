"""Tag surface-water returns in each lidar scan and publish them as a mask index.

Lidar flown at different tide stages images the sea surface as if it were
topography, so differencing two scans turns the tidal offset into a large fake
elevation change. This script finds those returns once, per scan, and writes
them out as polygons that both CountyLidarChange.py and TransectLidarProfiles.py
can consume.

Detection keys off the fact that every water return within one scan sits at
essentially the same NAVD88 elevation - the tide level at the time of flight.
Rather than classify pixels individually (noisy, because raw water returns are
speckled and full of dropouts), each scan is reduced to a single water-plane
elevation:

  1. Read every tile heavily decimated with average resampling. Averaging
     collapses the speckle, so water becomes genuinely flat while the beach
     face keeps its slope.
  2. Keep only cells whose gradient magnitude is near zero and whose elevation
     falls in a plausible tidal band.
  3. Histogram those flat low elevations. Water contributes one dominant spike;
     its location is the water plane.
  4. Measure the scan's water-surface noise at mask resolution inside the
     reference core, and set the tolerance above the plane from it.
  5. Mask everything at or below that plane plus the tolerance, close small
     dry pits and gaps without moving the mask's outer edge, sieve out specks,
     and polygonize.
  6. Keep only mask regions connected, across the whole scan, to the core, so
     low ground that is merely below the plane is not taken for water.

A reference layer of visible surface water sharpens every one of those steps.
Because it does not register exactly against the lidar shoreline it is never
used as a hard boundary; instead it is eroded inward to an always-wet core and
dilated outward to a permissive limit. The core seeds the histogram, so a broad
but genuinely flat tidal flat can no longer be mistaken for the water plane,
supplies the noise sample for the tolerance, anchors the connectivity test,
and is masked unconditionally in every scan. The limit bounds where detection
is allowed to mask at all, keeping low-lying inland ground out of it.

Scans with no convincing peak fall back to FALLBACK_WATER_Z so they are still
over-masked rather than silently passed through. WATER_Z_OVERRIDES lets a known
water level be forced in for any scan.

Output GeoPackage layers:
  water_tiles - one water polygon per tile, for windowed raster lookup
  water_scans - the dissolved water mask per scan, plus the detection stats
"""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import CRS
from rasterio import features
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import unary_union

INDEX_GPKG = Path(r"D:\Data\Lidar\lidar_tile_index_dtm.gpkg")
TILE_LAYER = "tiles"

OUT_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
TILE_MASK_LAYER = "water_tiles"
SCAN_MASK_LAYER = "water_scans"

# Restrict the run to these scan names while tuning. Empty = every scan.
ONLY_SCANS = ()

# Force a known water level (metres NAVD88) for a scan, bypassing detection.
WATER_Z_OVERRIDES = {}

# Reference layer of visible surface water. Optional, but it is what stops a
# broad, genuinely flat tidal flat from being read as the water plane.
# Set to None to run on elevation statistics alone.
REFERENCE_WATER = Path(r"D:\Data\GIS_general\VisibleSurfaceWater_OpenWater.shp")
# Eroded inward to an always-wet core, enough to absorb the misregistration
# between the reference shoreline and the lidar one.
SEED_EROSION_M = 30.0
# Dilated outward to bound where masking may happen. Generous, so a
# registration offset can never exclude real water.
LIMIT_DILATION_M = 150.

# Draw the histogram only from the core, rather than from every flat low cell.
SEED_HISTOGRAM = True
# Refuse to mask anything outside the dilated limit.
CONFINE_TO_REFERENCE = True
# Mask the core in every scan regardless of its elevation.
ALWAYS_MASK_REFERENCE_CORE = True

# Vertex tolerance applied to the reference before buffering. The layer is only
# ever used eroded by tens of metres, so this is far below its working accuracy,
# and statewide hydrography carries orders of magnitude more detail than that.
REFERENCE_SIMPLIFY_M = 2.0
# Buffer smoothness. Low is fine for a mask that is itself approximate.
BUFFER_QUAD_SEGS = 2

# Elevations are metres NAVD88 throughout.
# Band searched for the water plane. Must bracket the local tidal range but
# stay below anything that could be real upland terrain.
SEARCH_RANGE = (-2.0, 4.0)

# Gradient magnitude (rise/run) below which an averaged cell counts as flat.
FLAT_SLOPE = 0.02
# Grid the water-plane histogram is built on. Coarse enough to average out
# speckle, fine enough that the beach face still registers as sloped.
HIST_RES_M = 10.0
HIST_BIN_M = 0.05

# Half-width of the window around the peak bin used to refine the estimate.
PEAK_WINDOW_M = 0.25
# Share of the flat low-lying cells that must fall within PEAK_WINDOW_M of the
# dominant bin. Measured over the window, not a single bin, so it stays tied to
# peak sharpness rather than to HIST_BIN_M. Calibrated against the coastal
# reference: real water planes score 0.18-0.63, topobathy (which maps the bed,
# so has no water surface) scores 0.08.
MIN_PEAK_FRACTION = 0.15
MIN_PEAK_CELLS = 200

# Reject a pool with more than this share of its mass against either end of
# SEARCH_RANGE: the distribution is truncated, so its mode means nothing. A
# correct reference keeps this near zero; it firing means something is wrong.
MAX_EDGE_FRACTION = 0.25
EDGE_GUARD_M = 0.10

# Cap on tiles read for the histogram; scans with more are strided.
MAX_HIST_TILES = 400
# Reject a detected plane above this; a higher peak is more likely a tidal flat.
# Must sit above local MHHW, which is about 3.4 m NAVD88 in Puget Sound.
MAX_WATER_Z = 4.0

# Used when detection fails or is rejected. Set near the local low-water datum.
FALLBACK_WATER_Z = 0.0

# Raised above the water plane before masking, to catch wave crests and the
# swash edge. Larger values over-mask. Used as-is when ADAPTIVE_TOLERANCE is off
# or the noise cannot be measured (fallback level, no reference, too few cells).
Z_TOLERANCE = 0.1
# Measure the tolerance per scan instead: the median offset of MASK_RES_M core
# cells from the plane plus TOLERANCE_SIGMAS robust standard deviations
# (1.4826 * MAD), clamped to TOLERANCE_RANGE. Calm scans get a tight cut and
# choppy ones a looser one.
ADAPTIVE_TOLERANCE = True
TOLERANCE_SIGMAS = 2.0
TOLERANCE_RANGE = (0.05, 0.5)
# Core cells further than this from the plane are piers, boats or misregistered
# land rather than water, and are left out of the noise estimate.
TOLERANCE_OUTLIER_M = 1.0
# Cap on core tiles read at MASK_RES_M for the estimate, and the samples needed.
MAX_TOLERANCE_TILES = 40
MIN_TOLERANCE_CELLS = 1000

# Grid the mask itself is rasterized on, before polygonizing.
MASK_RES_M = 1.0
# Morphological closing radius. Fills dry pits (wave crests above the cut,
# dropouts) and gaps narrower than twice this without moving the outer edge of
# the mask, so it does not eat into the beach the way an outward buffer does.
CLOSING_M = 5.0
# Connected mask regions smaller than this many cells are dropped as noise.
MIN_MASK_PIXELS = 50
# Keep only mask regions connected, across tile seams, to the reference core.
# Drops isolated low ground (backshore hollows, ponds) that sits below the plane.
REQUIRE_CORE_CONNECTION = True

# Undeclared nodata sentinels such as -9999 or -3.4e38.
VALID_Z_RANGE = (-200.0, 5000.0)

# Tiles in WA State Plane are US survey feet; False keeps native units.
CONVERT_Z_TO_METRES = True


def linear_unit(crs):
    axis = CRS.from_user_input(crs).axis_info
    return axis[0].unit_conversion_factor if axis else 1.0


class ReferenceWater:
    """Reference water polygons held at two buffers and cached per raster CRS.

    'core' is the layer eroded inward, so every cell inside it is water no
    matter how the reference is registered. 'limit' is the layer dilated
    outward, marking everywhere water could plausibly be.
    """

    def __init__(self, path, index_crs, clip_bounds):
        # Read only what overlaps the tile index. The reference is statewide and
        # the lidar is not; without this the whole file is parsed and reprojected
        # before anything is thrown away.
        start = time.perf_counter()
        bbox = gpd.GeoSeries([box(*clip_bounds)], crs=index_crs)
        gdf = gpd.read_file(path, bbox=bbox)
        if gdf.crs is None:
            raise SystemExit(f"{path} has no CRS")
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            raise SystemExit(f"{path} does not overlap the tile index")
        print(
            f"  read {len(gdf)} reference features in {time.perf_counter() - start:.1f}s"
        )

        # Z coordinates roughly double the cost of every downstream GEOS call and
        # carry no information here, and the vertex density of state hydrography
        # dominates the union cost far more than the feature count does.
        start = time.perf_counter()
        geom = shapely.force_2d(gdf.geometry.to_crs(index_crs).values)
        geom = shapely.simplify(geom, REFERENCE_SIMPLIFY_M)
        geom = shapely.make_valid(geom)
        # Dissolving first is what makes the buffers meaningful: eroding
        # overlapping features individually would carve a gap along every shared
        # edge. Exploding afterwards costs nothing, because union_all leaves the
        # parts disjoint, and buffering them one at a time rather than as one
        # giant multipolygon is what keeps GEOS from grinding for hours.
        merged = shapely.get_parts(shapely.union_all(geom))
        print(
            f"  dissolved to {merged.size} parts in {time.perf_counter() - start:.1f}s"
        )

        start = time.perf_counter()
        self._parts = {
            "core": self._frame(
                shapely.buffer(merged, -SEED_EROSION_M, quad_segs=BUFFER_QUAD_SEGS),
                index_crs,
            ),
            "limit": self._frame(
                shapely.buffer(merged, LIMIT_DILATION_M, quad_segs=BUFFER_QUAD_SEGS),
                index_crs,
            ),
        }
        self._cache = {}
        print(
            f"reference water: {len(gdf)} features, "
            f"core {self._parts['core'].geometry.area.sum() / 1e6:.1f} km2, "
            f"limit {self._parts['limit'].geometry.area.sum() / 1e6:.1f} km2, "
            f"buffered in {time.perf_counter() - start:.1f}s"
        )

    def intersecting(self, kind, geoms):
        """Boolean array marking which of `geoms` touch this buffer at all."""
        part = self._parts[kind]
        keep = np.zeros(len(geoms), dtype=bool)
        if part.empty:
            return keep
        hits = part.sindex.query(geoms, predicate="intersects")
        keep[np.unique(hits[0])] = True
        return keep

    @staticmethod
    def _frame(geoms, crs):
        """One row per polygon, with the empties that erosion leaves dropped."""
        parts = shapely.get_parts(geoms)
        parts = parts[~(shapely.is_empty(parts) | shapely.is_missing(parts))]
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries(parts, crs=crs), crs=crs)

    def mask(self, kind, crs, transform, shape_hw):
        """Boolean array marking cells of the given grid inside this buffer."""
        key = (kind, str(crs))
        if key not in self._cache:
            self._cache[key] = self._parts[kind].to_crs(crs)
        local = self._cache[key]

        empty = np.zeros(shape_hw, dtype=bool)
        if local.empty:
            return empty

        bounds = rasterio.transform.array_bounds(*shape_hw, transform)
        hits = np.unique(local.sindex.query(box(*bounds), predicate="intersects"))
        if hits.size == 0:
            return empty
        return features.geometry_mask(
            local.geometry.iloc[hits],
            out_shape=shape_hw,
            transform=transform,
            invert=True,
        )


def load_reference(index_crs, clip_bounds):
    if REFERENCE_WATER is None:
        return None
    if not Path(REFERENCE_WATER).exists():
        print(f"WARNING: {REFERENCE_WATER} not found; detecting without a reference")
        return None
    return ReferenceWater(REFERENCE_WATER, index_crs, clip_bounds)


def read_grid(src, target_res_m):
    """Decimated band 1 in metres, plus its transform and effective resolution.

    Average resampling is deliberate: it is what turns speckled water into a
    flat surface while leaving sloped terrain sloped.
    """
    unit = linear_unit(src.crs)
    native_res = abs(src.res[0]) * unit
    factor = max(1, round(target_res_m / native_res)) if native_res else 1
    height = max(1, src.height // factor)
    width = max(1, src.width // factor)

    band = src.read(
        1, out_shape=(height, width), masked=True, resampling=Resampling.average
    )
    z = band.astype("float32").filled(np.nan)
    z *= unit if CONVERT_Z_TO_METRES else 1.0

    low, high = VALID_Z_RANGE
    with np.errstate(invalid="ignore"):
        z[~np.isfinite(z) | (z < low) | (z > high)] = np.nan

    transform = src.transform * src.transform.scale(
        src.width / width, src.height / height
    )
    return z, transform, native_res * factor


def flat_low_mask(z, res_m):
    """Cells that are both near-flat and within SEARCH_RANGE."""
    if z.shape[0] < 3 or z.shape[1] < 3 or not res_m:
        return np.zeros(z.shape, dtype=bool)

    dy, dx = np.gradient(z)
    with np.errstate(invalid="ignore"):
        slope = np.hypot(dx, dy) / res_m
        low, high = SEARCH_RANGE
        return np.isfinite(z) & (z >= low) & (z <= high) & (slope <= FLAT_SLOPE)


def peak_estimate(pool):
    """Dominant elevation in a pool, plus the two numbers needed to trust it.

    'fraction' is measured over the same window the estimate is refined on, so
    the threshold tracks how sharp the peak is rather than how HIST_BIN_M is set.
    'censored' is the share of the pool against either end of SEARCH_RANGE; mass
    piled on the edge means the real distribution is cut off and the dominant bin
    is an artefact of the search band, not a surface.
    """
    low, high = SEARCH_RANGE
    edges = np.arange(low, high + HIST_BIN_M, HIST_BIN_M)
    counts, _ = np.histogram(pool, bins=edges)
    peak = int(np.argmax(counts))
    centre = float((edges[peak] + edges[peak + 1]) / 2)

    near = np.abs(pool - centre) <= PEAK_WINDOW_M
    fraction = float(near.mean())
    water_z = float(np.median(pool[near])) if near.any() else centre
    censored = float(
        max((pool <= low + EDGE_GUARD_M).mean(), (pool >= high - EDGE_GUARD_M).mean())
    )
    return water_z, fraction, censored


def water_plane(scan_name, paths, reference):
    """Estimate the flown water level for one scan.

    Returns (elevation, method, stats). Method records whether the value came
    from detection, an override, or the fallback, and whether the histogram was
    seeded by the reference layer, so the index stays auditable.
    """
    if scan_name in WATER_Z_OVERRIDES:
        return float(WATER_Z_OVERRIDES[scan_name]), "override", {}

    seed = reference is not None and SEED_HISTOGRAM
    step = max(1, len(paths) // MAX_HIST_TILES)
    seeded, every = [], []
    for path in paths[::step]:
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                z, transform, res_m = read_grid(src, HIST_RES_M)
                core = (
                    reference.mask("core", src.crs, transform, z.shape)
                    if seed
                    else None
                )
        except (rasterio.errors.RasterioError, OSError) as exc:
            print(f"    unreadable, skipped: {Path(path).name} ({exc})")
            continue

        keep = flat_low_mask(z, res_m)
        if keep.any():
            every.append(z[keep])
        if core is not None:
            keep &= core
            if keep.any():
                seeded.append(z[keep])

    seeded_pool = np.concatenate(seeded) if seeded else np.empty(0, dtype="float32")
    every_pool = np.concatenate(every) if every else np.empty(0, dtype="float32")

    # Exactly one pool is judged. A seed that is present but rejected means the
    # fallback, never a retry against every flat low cell in the scan: that pool
    # is where a broad tidal flat passes for the water plane, which is the whole
    # reason the seed exists. The unseeded pool is only for runs with no usable
    # seed at all - no reference layer, SEED_HISTOGRAM off, or an empty core.
    if seeded_pool.size >= MIN_PEAK_CELLS:
        pool, source = seeded_pool, "seeded"
    else:
        pool, source = every_pool, "unseeded"

    if pool.size == 0:
        return FALLBACK_WATER_Z, "fallback:no-flat-cells", {}
    if pool.size < MIN_PEAK_CELLS:
        return FALLBACK_WATER_Z, "fallback:too-few-cells", {"cells": int(pool.size)}

    water_z, fraction, censored = peak_estimate(pool)
    stats = {
        "cells": int(pool.size),
        "fraction": fraction,
        "censored": censored,
        "peak": water_z,
    }
    if censored > MAX_EDGE_FRACTION:
        return FALLBACK_WATER_Z, f"fallback:censored:{source}", stats
    if fraction < MIN_PEAK_FRACTION:
        return FALLBACK_WATER_Z, f"fallback:weak-peak:{source}", stats
    if water_z > MAX_WATER_Z:
        return FALLBACK_WATER_Z, f"fallback:peak-too-high:{source}", stats
    return water_z, f"detected:{source}", stats


def water_tolerance(paths, water_z, reference):
    """Height above the plane to mask for one scan, plus the cells it rests on.

    Measured on the same MASK_RES_M grid the mask uses, inside the core where
    every valid cell should be water, so it reflects the wave and sensor noise
    the cut actually has to absorb.
    """
    if not ADAPTIVE_TOLERANCE or reference is None:
        return Z_TOLERANCE, 0

    step = max(1, len(paths) // MAX_TOLERANCE_TILES)
    samples = []
    for path in paths[::step]:
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                z, transform, _ = read_grid(src, MASK_RES_M)
                core = reference.mask("core", src.crs, transform, z.shape)
        except (rasterio.errors.RasterioError, OSError) as exc:
            print(f"    unreadable, skipped: {Path(path).name} ({exc})")
            continue
        offset = z[core] - water_z
        offset = offset[np.isfinite(offset) & (np.abs(offset) <= TOLERANCE_OUTLIER_M)]
        if offset.size:
            samples.append(offset)

    offset = np.concatenate(samples) if samples else np.empty(0, dtype="float32")
    if offset.size < MIN_TOLERANCE_CELLS:
        return Z_TOLERANCE, int(offset.size)

    centre = float(np.median(offset))
    sigma = 1.4826 * float(np.median(np.abs(offset - centre)))
    tolerance = float(np.clip(centre + TOLERANCE_SIGMAS * sigma, *TOLERANCE_RANGE))
    return tolerance, int(offset.size)


def close_mask(wet, res_m):
    """Morphological closing with a CLOSING_M disk, safe at the tile edge."""
    radius = int(round(CLOSING_M / res_m)) if res_m else 0
    if radius < 1:
        return wet
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = x * x + y * y <= radius * radius
    # Edge padding keeps the erosion half of the closing from treating the
    # outside of the tile as dry and shaving the mask back from the border.
    padded = np.pad(wet, radius, mode="edge")
    closed = ndimage.binary_closing(padded, structure=disk)
    return closed[radius:-radius, radius:-radius]


def tile_water_polygons(src, water_z, tolerance, reference):
    """Polygon covering this tile's water returns in its native CRS, or None."""
    z, transform, res_m = read_grid(src, MASK_RES_M)
    with np.errstate(invalid="ignore"):
        wet = np.isfinite(z) & (z <= water_z + tolerance)

    if reference is not None:
        if CONFINE_TO_REFERENCE:
            wet &= reference.mask("limit", src.crs, transform, z.shape)
        if ALWAYS_MASK_REFERENCE_CORE:
            wet |= reference.mask("core", src.crs, transform, z.shape)

    if not wet.any():
        return None

    wet = close_mask(wet, res_m)
    cleaned = features.sieve(wet.astype("uint8"), size=MIN_MASK_PIXELS)
    geoms = [
        shape(geom)
        for geom, value in features.shapes(
            cleaned, mask=cleaned > 0, transform=transform
        )
        if value
    ]
    return unary_union(geoms) if geoms else None


def keep_core_connected(masked, reference):
    """Drop tile polygon parts not connected, across tile seams, to the core."""
    rows = masked.assign(_row=np.arange(len(masked))).explode(ignore_index=True)
    parts = rows.geometry.values
    # Adjacent tiles' grids need not line up exactly, so bridge seams by a cell
    # when forming connected regions. The bridge is only used for the test.
    bridge = MASK_RES_M / linear_unit(masked.crs)
    regions = shapely.get_parts(
        shapely.union_all(shapely.buffer(parts, bridge, quad_segs=1))
    )
    regions = regions[reference.intersecting("core", regions)]
    if regions.size == 0:
        return masked.iloc[:0]

    hits = shapely.STRtree(regions).query(parts, predicate="intersects")
    rows = rows.iloc[np.unique(hits[0])]
    kept = rows.dissolve(by="_row", aggfunc="first", sort=True)
    return kept.reset_index(drop=True)[masked.columns]


def mask_scan(scan_name, group, water_z, tolerance, index_crs, reference):
    """Water polygons for every tile in a scan, reprojected to the index CRS."""
    by_crs = {}
    for name, path in zip(group["name"], group["path"]):
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                geom = tile_water_polygons(src, water_z, tolerance, reference)
                crs = src.crs
        except (rasterio.errors.RasterioError, OSError) as exc:
            print(f"    unreadable, skipped: {Path(path).name} ({exc})")
            continue

        if geom is None or geom.is_empty:
            continue
        by_crs.setdefault(crs, []).append(
            {
                "scan": scan_name,
                "name": name,
                "path": path,
                "water_z": water_z,
                "z_tolerance": tolerance,
                "geometry": geom,
            }
        )

    frames = [
        gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_crs(index_crs)
        for crs, records in by_crs.items()
    ]
    if not frames:
        return None
    masked = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=index_crs
    )
    if reference is not None and REQUIRE_CORE_CONNECTION:
        before = masked.geometry.area.sum()
        masked = keep_core_connected(masked, reference)
        dropped = before - masked.geometry.area.sum()
        print(f"    dropped {dropped / 1e6:.3f} km2 not connected to the core")
    return masked


def main():
    tiles = gpd.read_file(INDEX_GPKG, layer=TILE_LAYER)
    if tiles.crs is None:
        raise SystemExit(f"{INDEX_GPKG} layer {TILE_LAYER!r} has no CRS")
    index_crs = tiles.crs
    if ONLY_SCANS:
        tiles = tiles[tiles["scan"].isin(ONLY_SCANS)]
    if tiles.empty:
        raise SystemExit("no tiles to process")

    groups = list(tiles.groupby("scan", sort=True))
    print(f"{len(tiles)} tiles across {len(groups)} scans, index CRS {index_crs}")

    reference = load_reference(index_crs, tiles.total_bounds)

    if reference is not None:
        # Nothing outside the limit can ever be masked, so reading those tiles at
        # full resolution only to polygonize an empty result is pure cost.
        if CONFINE_TO_REFERENCE:
            before = len(tiles)
            tiles = tiles[reference.intersecting("limit", tiles.geometry.values)]
            if tiles.empty:
                raise SystemExit("no tiles intersect the reference water limit")
            print(f"{len(tiles)} of {before} tiles intersect the reference limit")
        tiles = tiles.assign(
            in_core=reference.intersecting("core", tiles.geometry.values)
        )
        groups = list(tiles.groupby("scan", sort=True))

    tile_frames = []
    scan_records = []
    for i, (scan_name, group) in enumerate(groups, 1):
        print(f"[{i}/{len(groups)}] {scan_name} ({len(group)} tiles)")
        # Only tiles touching the always-wet core can seed the histogram, so
        # striding the whole scan mostly samples tiles that contribute nothing.
        hist_group = group
        if "in_core" in group and group["in_core"].any():
            hist_group = group[group["in_core"]]
        paths = hist_group["path"].tolist()

        water_z, method, stats = water_plane(scan_name, paths, reference)
        detail = ", ".join(f"{k}={v:.3g}" for k, v in stats.items())
        print(f"    water plane {water_z:.2f} m NAVD88 [{method}] {detail}")

        # A fallback plane is a guess, so the core's spread around it means nothing.
        if method.startswith("fallback"):
            tolerance, tol_cells = Z_TOLERANCE, 0
        else:
            tolerance, tol_cells = water_tolerance(paths, water_z, reference)
        print(f"    tolerance {tolerance:.3f} m from {tol_cells} core cells")

        masked = mask_scan(
            scan_name, group, water_z, tolerance, index_crs, reference
        )
        if masked is None or masked.empty:
            print("    no water returns found")
            continue

        tile_frames.append(masked)
        merged = unary_union(masked.geometry.values)
        scan_records.append(
            {
                "scan": scan_name,
                "year": group["year"].iloc[0] if "year" in group else None,
                "water_z": water_z,
                "z_tolerance": tolerance,
                "tolerance_cells": tol_cells,
                "method": method,
                "peak_fraction": stats.get("fraction"),
                "sample_cells": stats.get("cells"),
                "tile_count": int(masked["name"].nunique()),
                "area_m2": merged.area,
                "geometry": merged,
            }
        )
        print(f"    masked {merged.area / 1e6:.2f} km2")

    if not scan_records:
        print("No water masks produced.")
        return

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    water_tiles = gpd.GeoDataFrame(
        pd.concat(tile_frames, ignore_index=True), geometry="geometry", crs=index_crs
    )
    water_tiles.to_file(OUT_GPKG, layer=TILE_MASK_LAYER, driver="GPKG")
    print(f"Wrote {len(water_tiles)} tile masks to {OUT_GPKG} ({TILE_MASK_LAYER})")

    water_scans = gpd.GeoDataFrame(scan_records, geometry="geometry", crs=index_crs)
    water_scans.to_file(OUT_GPKG, layer=SCAN_MASK_LAYER, driver="GPKG")
    print(f"Wrote {len(water_scans)} scan masks to {OUT_GPKG} ({SCAN_MASK_LAYER})")

    fallbacks = water_scans[water_scans["method"].str.startswith("fallback")]
    if not fallbacks.empty:
        print(f"WARNING: {len(fallbacks)} scan(s) used the fallback water level:")
        for _, row in fallbacks.iterrows():
            print(f"  {row['scan']}: {row['method']}")


if __name__ == "__main__":
    main()
