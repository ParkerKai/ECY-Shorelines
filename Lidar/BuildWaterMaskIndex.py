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
elevation. Topobathy scans (SKIP_SCAN_PATTERNS) are left unmasked, because
their returns under water are the real bed rather than the sea surface.

  1. Read every tile heavily decimated with average resampling. Averaging
     collapses the speckle, so water becomes genuinely flat while the beach
     face keeps its slope.
  2. Keep only cells whose gradient magnitude is near zero and whose elevation
     falls in a plausible tidal band.
  3. Histogram those flat low elevations. Water contributes one dominant spike;
     its location is the water plane. The tide can change while a scan is
     flown, so each tile touching the core also gets its own plane from its
     own histogram, and falls back to the scan plane only when that fails.
  4. Treat every cell at or below the plane, or with no return at all, as a
     water candidate. There is no tolerance above the plane: wave crests and
     other roughness are handled by step 6 instead.
  5. Close narrow dry gaps without moving the mask's outer edge, sieve out
     specks, and polygonize.
  6. Keep only candidate regions connected, across the whole scan, to the
     core, so low ground that is merely below the plane is not taken for water.
     Then fill every hole in the dissolved water polygon smaller than
     MAX_HOLE_AREA_M2. Whatever pokes above the plane but is surrounded by
     water (wave crests, boats, pilings) becomes water. The shoreline is then
     just the line where the plane meets the beach.
  7. Land is the scan footprint minus the water.

A reference layer of visible surface water sharpens every one of those steps.
Because it does not register exactly against the lidar shoreline it is never
used as a hard boundary; instead it is eroded inward to a core and dilated
outward to a permissive limit. The core seeds the histogram, so a broad but
genuinely flat tidal flat can no longer be mistaken for the water plane, and
anchors the connectivity test that decides what is seaward. It is not forced to
water by default, because the reference also covers flats that are exposed at
lower tides, and ground above the plane belongs on the land side. The limit
bounds where detection is allowed to mask at all, keeping low-lying inland
ground out of it.

Scans with no convincing peak fall back to FALLBACK_WATER_Z so they are still
over-masked rather than silently passed through. WATER_Z_OVERRIDES lets a known
water level be forced in for any scan.

Output GeoPackage layers:
  water_tiles - one water polygon per tile, for windowed raster lookup
  water_scans - the dissolved water mask per scan, plus the detection stats
  land_scans  - the scan footprint minus its water mask, for every processed
                scan (including all-land scans outside the reference limit)

A QA/QC log with the same name as the GeoPackage (.log) is written beside it.
It records the configuration and code version, every scan and tile plane
decision with its histogram stats and the reason a tile did or did not get its
own plane, and the area each masking step added or removed.

Consumers only need water_tiles or water_scans: within a scan's footprint,
anything not water is land. check_mask_coverage() uses the scans recorded in
land_scans to flag scans a stale mask knows nothing about.
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
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
LAND_SCAN_LAYER = "land_scans"

# A QA/QC log is written next to OUT_GPKG, with the same name and a .log suffix,
# overwritten on each run like the GeoPackage. The console gets a summary; the
# file also gets every per-tile plane decision and area accounting step.
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG

# Restrict the run to these scan names while tuning. Empty = every scan.
ONLY_SCANS = ()

# Scans whose name contains any of these (case-insensitive) are never masked.
# Topobathy lidar maps the bed through the water, so there is no water surface
# to cut at and every return is real, comparable topography.
SKIP_SCAN_PATTERNS = ("topobathy",)

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
# Mask the core in every scan regardless of its elevation. Off, because the
# reference covers tidal flats that are dry above the plane at lower tides.
ALWAYS_MASK_REFERENCE_CORE = False

# Vertex tolerance applied to the reference before buffering. The layer is only
# ever used eroded by tens of metres, so this is far below its working accuracy,
# and statewide hydrography carries orders of magnitude more detail than that.
REFERENCE_SIMPLIFY_M = 2.0
# Buffer smoothness. Low is fine for a mask that is itself approximate.
BUFFER_QUAD_SEGS = 2

# Elevations are metres NAVD88 throughout.
# Band searched for the water plane. Must bracket the local tidal range but
# stay below anything that could be real upland terrain.
SEARCH_RANGE = (-2.0, 3.0)  # Nov 2, 2026: widened from (-1.0, 3.0); bainbridge_2015 water sits near -1.4

# Gradient magnitude (rise/run) below which an averaged cell counts as flat.
FLAT_SLOPE = 0.01  # Oct 30, 2026: Tuned from 0.02 (2x stricter to eliminate 40% bathymetry contamination)

# Grid the water-plane histogram is built on. Coarse enough to average out
# speckle, fine enough that the beach face still registers as sloped.
HIST_RES_M = 15.0  # Oct 30, 2026: Tuned from 10.0 for better speckle suppression on smooth water
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

# Cap on tiles read for the scan histogram; scans with more are strided.
MAX_HIST_TILES = 400
# Also fit a plane to each tile touching the core, for scans flown across a
# changing tide. Reads every such tile at HIST_RES_M, ignoring MAX_HIST_TILES.
PER_TILE_PLANE = True
# A tile keeps its own plane only if its peak is at least this sharp; otherwise
# it uses the scan plane. A single tile's pool is small enough that a tidal
# flat or marsh can win a weak peak. Checked against the elevation where returns
# stop at the shore: tiles scoring 0.37 and up matched it within about 0.1 m,
# while island_2014 tiles at 0.22-0.32 landed up to 2.4 m off.
TILE_MIN_PEAK_FRACTION = 0.35
# Reject a detected plane above this; a higher peak is more likely a tidal flat.
# Must sit above local MHHW, which is about 3.4 m NAVD88 in Puget Sound.
MAX_WATER_Z = 4.0

# Used when detection fails or is rejected. Set near the local low-water datum.
FALLBACK_WATER_Z = -1

# Grid the mask itself is rasterized on, before polygonizing.
MASK_RES_M = 1.0
# Morphological closing radius. Smooths the swash edge, where wave crests above
# the plane touch the beach, by bridging dry gaps narrower than twice this
# without moving the outer edge of the mask.
CLOSING_M = 5.0
# Holes in a scan's dissolved water polygon smaller than this are filled. Wave
# crests, boats and pilings surrounded by water become water. Larger holes are
# kept as real islands.
MAX_HOLE_AREA_M2 = 1000.0
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


log = logging.getLogger("BuildWaterMaskIndex")


def log_path_for(gpkg):
    return Path(gpkg).with_suffix(".log")


def setup_logging(log_path):
    """Console summary plus a detailed QA/QC file, replacing earlier handlers."""
    close_logging()
    log.setLevel(min(CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL))
    log.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CONSOLE_LOG_LEVEL)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file.setLevel(FILE_LOG_LEVEL)
    file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    )
    log.addHandler(file)


def close_logging():
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


def log_run_header():
    """What produced this mask: when, from which code, with which settings."""
    log.info(f"BuildWaterMaskIndex run started {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.debug(f"python {sys.version.split()[0]}, script {Path(__file__).resolve()}")
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    log.debug(f"git commit {commit or 'unknown'}")
    log.debug("configuration:")
    for name, value in globals().items():
        if name.isupper() and isinstance(
            value, (str, int, float, bool, tuple, dict, Path, type(None))
        ):
            log.debug(f"  {name} = {value!r}")


def fmt_stats(stats):
    """Histogram stats as a compact, consistently ordered string."""
    parts = []
    if "cells" in stats:
        parts.append(f"cells={stats['cells']}")
    if "fraction" in stats:
        parts.append(f"fraction={stats['fraction']:.3f}")
    if "censored" in stats:
        parts.append(f"censored={stats['censored']:.3f}")
    if "peak" in stats:
        parts.append(f"peak={stats['peak']:+.2f}")
    return ", ".join(parts) or "no stats"


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
        log.info(
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
        log.info(
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
        log.info(
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
        log.warning(f"WARNING: {REFERENCE_WATER} not found; detecting without a reference")
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


def plane_pools(paths, reference):
    """{path: (seeded, every)} flat low elevations per readable tile."""
    seed = reference is not None and SEED_HISTOGRAM
    pools = {}
    for path in paths:
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
            log.warning(f"    unreadable, skipped: {Path(path).name} ({exc})")
            continue

        keep = flat_low_mask(z, res_m)
        every = z[keep]
        seeded = z[keep & core] if core is not None else every[:0]
        pools[path] = (seeded, every)
    return pools


def judge_plane(seeded_pool, every_pool):
    """(elevation, method, stats) from one tile's or one scan's pools."""
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


def concat(arrays):
    arrays = list(arrays)
    return np.concatenate(arrays) if arrays else np.empty(0, dtype="float32")


def water_plane(scan_name, paths, reference, tile_paths=()):
    """Estimate the flown water level for one scan, and optionally per tile.

    Returns (elevation, method, stats, tile_planes). Method records whether the
    value came from detection, an override, or the fallback, and whether the
    histogram was seeded by the reference layer, so the index stays auditable.
    tile_planes maps each of `tile_paths` whose own seeded histogram passes
    every check to its (elevation, method); the rest are left to the scan plane.
    """
    if scan_name in WATER_Z_OVERRIDES:
        water_z = float(WATER_Z_OVERRIDES[scan_name])
        log.info(f"    WATER_Z_OVERRIDES forces the plane to {water_z:+.2f} m")
        return water_z, "override", {}, {}

    step = max(1, len(paths) // MAX_HIST_TILES)
    sample = paths[::step]
    pools = plane_pools(dict.fromkeys([*sample, *tile_paths]), reference)

    seeded = concat(pools[p][0] for p in sample if p in pools)
    every = concat(pools[p][1] for p in sample if p in pools)
    log.debug(
        f"    scan histogram: {len(sample)} of {len(paths)} core tiles sampled "
        f"(stride {step}), {seeded.size} seeded and {every.size} unseeded flat "
        f"cells in {SEARCH_RANGE} m"
    )
    water_z, method, stats = judge_plane(seeded, every)
    log.debug(f"    scan plane: {fmt_stats(stats)} -> {water_z:+.2f} m [{method}]")

    tile_planes = {}
    if tile_paths:
        log.debug(
            f"    per-tile planes (own plane needs a detected seeded peak with "
            f"fraction >= TILE_MIN_PEAK_FRACTION {TILE_MIN_PEAK_FRACTION}):"
        )
    for path in tile_paths:
        name = Path(path).name
        if path not in pools:
            log.debug(f"      {name}: unreadable -> scan plane")
            continue
        # Seeded only: a lone tile's unseeded pool is all tidal flat risk.
        tile_z, tile_method, tile_stats = judge_plane(pools[path][0], np.empty(0))
        if not tile_method.startswith("detected"):
            outcome = f"scan plane ({tile_method})"
        elif tile_stats["fraction"] < TILE_MIN_PEAK_FRACTION:
            outcome = (
                f"scan plane (weak tile peak {tile_stats['fraction']:.3f} < "
                f"{TILE_MIN_PEAK_FRACTION})"
            )
        else:
            tile_planes[path] = (tile_z, f"tile:{tile_method}")
            outcome = f"own plane {tile_z:+.2f} m"
        log.debug(f"      {name}: {fmt_stats(tile_stats)} -> {outcome}")
    return water_z, method, stats, tile_planes


def polygonal(geom):
    """Only the polygon parts of `geom`, as one (Multi)Polygon or an empty one."""
    if geom is None or geom.is_empty:
        return shapely.MultiPolygon()
    parts = [
        p for p in shapely.get_parts(shapely.make_valid(geom))
        if p.geom_type in ("Polygon", "MultiPolygon")
    ]
    return shapely.union_all(parts) if parts else shapely.MultiPolygon()


def fill_small_holes(geom, max_area):
    """`geom` with every interior ring smaller than `max_area` filled."""
    filled = []
    for part in shapely.get_parts(polygonal(geom)):
        keep = [r for r in part.interiors if shapely.Polygon(r).area >= max_area]
        filled.append(shapely.Polygon(part.exterior, keep))
    # Union, because filling a hole can swallow another part sitting inside it.
    return polygonal(shapely.union_all(filled)) if filled else shapely.MultiPolygon()


def clip_to_tiles(geom, tile_geoms):
    """`geom` clipped to each tile footprint, cheaply via its bounding box first."""
    out = []
    for tile in tile_geoms:
        rough = shapely.make_valid(shapely.clip_by_rect(geom, *tile.bounds))
        out.append(polygonal(shapely.intersection(rough, tile)))
    return out


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


def polygonize(mask, transform):
    """Union of the True regions of a boolean grid, or None."""
    geoms = [
        shape(geom)
        for geom, value in features.shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if value
    ]
    return unary_union(geoms) if geoms else None


def tile_water_polygons(src, water_z, reference, overlap=None):
    """(water candidates, confirmed dry ground) for one tile in its native CRS.

    A candidate is any cell at or below the plane, or with no return at all,
    since water is often sparse in lidar. Whether a candidate really is water
    is decided later, scan-wide, by its connection to the core.

    Tiles in a scan can overlap, and one tile's no-data padding must not
    override another's real ground. So where this tile overlaps another one
    (`overlap`, a geometry in the tile CRS), ground it sees above the plane and
    does not mask itself is returned as confirmed dry, for the scan to
    subtract. Either value may be None.
    """
    z, transform, res_m = read_grid(src, MASK_RES_M)
    with np.errstate(invalid="ignore"):
        wet = ~(z > water_z)

    if reference is not None:
        if CONFINE_TO_REFERENCE:
            wet &= reference.mask("limit", src.crs, transform, z.shape)
        if ALWAYS_MASK_REFERENCE_CORE:
            wet |= reference.mask("core", src.crs, transform, z.shape)

    if wet.any():
        wet = close_mask(wet, res_m)
        wet = features.sieve(wet.astype("uint8"), size=MIN_MASK_PIXELS) > 0

    dry = None
    if overlap is not None and not overlap.is_empty:
        shared = features.geometry_mask(
            [overlap], out_shape=z.shape, transform=transform, invert=True
        )
        dry = polygonize(shared & np.isfinite(z) & ~wet, transform)
    return (polygonize(wet, transform) if wet.any() else None), dry


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


def mask_scan(scan_name, group, water_z, index_crs, reference, tile_planes=None):
    """Water mask for one scan in the index CRS, as (per-tile frame, dissolved).

    Each tile is cut at its own plane in `tile_planes` ({path: (z, method)})
    when it has one, and at the scan's `water_z` otherwise. Returns
    (None, empty) when nothing is water.
    """
    tile_planes = tile_planes or {}
    unit_area = linear_unit(index_crs) ** 2
    footprints = group.geometry.values
    tree = shapely.STRtree(footprints)
    by_crs, dry_by_crs = {}, {}
    log.debug("    tile cuts (candidate = at/below plane or no data, in limit):")
    for i, (name, path) in enumerate(zip(group["name"], group["path"])):
        tile_z, source = tile_planes.get(path, (water_z, "scan"))
        others = [j for j in tree.query(footprints[i], predicate="intersects") if j != i]
        overlap = (
            shapely.intersection(footprints[i], shapely.union_all(footprints[others]))
            if others
            else None
        )
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    log.warning(f"    no CRS, skipped: {name}")
                    continue
                if overlap is not None:
                    overlap = gpd.GeoSeries([overlap], crs=index_crs).to_crs(src.crs).iloc[0]
                geom, dry = tile_water_polygons(src, tile_z, reference, overlap)
                crs = src.crs
        except (rasterio.errors.RasterioError, OSError) as exc:
            log.warning(f"    unreadable, skipped: {Path(path).name} ({exc})")
            continue

        tile_unit = linear_unit(crs) ** 2
        wet_km2 = 0.0 if geom is None else geom.area * tile_unit / 1e6
        dry_km2 = 0.0 if dry is None else dry.area * tile_unit / 1e6
        log.debug(
            f"      {name}: plane {tile_z:+.2f} m [{source}], "
            f"candidate {wet_km2:.3f} km2, dry in overlap {dry_km2:.3f} km2"
        )
        if dry is not None and not dry.is_empty:
            dry_by_crs.setdefault(crs, []).append(dry)
        if geom is None or geom.is_empty:
            continue
        by_crs.setdefault(crs, []).append({"name": name, "geometry": geom})

    frames = [
        gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_crs(index_crs)
        for crs, records in by_crs.items()
    ]
    if not frames:
        log.info("    no tile produced water candidates")
        return None, shapely.MultiPolygon()
    masked = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=index_crs
    )
    log.debug(
        f"    candidates: {masked.geometry.area.sum() * unit_area / 1e6:.3f} km2 "
        f"(overlapping tiles counted twice)"
    )
    if dry_by_crs:
        dry = shapely.union_all(
            [
                gpd.GeoSeries(geoms, crs=crs).to_crs(index_crs).union_all()
                for crs, geoms in dry_by_crs.items()
            ]
        )
        before = masked.geometry.area.sum()
        masked = masked.assign(
            geometry=[polygonal(g.difference(dry)) for g in masked.geometry.values]
        )
        masked = masked[~masked.geometry.is_empty]
        removed = max(0.0, (before - masked.geometry.area.sum()) * unit_area)
        log.info(f"    removed {removed / 1e6:.3f} km2 seen as dry ground by an overlapping tile")
    if masked.empty:
        log.info("    nothing left after removing dry overlap")
        return None, shapely.MultiPolygon()
    if reference is not None and REQUIRE_CORE_CONNECTION:
        before = masked.geometry.area.sum()
        masked = keep_core_connected(masked, reference)
        dropped = max(0.0, (before - masked.geometry.area.sum()) * unit_area)
        log.info(f"    dropped {dropped / 1e6:.3f} km2 not connected to the core")
    if masked.empty:
        log.info("    nothing connected to the reference core")
        return None, shapely.MultiPolygon()

    # Holes are filled on the dissolved scan, not per tile, so roughness that
    # straddles a tile seam is caught, as are slivers where tile grids misalign.
    connected = polygonal(unary_union(masked.geometry.values))
    water = fill_small_holes(connected, MAX_HOLE_AREA_M2 / unit_area)
    filled = (water.area - connected.area) * unit_area
    trimmed = polygonal(water.intersection(unary_union(group.geometry.values)))
    outside = (water.area - trimmed.area) * unit_area
    water = trimmed
    parts = shapely.get_parts(water)
    holes = sum(len(p.interiors) for p in parts)
    log.info(
        f"    filled {filled / 1e6:.3f} km2 of holes < {MAX_HOLE_AREA_M2:g} m2; "
        f"{len(parts)} water part(s), {holes} larger hole(s) kept as land"
    )
    log.debug(f"    trimmed {outside / 1e6:.4f} km2 outside the tile footprints")
    if water.is_empty:
        return None, water

    clipped = clip_to_tiles(water, group.geometry.values)
    planes = [tile_planes.get(p, (water_z, "scan")) for p in group["path"]]
    per_tile = gpd.GeoDataFrame(
        {
            "scan": scan_name,
            "name": group["name"].values,
            "path": group["path"].values,
            "water_z": [z for z, _ in planes],
            "plane": [m for _, m in planes],
        },
        geometry=clipped,
        crs=index_crs,
    )
    per_tile = per_tile[~per_tile.geometry.is_empty].reset_index(drop=True)
    return per_tile, water


def is_skipped_scan(scan_name):
    """True for scans that are deliberately never masked (SKIP_SCAN_PATTERNS)."""
    lowered = str(scan_name).lower()
    return any(p.lower() in lowered for p in SKIP_SCAN_PATTERNS)


def check_mask_coverage(scan_names, gpkg=OUT_GPKG):
    """Warn about scans the mask GeoPackage says nothing about.

    Every masked scan is in land_scans, and those with water also in
    water_scans. A scan in neither was never processed, so it will pass
    through unmasked: expected for SKIP_SCAN_PATTERNS, otherwise a sign the
    mask is stale and BuildWaterMaskIndex.py needs re-running. Returns the
    scans missing from the mask that are not deliberately skipped.
    """
    import pyogrio

    gpkg = Path(gpkg)
    if not gpkg.exists():
        return sorted(set(scan_names))
    layers = set(pyogrio.list_layers(gpkg)[:, 0])
    if LAND_SCAN_LAYER not in layers:
        print(
            f"WARNING: {gpkg.name} has no {LAND_SCAN_LAYER!r} layer; it predates "
            "the water-plane cut. Re-run BuildWaterMaskIndex.py"
        )
        return []
    known = set()
    for layer in (SCAN_MASK_LAYER, LAND_SCAN_LAYER):
        if layer in layers:
            known |= set(
                pyogrio.read_dataframe(
                    gpkg, layer=layer, columns=["scan"], read_geometry=False
                )["scan"]
            )
    missing = sorted(set(scan_names) - known)
    skipped = [s for s in missing if is_skipped_scan(s)]
    stale = [s for s in missing if not is_skipped_scan(s)]
    if skipped:
        print(f"  {len(skipped)} scan(s) left unmasked by design: {', '.join(skipped)}")
    if stale:
        print(
            f"WARNING: {len(stale)} scan(s) missing from {gpkg.name} will not be "
            f"masked; re-run BuildWaterMaskIndex.py: {', '.join(stale)}"
        )
    return stale


def main():
    log_path = log_path_for(OUT_GPKG)
    setup_logging(log_path)
    start = time.perf_counter()
    try:
        log_run_header()
        log.info(f"QA/QC log: {log_path}")
        build()
    except SystemExit as exc:
        log.error(f"stopped: {exc}")
        raise
    except Exception:
        log.exception("run failed")
        raise
    finally:
        log.info(f"total {time.perf_counter() - start:.0f}s")
        close_logging()


def build():
    tiles = gpd.read_file(INDEX_GPKG, layer=TILE_LAYER)
    if tiles.crs is None:
        raise SystemExit(f"{INDEX_GPKG} layer {TILE_LAYER!r} has no CRS")
    index_crs = tiles.crs
    log.debug(f"tile index: {len(tiles)} tiles across {tiles['scan'].nunique()} scans")
    if ONLY_SCANS:
        tiles = tiles[tiles["scan"].isin(ONLY_SCANS)]
        log.info(f"ONLY_SCANS restricts the run to {len(ONLY_SCANS)} scan(s)")
    if SKIP_SCAN_PATTERNS:
        skip = tiles["scan"].map(is_skipped_scan).to_numpy(bool)
        skipped = sorted(tiles.loc[skip, "scan"].unique())
        if skipped:
            log.info(
                f"skipping {len(skipped)} scans matching {SKIP_SCAN_PATTERNS} "
                f"(left unmasked): {', '.join(skipped)}"
            )
        tiles = tiles[~skip]
    if tiles.empty:
        raise SystemExit("no tiles to process")

    groups = list(tiles.groupby("scan", sort=True))
    log.info(f"{len(tiles)} tiles across {len(groups)} scans, index CRS {index_crs}")
    # Land covers every tile, including those the limit filter skips below.
    footprints = {
        scan: unary_union(group.geometry.values) for scan, group in groups
    }
    scan_years = {
        scan: group["year"].iloc[0] if "year" in group else None
        for scan, group in groups
    }
    unit_area = linear_unit(index_crs) ** 2

    reference = load_reference(index_crs, tiles.total_bounds)

    tile_count = tiles.groupby("scan").size()
    if reference is not None:
        # Nothing outside the limit can ever be masked, so reading those tiles at
        # full resolution only to polygonize an empty result is pure cost.
        if CONFINE_TO_REFERENCE:
            before = len(tiles)
            tiles = tiles[reference.intersecting("limit", tiles.geometry.values)]
            log.info(f"{len(tiles)} of {before} tiles intersect the reference limit")
        tiles = tiles.assign(
            in_core=reference.intersecting("core", tiles.geometry.values)
        )
        groups = list(tiles.groupby("scan", sort=True))

    tile_frames = []
    scan_records = []
    land_records = []
    for i, (scan_name, group) in enumerate(groups, 1):
        log.info(f"[{i}/{len(groups)}] {scan_name} ({len(group)} tiles)")
        outside = tile_count[scan_name] - len(group)
        if outside:
            log.debug(
                f"    {outside} more tile(s) lie outside the reference limit "
                "and are all land"
            )
        # Only tiles touching the always-wet core can seed the histogram, so
        # striding the whole scan mostly samples tiles that contribute nothing.
        hist_group = group
        if "in_core" in group and group["in_core"].any():
            hist_group = group[group["in_core"]]
        log.debug(
            f"    {len(hist_group)} of {len(group)} tiles touch the reference core "
            "and feed the histogram"
        )
        paths = hist_group["path"].tolist()

        tile_paths = paths if PER_TILE_PLANE else ()
        water_z, method, stats, tile_planes = water_plane(
            scan_name, paths, reference, tile_paths
        )
        log.info(
            f"    water plane {water_z:.2f} m NAVD88 [{method}] {fmt_stats(stats)}"
        )
        if method.startswith("fallback"):
            log.warning(
                f"    WARNING: scan plane fell back to FALLBACK_WATER_Z "
                f"{FALLBACK_WATER_Z} ({method})"
            )
        tile_z = [z for z, _ in tile_planes.values()]
        if tile_z:
            log.info(
                f"    {len(tile_z)} of {len(group)} tiles use their own plane, "
                f"{min(tile_z):.2f} to {max(tile_z):.2f} m"
            )

        masked, merged = mask_scan(
            scan_name, group, water_z, index_crs, reference, tile_planes
        )
        year = scan_years[scan_name]

        land = polygonal(footprints[scan_name].difference(merged))
        if not land.is_empty:
            land_records.append(
                {
                    "scan": scan_name,
                    "year": year,
                    "water_z": water_z,
                    "method": method,
                    "area_m2": land.area * unit_area,
                    "geometry": land,
                }
            )

        if masked is None or masked.empty:
            log.info("    no water returns found; whole footprint is land")
            continue

        tile_frames.append(masked)
        scan_records.append(
            {
                "scan": scan_name,
                "year": year,
                "water_z": water_z,
                "water_z_min": float(masked["water_z"].min()),
                "water_z_max": float(masked["water_z"].max()),
                "tile_planes": len(tile_planes),
                "method": method,
                "peak_fraction": stats.get("fraction"),
                "sample_cells": stats.get("cells"),
                "tile_count": int(masked["name"].nunique()),
                "area_m2": merged.area * unit_area,
                "geometry": merged,
            }
        )
        log.info(
            f"    masked {merged.area * unit_area / 1e6:.2f} km2 water, "
            f"{land.area * unit_area / 1e6:.2f} km2 land"
        )

    # Scans with no tile inside the reference limit can hold no water, so their
    # whole footprint is land. Recording them lets consumers tell a scan that
    # was checked and found dry from one missing from a stale mask.
    processed = {scan for scan, _ in groups}
    outside_scans = sorted(set(footprints) - processed)
    for scan_name in outside_scans:
        land = polygonal(footprints[scan_name])
        land_records.append(
            {
                "scan": scan_name,
                "year": scan_years[scan_name],
                "water_z": np.nan,
                "method": "outside-limit",
                "area_m2": land.area * unit_area,
                "geometry": land,
            }
        )
    if outside_scans:
        log.info(
            f"{len(outside_scans)} scans lie wholly outside the "
            "reference limit and were written as all land"
        )
        log.debug(f"  all-land scans: {', '.join(outside_scans)}")

    if not scan_records and not land_records:
        log.info("No water or land masks produced.")
        return

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    if scan_records:
        water_tiles = gpd.GeoDataFrame(
            pd.concat(tile_frames, ignore_index=True),
            geometry="geometry",
            crs=index_crs,
        )
        water_tiles.to_file(OUT_GPKG, layer=TILE_MASK_LAYER, driver="GPKG")
        log.info(f"Wrote {len(water_tiles)} tile masks to {OUT_GPKG} ({TILE_MASK_LAYER})")

        water_scans = gpd.GeoDataFrame(scan_records, geometry="geometry", crs=index_crs)
        water_scans.to_file(OUT_GPKG, layer=SCAN_MASK_LAYER, driver="GPKG")
        log.info(f"Wrote {len(water_scans)} scan masks to {OUT_GPKG} ({SCAN_MASK_LAYER})")
    else:
        log.info("No water masks produced.")

    if land_records:
        land_scans = gpd.GeoDataFrame(land_records, geometry="geometry", crs=index_crs)
        land_scans.to_file(OUT_GPKG, layer=LAND_SCAN_LAYER, driver="GPKG")
        log.info(f"Wrote {len(land_scans)} land polygons to {OUT_GPKG} ({LAND_SCAN_LAYER})")

    if not scan_records:
        return
    fallbacks = water_scans[water_scans["method"].str.startswith("fallback")]
    if not fallbacks.empty:
        log.warning(f"WARNING: {len(fallbacks)} scan(s) used the fallback water level:")
        for _, row in fallbacks.iterrows():
            log.warning(f"  {row['scan']}: {row['method']}")


if __name__ == "__main__":
    main()
