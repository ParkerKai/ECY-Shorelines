"""
Barriers to Shoreline Migration
Build barrier hit points by sampling landward transects from a shoreline,
with merged shoreline paths and smoothed transect orientations to prevent crossing
and shore-parallel artifacts. Landward side is determined once per merged path
and applied uniformly to all transects on that path.

Outputs (main):
- barrier_hits (points at hit location, the main product)
- shoreline_barrier_distance_lines (short segments along the shoreline connecting
  consecutive transect bases, with attributes:
  dist_m_start / dist_m_end (masked by min_hit_distance_m),
  dist_m_mean (masked mean of start/end),
  dist_m_raw_start / _end / _mean (raw values, NaN if no hit),
  reason_no_hit (e.g., "shoreline_first" if shoreline was the first intersect))

Optional QA (if export_qa=True):
- transects_hits (origin -> hit lines),
- transects_all (full-length chosen transects to max_search_m).

Author: Kai Parker (WA Dept. of Ecology)
Date: 2026-08-05
"""

# ==============================================================================
# Imports
# ==============================================================================

from dataclasses import dataclass
import fiona
import math
import os
from typing import Union, Tuple, List, Optional, Dict

import numpy as np
import geopandas as gpd
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    LinearRing,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)

from shapely.geometry.base import BaseGeometry  # <-- FIXED: import from geometry.base
from shapely.ops import unary_union, linemerge, nearest_points
from shapely.strtree import STRtree
from shapely import make_valid

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

GeoDF = gpd.GeoDataFrame

# ==============================================================================
# User Inputs
# ==============================================================================

county = "whatcom"  # lower case
ConditionSurvey = gpd.read_file(
    r"N:\SEA\Kai\Data\BarriersToMigration\SMU_ConditionSurvey_V1\OpenSourceVersions\ConditionSurvey.shp"
)
countyBnds = gpd.read_file(r"N:\SEA\Kai\Data\BarriersToMigration\CountyBounds.shp")

# Barriers
buildings = gpd.read_file(
    os.path.join(r"N:\SEA\Kai\Data\Assets", f"{county}_buildings.gpkg")
)
roads = gpd.read_file(os.path.join(r"N:\SEA\Kai\Data\Assets", f"{county}_roads.gpkg"))
rail = gpd.read_file(os.path.join(r"N:\SEA\Kai\Data\Assets", f"{county}_rail.gpkg"))

# DEM by county (10 m)
dem_path = os.path.join(
    r"N:\SEA\Kai\Data\DEM_ByCounty", f"{county.capitalize()}_DEM_10m.tif"
)

# Output directory
out_dir = os.path.join(r"N:\SEA\Kai\Data\BarriersToMigration", county)
os.makedirs(out_dir, exist_ok=True)
timestamp = "2026-08-05"

# Projection
TARGET_CRS = "EPSG:26910"  # NAD83 / UTM zone 10N
COUNTY_CRS_EPSG = 2927  # WA State Plane North HARN

# ==============================================================================
# Constants
# ==============================================================================

ASSETS_ROADS = "roads"
ASSETS_RAIL = "rail"
ASSETS_BUILDINGS = "buildings"
ASSETS_ARMORING = "armoring"
ASSETS_SHORELINE = "shoreline"

# ==============================================================================
# Config
# ==============================================================================


@dataclass
class BarrierConfig:
    target_crs: str = TARGET_CRS
    spacing_m: float = 50.0
    max_search_m: float = 1000.0
    snap_tolerance_m: float = 1.0
    dem_test_len_m: float = 25.0
    min_hit_distance_m: float = 0.0
    segment_fixed_onshore: bool = True
    buffer_distance_m: float = 150.0
    # Orientation smoothing
    orientation_smooth_m: float = 40.0  # Gaussian window size for tangent smoothing
    eps_for_tangent_m: float = 1.0  # small offset for tangent estimate
    max_turn_deg: Optional[float] = 25.0  # per-step turn cap
    # Landward side probe spacing
    side_probe_stride_m: float = 25.0  # spacing for landward-side probe samples
    side_assets_offset_m: float = 25.0  # parallel offset distance for asset fallback
    side_assets_buffer_m: float = (
        20.0  # buffer around parallel offset when counting assets
    )


# ==============================================================================
# Geometry helpers
# ==============================================================================


def densify_line(line: LineString, spacing_m: float) -> List:
    """Return a list of measures along the line at ~spacing_m intervals (including 0 and end)."""
    L = line.length
    if L <= 0:
        return [0.0]
    n = max(1, int(math.ceil(L / spacing_m)))
    measures = [i * L / n for i in range(n + 1)]
    return measures


def line_point(line: LineString, m: float) -> Point:
    """Interpolate a point on the line at measure m (clamped)."""
    return line.interpolate(max(0.0, min(m, line.length)))


def estimate_tangent(line: LineString, m: float, eps: float = 1.0) -> np.ndarray:
    """Estimate a unit tangent vector at measure m using a small window eps."""
    m0 = max(0.0, m - eps)
    m1 = min(line.length, m + eps)
    p0 = line.interpolate(m0)
    p1 = line.interpolate(m1)
    vx = p1.x - p0.x
    vy = p1.y - p0.y
    norm = math.hypot(vx, vy)
    if norm == 0:
        return np.array([1.0, 0.0])
    return np.array([vx / norm, vy / norm])


def left_normal(tangent: np.ndarray) -> np.ndarray:
    """Unit normal pointing to the left of the tangent."""
    return np.array([-tangent[1], tangent[0]])


def make_transect(origin: Point, normal: np.ndarray, length_m: float) -> LineString:
    """Create a transect line from origin along normal for length_m."""
    end = Point(origin.x + normal[0] * length_m, origin.y + normal[1] * length_m)
    return LineString([(origin.x, origin.y), (end.x, end.y)])


# ==============================================================================
# Raster helpers
# ==============================================================================


def open_warped_vrt(
    src_path: Optional[str], target_crs: str
) -> Tuple[Optional[rasterio.io.DatasetReader], Optional[WarpedVRT], Optional[float]]:
    """Open a source raster and wrap it in a WarpedVRT to target_crs."""
    if not src_path or not os.path.exists(src_path):
        return None, None, None
    src = rasterio.open(src_path)
    nodata = src.nodata
    vrt = WarpedVRT(src, crs=target_crs, resampling=Resampling.bilinear)
    return src, vrt, nodata


def sample_vrt_points(
    vrt: Optional[WarpedVRT], pts: List[Point], nodata: Optional[float]
) -> List:
    """Sample raster values at point coordinates."""
    if vrt is None or len(pts) == 0:
        return []
    coords = [(p.x, p.y) for p in pts]
    vals = [float(arr[0]) for arr in vrt.sample(coords)]
    sentinels = {-32768.0, -9999.0, -3.4e38}
    out: List[float] = []
    for v in vals:
        if (
            (nodata is not None and np.isclose(v, nodata))
            or np.isnan(v)
            or (v in sentinels)
        ):
            out.append(np.nan)
        else:
            out.append(v)
    return out


# ==============================================================================
# CRS / Cleaning helpers
# ==============================================================================


def ensure_crs(
    gdf: gpd.GeoDataFrame, expected_epsg: Union[int, str]
) -> gpd.GeoDataFrame:
    """Set or transform CRS safely."""
    epsg_str = str(expected_epsg)
    if gdf.crs is None:
        return gdf.set_crs(epsg_str, allow_override=True)
    if gdf.crs.to_string() != epsg_str:
        return gdf.to_crs(epsg_str)
    return gdf


def finalize_clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply make_valid and drop empties/nulls."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


# ==============================================================================
# Shoreline boundary conversion & merging
# ==============================================================================


def to_line_boundaries(gdf: gpd.GeoDataFrame, target_crs: str) -> List:
    """Flatten polygonal/collection shoreline to EXTERIOR boundaries (LineString parts)."""
    shore = gdf.to_crs(target_crs).copy()
    out: List[LineString] = []
    for geom in shore.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            out.append(geom)
        elif isinstance(geom, MultiLineString):
            out.extend(list(geom.geoms))
        elif isinstance(geom, Polygon):
            out.append(LineString(geom.exterior.coords))
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                out.append(LineString(p.exterior.coords))
        elif isinstance(geom, GeometryCollection):
            for g in geom.geoms:
                if isinstance(g, LineString):
                    out.append(g)
                elif isinstance(g, Polygon):
                    out.append(LineString(g.exterior.coords))
                elif isinstance(g, MultiLineString):
                    out.extend(list(g.geoms))
                elif isinstance(g, MultiPolygon):
                    for p in g.geoms:
                        out.append(LineString(p.exterior.coords))
    return out


def merge_shore_paths(lines: List[LineString]) -> List:
    """Merge shoreline boundary segments into continuous LineStrings (per connected component)."""
    if not lines:
        return []
    u = unary_union(lines)
    lm = linemerge(u)
    if isinstance(lm, LineString):
        return [lm]
    elif isinstance(lm, MultiLineString):
        return list(lm.geoms)
    else:
        merged: List[LineString] = []
        if hasattr(lm, "geoms"):
            for g in lm.geoms:
                if isinstance(g, LineString):
                    merged.append(g)
                elif isinstance(g, MultiLineString):
                    merged.extend(list(g.geoms))
        return merged


# ==============================================================================
# Safe write
# ==============================================================================


def safe_write_gpkg(gdf: gpd.GeoDataFrame, path: str, layer: str):
    """Safely overwrite a layer in a GeoPackage."""
    if os.path.exists(path):
        try:
            fiona.remove(path, layer=layer)
        except Exception:
            pass
    gdf.to_file(path, layer=layer, driver="GPKG")


# ==============================================================================
# GDF from records
# ==============================================================================


def make_gdf_from_records(records: List[dict], crs: str) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from dict records containing 'geometry'."""
    if records and ("geometry" in records[0]):
        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    else:
        return gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries([], crs=crs)}, geometry="geometry", crs=crs
        )


# ==============================================================================
# Type filtering & STRtree builders
# ==============================================================================


def _only_line_like(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or len(gdf) == 0:
        return gdf
    types = gdf.geometry.type.astype(str)
    sel = types.isin(["LineString", "MultiLineString"])
    return gdf.loc[sel].copy()


def build_boundary_strtree_from_polygons_exterior_only(
    polys_gdf: Optional[gpd.GeoDataFrame], crs_str: str, src_label: str
) -> Tuple[Optional[STRtree], Dict[int, Tuple[str, int]]]:
    if polys_gdf is None or len(polys_gdf) == 0:
        return None, {}
    polys = polys_gdf.to_crs(crs_str)
    lines: List[LineString] = []
    for geom in polys.geometry:
        if isinstance(geom, Polygon):
            lines.append(LineString(geom.exterior.coords))
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                lines.append(LineString(p.exterior.coords))
    if not lines:
        return None, {}
    tree = STRtree(lines)
    idmap: Dict[int, Tuple[str, int]] = {}
    for i, g in enumerate(tree.geometries):
        idmap[id(g)] = (src_label, i)
    return tree, idmap


def build_strtree_from_lines(
    lines_gdf: Optional[gpd.GeoDataFrame], crs_str: str, src_label: str
) -> Tuple[Optional[STRtree], Dict[int, Tuple[str, int]]]:
    if lines_gdf is None or len(lines_gdf) == 0:
        return None, {}
    lines = lines_gdf.to_crs(crs_str).geometry.tolist()
    lines = [g for g in lines if g is not None and not g.is_empty]
    if not lines:
        return None, {}
    tree = STRtree(lines)
    idmap = {id(g): (src_label, i) for i, g in enumerate(tree.geometries)}
    return tree, idmap


def query_tree_geoms(
    tree: Optional[STRtree], geom: BaseGeometry, buffer_tol: float = 0.0
) -> List:
    """Query a STRtree with optional buffer around the query geometry."""
    if tree is None or geom is None or geom.is_empty:
        return []
    qgeom = geom.buffer(buffer_tol) if buffer_tol and buffer_tol > 0 else geom
    result = tree.query(qgeom)
    if len(result) == 0:
        return []
    # shapely 2.x returns geometries directly
    if isinstance(result[0], BaseGeometry):
        return result
    else:
        return [tree.geometries[i] for i in result]


# ==============================================================================
# Intersection helper (robust "first along the transect")
# ==============================================================================


def _decompose_to_points(geom: BaseGeometry) -> List:
    """Extract representative points from an intersection geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, (LineString, LinearRing)):
        return [Point(geom.coords[0]), Point(geom.coords[-1])]
    pts: List[Point] = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            pts.extend(_decompose_to_points(g))
    return pts


def first_intersection_on_transect(
    transect: LineString,
    geoms: Sequence[BaseGeometry],
    src_map: Mapping[int, Tuple[Optional[str], Optional[int]]],
    snap_tol: float = 1.0,
    origin_eps: float = 0.5,
) -> Tuple[Optional[Point], Optional[str], Optional[float], Optional[int]]:
    """
    Return the nearest hit along transect:
    - Prefer strict intersections,
    - If no intersection, treat any geometry within snap_tol as a near-miss and use
      the nearest point ON THE TRANSECT as the hit.
    Ignores contacts at or extremely close to the origin (epsilon).
    """
    if transect is None or transect.is_empty or len(geoms) == 0:
        return None, None, None, None

    # strict intersections
    cand: List[Tuple[float, Point, str, int]] = []
    for g in geoms:
        inter = transect.intersection(g)
        if inter is None or inter.is_empty:
            continue
        pts = _decompose_to_points(inter)
        for p in pts:
            d = transect.project(p)
            if d <= origin_eps:  # ignore origin touch
                continue
            src_label, src_idx = src_map.get(id(g), (None, None))
            cand.append((d, p, src_label, src_idx))

    if cand:
        cand.sort(key=lambda t: t[0])
        d, p, src_label, src_idx = cand[0]
        return p, src_label, d, src_idx

    # near-miss
    nm: List[Tuple[float, Point, str, int]] = []
    for g in geoms:
        dist = transect.distance(g)
        if dist <= snap_tol:
            npA, _ = nearest_points(transect, g)
            d = transect.project(npA)
            if d <= origin_eps:
                continue
            src_label, src_idx = src_map.get(id(g), (None, None))
            nm.append((d, npA, src_label, src_idx))
    if nm:
        nm.sort(key=lambda t: t[0])
        d, p, src_label, src_idx = nm[0]
        return p, src_label, d, src_idx

    return None, None, None, None


# ==============================================================================
# Orientation smoothing
# ==============================================================================


def _gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones_like(distances)
    w = np.exp(-0.5 * (distances / sigma) ** 2)
    w[w < 1e-9] = 0.0
    return w


def smooth_tangents_for_samples(
    line: LineString,
    measures: List[float],
    eps_for_tangent: float,
    smooth_m: float,
    max_turn_deg: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Smooth unit tangent vectors along a line using Gaussian moving-average over distance.
    Optionally cap per-step turn angle to max_turn_deg.
    """
    # raw unit tangents
    raw = [estimate_tangent(line, m, eps=eps_for_tangent) for m in measures]
    raw = [t / (np.linalg.norm(t) + 1e-12) for t in raw]

    if smooth_m <= 0:
        smoothed = raw
    else:
        sigma = smooth_m / 2.0
        smoothed: List[np.ndarray] = []
        s = np.array(measures)
        for i, m in enumerate(measures):
            dists = np.abs(s - m)
            mask = dists <= (3.0 * sigma)
            if not np.any(mask):
                smoothed.append(raw[i])
                continue
            w = _gaussian_weights(dists[mask], sigma)
            vecs = np.array([raw[j] for j in np.where(mask)[0]])
            v = (w[:, None] * vecs).sum(axis=0)
            nrm = np.linalg.norm(v)
            smoothed.append(raw[i] if nrm < 1e-12 else (v / nrm))

    # Optional: cap per-step turn
    if max_turn_deg is not None and max_turn_deg > 0 and len(smoothed) >= 2:
        capped = [smoothed[0]]
        max_turn_rad = math.radians(max_turn_deg)
        for i in range(1, len(smoothed)):
            a = capped[-1]
            b = smoothed[i]
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            ang = math.acos(dot)
            if ang <= max_turn_rad:
                capped.append(b)
                continue
            # Rotate 'a' toward 'b' by max_turn_rad (signed)
            sign = np.sign(a[0] * b[1] - a[1] * b[0])
            theta = max_turn_rad * sign
            c = math.cos(theta)
            s_ = math.sin(theta)
            R = np.array([[c, -s_], [s_, c]])
            a_rot = R @ a
            capped.append(a_rot / (np.linalg.norm(a_rot) + 1e-12))
        smoothed = capped

    return smoothed


# ==============================================================================
# Landward side determination (per merged path)
# ==============================================================================


def determine_landward_side_for_path(
    path: LineString,
    measures: List[float],
    tangents_sm: List[np.ndarray],
    config: BarrierConfig,
    dem_vrt: Optional[WarpedVRT],
    dem_nodata: Optional[float],
    tree_roads: Optional[STRtree],
    tree_rail: Optional[STRtree],
    tree_poly_boundaries: Optional[STRtree],
    tree_armor_lines: Optional[STRtree],
) -> Tuple[int, str]:
    """
    Decide landward side for the entire merged path.
    Returns (sign, decision_label) where sign=+1 means LEFT, -1 means RIGHT.
    """
    if len(measures) == 0:
        return +1, "path_landward_fixed_left_default"

    # Downsample probe points to about side_probe_stride_m spacing
    stride = max(
        1, int(round(config.side_probe_stride_m / max(config.spacing_m, 1e-6)))
    )
    idxs = list(range(0, len(measures), stride))
    if len(measures) - 1 not in idxs:
        idxs.append(len(measures) - 1)

    # Primary: DEM-based mean elevation difference
    if dem_vrt is not None:
        left_pts: List[Point] = []
        right_pts: List[Point] = []
        for i in idxs:
            m = measures[i]
            t = tangents_sm[i]
            origin = line_point(path, m)
            n_left = left_normal(t)
            # sample at 1/3 and 2/3 of dem_test_len_m
            for frac in (1.0 / 3.0, 2.0 / 3.0):
                d = config.dem_test_len_m * frac
                lp = Point(origin.x + n_left[0] * d, origin.y + n_left[1] * d)
                rp = Point(origin.x - n_left[0] * d, origin.y - n_left[1] * d)
                left_pts.append(lp)
                right_pts.append(rp)
        left_vals = sample_vrt_points(dem_vrt, left_pts, dem_nodata)
        right_vals = sample_vrt_points(dem_vrt, right_pts, dem_nodata)
        l_mean = np.nanmean(left_vals) if np.any(~np.isnan(left_vals)) else np.nan
        r_mean = np.nanmean(right_vals) if np.any(~np.isnan(right_vals)) else np.nan
        if not np.isnan(l_mean) or not np.isnan(r_mean):
            if np.isnan(r_mean) or (not np.isnan(l_mean) and l_mean >= r_mean):
                return +1, "path_landward_fixed_dem"
            else:
                return -1, "path_landward_fixed_dem"

    # Fallback: asset density using parallel offsets + buffer
    try:
        left_off = path.parallel_offset(
            config.side_assets_offset_m, "left", join_style=2
        )
        right_off = path.parallel_offset(
            config.side_assets_offset_m, "right", join_style=2
        )
        left_buf = (
            left_off.buffer(config.side_assets_buffer_m)
            if left_off is not None
            else None
        )
        right_buf = (
            right_off.buffer(config.side_assets_buffer_m)
            if right_off is not None
            else None
        )

        def count_assets(buf_geom: Optional[BaseGeometry]) -> int:
            if buf_geom is None:
                return 0
            cnt = 0
            for tree in (tree_roads, tree_rail, tree_poly_boundaries, tree_armor_lines):
                geoms = query_tree_geoms(tree, buf_geom, buffer_tol=0.0)
                cnt += len(geoms)
            return cnt

        l_cnt = count_assets(left_buf)
        r_cnt = count_assets(right_buf)
        if l_cnt > r_cnt:
            return +1, "path_landward_fixed_assets"
        elif r_cnt > l_cnt:
            return -1, "path_landward_fixed_assets"
    except Exception:
        pass  # parallel_offset may fail on degenerate cases; continue to default

    # Final fallback: left-of-tangent
    return +1, "path_landward_fixed_left_default"


# ==============================================================================
# Main barrier construction (using merged shoreline paths & fixed landward side)
# ==============================================================================


def build_barrier_hits(
    shoreline_gdf: gpd.GeoDataFrame,
    barrier_roads_gdf: Optional[gpd.GeoDataFrame],
    barrier_rail_gdf: Optional[gpd.GeoDataFrame],
    barrier_polys_gdf: Optional[gpd.GeoDataFrame],
    armoring_gdf: Optional[gpd.GeoDataFrame],
    config: BarrierConfig,
    dem_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
    county_name: Optional[str] = None,
    export_qa: bool = True,
) -> gpd.GeoDataFrame:

    # STRtrees
    tree_roads, src_map_roads = build_strtree_from_lines(
        barrier_roads_gdf, config.target_crs, ASSETS_ROADS
    )
    tree_rail, src_map_rail = build_strtree_from_lines(
        barrier_rail_gdf, config.target_crs, ASSETS_RAIL
    )
    tree_poly_boundaries, src_map_polys = (
        build_boundary_strtree_from_polygons_exterior_only(
            barrier_polys_gdf, config.target_crs, ASSETS_BUILDINGS
        )
    )
    tree_armor_lines, src_map_armor_lines = build_strtree_from_lines(
        armoring_gdf, config.target_crs, ASSETS_ARMORING
    )

    # Unified src_map for hits
    unified_src_map: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
    for mapp in (src_map_roads, src_map_rail, src_map_polys, src_map_armor_lines):
        if mapp:
            unified_src_map.update(mapp)

    # Shoreline boundaries -> merged paths
    raw_shore_lines = to_line_boundaries(shoreline_gdf, config.target_crs)
    if len(raw_shore_lines) == 0:
        raise RuntimeError("No shoreline boundary lines found.")
    merged_paths = merge_shore_paths(raw_shore_lines)
    if len(merged_paths) == 0:
        raise RuntimeError("Failed to merge shoreline into continuous paths.")

    # Shoreline union for shoreline-first guard
    shore_union = unary_union(merged_paths)
    shore_tree = STRtree([shore_union])
    shore_src_map = {id(shore_union): (ASSETS_SHORELINE, 0)}

    # DEM VRT
    _, dem_vrt, dem_nodata = open_warped_vrt(dem_path, config.target_crs)

    barrier_records: List[dict] = []
    qa_transects_hits: List[dict] = []
    qa_transects_all: List[dict] = []
    dist_segments: List[dict] = []

    for path in merged_paths:
        measures = densify_line(path, config.spacing_m)
        tangents_sm = smooth_tangents_for_samples(
            path,
            measures,
            eps_for_tangent=config.eps_for_tangent_m,
            smooth_m=config.orientation_smooth_m,
            max_turn_deg=config.max_turn_deg,
        )

        # Determine landward side once for this path
        side_sign, side_decision = determine_landward_side_for_path(
            path,
            measures,
            tangents_sm,
            config,
            dem_vrt,
            dem_nodata,
            tree_roads,
            tree_rail,
            tree_poly_boundaries,
            tree_armor_lines,
        )

        prev_hit_dist = None
        prev_base_pt = None
        prev_reason_no_hit = None

        iterator = enumerate(zip(measures, tangents_sm))
        if TQDM_AVAILABLE:
            iterator = tqdm(
                iterator, total=len(measures), desc="Sampling merged shoreline"
            )

        for idx, (m, tvec) in iterator:
            origin = line_point(path, m)

            # Fixed landward normal for this path
            base_left = left_normal(tvec)
            normal = base_left if side_sign > 0 else -base_left
            transect = make_transect(origin, normal, config.max_search_m)

            # Shoreline-first guard
            sh_geoms = query_tree_geoms(
                shore_tree, transect, buffer_tol=config.snap_tolerance_m
            )
            sh_hit, _, sh_d, _ = first_intersection_on_transect(
                transect,
                sh_geoms,
                shore_src_map,
                snap_tol=config.snap_tolerance_m,
                origin_eps=0.5,
            )

            # Candidate asset geoms on the chosen side
            all_asset_geoms = []
            for tree in (tree_roads, tree_rail, tree_poly_boundaries, tree_armor_lines):
                all_asset_geoms.extend(
                    query_tree_geoms(tree, transect, buffer_tol=config.snap_tolerance_m)
                )

            hit_pt, hit_src, hit_d, hit_idx = first_intersection_on_transect(
                transect,
                all_asset_geoms,
                unified_src_map,
                snap_tol=config.snap_tolerance_m,
                origin_eps=0.5,
            )

            reason_no_hit = None
            if sh_d is not None and (hit_d is None or sh_d < hit_d):
                hit_pt, hit_src, hit_d, hit_idx = None, None, None, None
                reason_no_hit = "shoreline_first"

            # Record barrier hit point (if any)
            if hit_pt is not None:
                barrier_records.append(
                    {
                        "geometry": hit_pt,
                        "source_layer": hit_src,
                        "distance_m": float(hit_d),
                        "decision": side_decision,
                        "transect_len_m": config.max_search_m,
                        "timestamp": timestamp,
                        "county": county_name,
                    }
                )
                if export_qa:
                    qa_transects_hits.append(
                        {
                            "geometry": LineString(
                                [(origin.x, origin.y), (hit_pt.x, hit_pt.y)]
                            ),
                            "source_layer": hit_src,
                            "distance_m": float(hit_d),
                            "decision": side_decision,
                        }
                    )
            else:
                if export_qa:
                    qa_transects_hits.append(
                        {
                            "geometry": LineString(
                                [(origin.x, origin.y), (origin.x, origin.y)]
                            ),
                            "source_layer": None,
                            "distance_m": np.nan,
                            "decision": side_decision,
                            "reason_no_hit": reason_no_hit,
                        }
                    )

            if export_qa:
                qa_transects_all.append(
                    {
                        "geometry": transect,
                        "decision": side_decision,
                        "has_hit": hit_pt is not None,
                        "reason_no_hit": reason_no_hit,
                    }
                )

            # Distance segments along shoreline between consecutive bases
            if prev_base_pt is not None:
                seg = LineString(
                    [(prev_base_pt.x, prev_base_pt.y), (origin.x, origin.y)]
                )
                raw_start = prev_hit_dist if prev_hit_dist is not None else np.nan
                raw_end = hit_d if hit_d is not None else np.nan
                raw_mean = (
                    float(np.nanmean([raw_start, raw_end]))
                    if not (np.isnan(raw_start) and np.isnan(raw_end))
                    else np.nan
                )

                def mask(v):
                    if np.isnan(v):
                        return np.nan
                    return v if v >= config.min_hit_distance_m else np.nan

                masked_start = mask(raw_start)
                masked_end = mask(raw_end)
                masked_mean = (
                    float(np.nanmean([masked_start, masked_end]))
                    if not (np.isnan(masked_start) and np.isnan(masked_end))
                    else np.nan
                )

                dist_segments.append(
                    {
                        "geometry": seg,
                        "dist_m_raw_start": raw_start,
                        "dist_m_raw_end": raw_end,
                        "dist_m_raw_mean": raw_mean,
                        "dist_m_start": masked_start,
                        "dist_m_end": masked_end,
                        "dist_m_mean": masked_mean,
                        "reason_no_hit": prev_reason_no_hit
                        if hit_pt is not None
                        else reason_no_hit,
                    }
                )

            prev_base_pt = origin
            prev_hit_dist = hit_d if hit_d is not None else np.nan
            prev_reason_no_hit = reason_no_hit

    # Convert records to GeoDataFrames
    gdf_hits = make_gdf_from_records(barrier_records, config.target_crs)
    gdf_segs = make_gdf_from_records(dist_segments, config.target_crs)
    gdf_tran_hits = make_gdf_from_records(qa_transects_hits, config.target_crs)
    gdf_tran_all = make_gdf_from_records(qa_transects_all, config.target_crs)

    # Write outputs
    if out_dir:
        safe_write_gpkg(
            gdf_hits, os.path.join(out_dir, f"barrier_hits_.gpkg"), "barrier_hits"
        )
        safe_write_gpkg(
            gdf_segs,
            os.path.join(out_dir, f"shoreline_barrier_distance_lines_.gpkg"),
            "shoreline_barrier_distance_lines",
        )
        if export_qa:
            safe_write_gpkg(
                gdf_tran_hits,
                os.path.join(out_dir, f"transects_hits_.gpkg"),
                "transects_hits",
            )
            safe_write_gpkg(
                gdf_tran_all,
                os.path.join(out_dir, f"transects_all_.gpkg"),
                "transects_all",
            )

    return gdf_hits


# ==============================================================================
# Pre-Process (clean + clip) and Run
# ==============================================================================


def main():
    print("Fixing geometries and setting CRS")
    countyBnds_fixed = ensure_crs(countyBnds, COUNTY_CRS_EPSG)
    ConditionSurvey_fixed = ensure_crs(ConditionSurvey, TARGET_CRS)
    buildings_fixed = ensure_crs(buildings, TARGET_CRS)
    roads_fixed = ensure_crs(roads, TARGET_CRS)
    rails_fixed = ensure_crs(rail, TARGET_CRS)

    county_fixed = finalize_clean(
        countyBnds_fixed[
            countyBnds_fixed["COUNTY_NM"].astype(str).str.strip().str.lower()
            == county.strip().lower()
        ]
    )
    shoreline_fixed = finalize_clean(ConditionSurvey_fixed)
    buildings_fixed = finalize_clean(buildings_fixed)
    roads_fixed = finalize_clean(roads_fixed)
    rails_fixed = finalize_clean(rails_fixed)

    roads_fixed = _only_line_like(roads_fixed)
    rails_fixed = _only_line_like(rails_fixed)

    # Armoring: subset of ConditionSurvey where Modified == 1
    armoring_fixed = finalize_clean(
        ConditionSurvey_fixed[ConditionSurvey_fixed["Modified"] == 1].copy()
    )

    # Clip shoreline to county bounds
    print("Clipping shoreline (ConditionSurvey) to county bounds")
    shoreline_clip = gpd.clip(
        shoreline_fixed.to_crs(TARGET_CRS), county_fixed.to_crs(TARGET_CRS)
    )
    safe_write_gpkg(
        shoreline_clip, os.path.join(out_dir, f"shoreline_.gpkg"), "shoreline"
    )

    # 150 m buffer around shoreline boundaries
    print("Creating 150 m shoreline-derived buffer")
    cfg_for_buffer = BarrierConfig(target_crs=TARGET_CRS, buffer_distance_m=150.0)
    shore_lines_for_buffer = to_line_boundaries(
        shoreline_clip, cfg_for_buffer.target_crs
    )
    if len(shore_lines_for_buffer) == 0:
        raise RuntimeError("No shoreline boundary lines found to build buffer.")
    shore_union_for_buffer = unary_union(shore_lines_for_buffer)
    buffer_geom = shore_union_for_buffer.buffer(cfg_for_buffer.buffer_distance_m)
    buffer_clip = gpd.GeoDataFrame(
        geometry=[buffer_geom], crs=cfg_for_buffer.target_crs
    )
    safe_write_gpkg(buffer_clip, os.path.join(out_dir, f"buffer_.gpkg"), "buffer")

    # Clip other layers to shoreline buffer
    print("Clipping buildings to shoreline buffer")
    buildings_clip = gpd.clip(buildings_fixed.to_crs(buffer_clip.crs), buffer_clip)
    safe_write_gpkg(
        buildings_clip, os.path.join(out_dir, f"buildings_.gpkg"), "buildings"
    )

    print("Clipping roads to shoreline buffer")
    roads_clip = gpd.clip(roads_fixed.to_crs(buffer_clip.crs), buffer_clip)
    safe_write_gpkg(roads_clip, os.path.join(out_dir, f"roads_.gpkg"), "roads")

    print("Clipping rail to shoreline buffer")
    rail_clip = gpd.clip(rails_fixed.to_crs(buffer_clip.crs), buffer_clip)
    safe_write_gpkg(rail_clip, os.path.join(out_dir, f"rail_.gpkg"), "rail")

    print("Clipping Armoring (ConditionSurvey) to county bounds")
    armoring_clip = gpd.clip(
        armoring_fixed.to_crs(buffer_clip.crs), county_fixed.to_crs(buffer_clip.crs)
    )
    safe_write_gpkg(armoring_clip, os.path.join(out_dir, f"armoring_.gpkg"), "armoring")

    # Optional quick-look plot
    fig, ax = plt.subplots(figsize=(8, 8))
    shoreline_clip.plot(ax=ax, color="black", linewidth=0.8)
    gpd.GeoDataFrame(geometry=[buffer_geom], crs=buffer_clip.crs).plot(
        ax=ax, color="none", edgecolor="orange", linewidth=0.6
    )
    armoring_clip.plot(ax=ax, color="red", linewidth=0.8)
    roads_clip.plot(ax=ax, color="blue", linewidth=0.6)
    rail_clip.plot(ax=ax, color="purple", linewidth=0.6)
    buildings_clip.plot(ax=ax, color="green", linewidth=0.6)
    fig.savefig(os.path.join(out_dir, f"clip_preview_.png"), dpi=200)
    plt.close(fig)

    # Run with merged shoreline paths, smoothed orientation, fixed landward side
    print(
        "Running barrier hits construction on merged shoreline paths with fixed landward side"
    )
    cfg = BarrierConfig(
        target_crs=TARGET_CRS,
        spacing_m=2,  # fine spacing for high-res shoreline
        max_search_m=300,
        dem_test_len_m=25,
        snap_tolerance_m=1.0,
        min_hit_distance_m=0.0,
        segment_fixed_onshore=True,
        buffer_distance_m=150.0,
        orientation_smooth_m=40.0,
        eps_for_tangent_m=1.0,
        max_turn_deg=25.0,
        side_probe_stride_m=25.0,
        side_assets_offset_m=25.0,
        side_assets_buffer_m=20.0,
    )

    print(
        "Processing Barriers to Migration (hits-only, vector) + shoreline distance polylines"
    )
    gdf_out = build_barrier_hits(
        shoreline_gdf=shoreline_clip,
        barrier_roads_gdf=roads_clip,
        barrier_rail_gdf=rail_clip,
        barrier_polys_gdf=buildings_clip,
        armoring_gdf=armoring_clip,
        config=cfg,
        dem_path=dem_path,
        out_dir=out_dir,
        timestamp=timestamp,
        county_name=county.capitalize(),
        export_qa=True,
    )

    print(f"Wrote {len(gdf_out)} barrier hits")
    if len(gdf_out):
        by_src = gdf_out["source_layer"].value_counts(dropna=False)
        print("Source breakdown:")
        print(by_src.to_string())

    print(
        "Tip: map 'shoreline_barrier_distance_lines_.gpkg' layer using 'dist_m_mean' (masked) or 'dist_m_raw_mean' (raw)."
    )


if __name__ == "__main__":
    main()
