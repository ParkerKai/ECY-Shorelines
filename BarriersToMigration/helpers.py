# barriers_to_migration/helpers.py
import logging
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, nearest_points, unary_union
from shapely.strtree import STRtree

GeoDF = gpd.GeoDataFrame
logger = logging.getLogger("barriers_to_migration")


class _TransformedRasterSampler:
    """Sample a source raster after transforming target-CRS coordinates."""

    def __init__(self, dataset, source_crs: str, target_crs: str):
        self.dataset = dataset
        self.transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)

    def sample(self, coords):
        coords = list(coords)
        if not coords:
            return iter(())
        xs, ys = zip(*coords, strict=True)
        source_xs, source_ys = self.transformer.transform(xs, ys)

        def values():
            inverse_transform = ~self.dataset.transform
            for x, y in zip(source_xs, source_ys, strict=True):
                col_float, row_float = inverse_transform * (x, y)
                row = math.floor(row_float)
                col = math.floor(col_float)
                value = self.dataset.read(
                    1,
                    window=Window(col, row, 1, 1),
                    boundless=True,
                    fill_value=self.dataset.nodata,
                )[0, 0]
                yield np.array([value])

        return values()

    def close(self) -> None:
        pass


# ---------------------------
# Naming / path helpers
# ---------------------------


def normalize_county_name(value: object) -> str:
    return re.sub(r"[\s_]+", "_", str(value).strip().lower())


# ---------------------------
# CRS / Cleaning helpers
# ---------------------------
def ensure_crs(gdf: GeoDF, expected_epsg: int | str) -> GeoDF:
    """Set or transform CRS safely to `expected_epsg`."""
    epsg_str = str(expected_epsg)
    if gdf.crs is None:
        return gdf.set_crs(epsg_str, allow_override=True)
    if gdf.crs.to_string() != epsg_str:
        return gdf.to_crs(epsg_str)
    return gdf


def finalize_clean(gdf: GeoDF) -> GeoDF:
    """Apply make_valid and drop empties/nulls."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


# ---------------------------
# Safe write to GeoPackage
# ---------------------------


def safe_write_gpkg(gdf, path, layer, mode=None, engine="pyogrio"):
    """
    Robust writer for GPKG:
    - Ensures parent directory exists
    - Clears read-only attribute if present
    - Uses 'w' on first write, 'a' thereafter (unless overridden)
    """
    if gdf is None or gdf.empty:
        # Still create an empty layer if you want; otherwise just return early
        # return
        pass

    # 1) Ensure parent dir
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 2) Choose mode if not provided
    if mode is None:
        mode = "a" if os.path.exists(path) else "w"

    # 3) If file exists but is read-only, clear attribute (Windows)
    if os.path.exists(path) and not os.access(path, os.W_OK):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except Exception as e:
            raise PermissionError(
                f"{path} is not writable; clear read-only and close in other apps."
            ) from e

    # 4) Write
    gdf.to_file(path, layer=layer, driver="GPKG", mode=mode, engine=engine)


def write_layers_gpkg(layers: dict[str, gpd.GeoDataFrame], path: str):
    """
    Write multiple layers to a single GPKG:
    - First layer creates the dataset ('w'), subsequent layers append ('a').
    """
    first = True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for layer_name, gdf in layers.items():
        mode = "w" if first else "a"
        gdf.to_file(path, layer=layer_name, driver="GPKG", mode=mode)
        first = False


def ensure_gpkg_writable(path: str):
    """
    Ensure the output directory exists and the target GeoPackage is removed
    (including SQLite sidecars) so we can perform a clean write.

    Remove an existing GeoPackage and any SQLite sidecar files to guarantee a clean write.
        Sidecars include:
          - <file>.gpkg-wal  (Write-Ahead Log)
          - <file>.gpkg-shm  (Shared memory)
          - <file>.gpkg-journal (Rollback journal when WAL is off)


    """
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    gpkg = Path(path)
    sidecars = [
        gpkg,
        gpkg.with_name(gpkg.name + "-wal"),
        gpkg.with_name(gpkg.name + "-shm"),
        gpkg.with_name(gpkg.name + "-journal"),
    ]
    for p in sidecars:
        try:
            if p.exists():
                p.unlink()
                logger.debug(f"Removed existing file: {p}")
        except Exception as e:
            raise RuntimeError(
                f"Unable to remove existing GeoPackage or sidecar '{p}': {e}"
            ) from e


# ---------------------------
# Raster helpers
# ---------------------------
def open_warped_vrt(
    src_path: str | None, target_crs: str, source_crs: str | None = None
) -> tuple[
    rasterio.io.DatasetReader | None,
    WarpedVRT | _TransformedRasterSampler | None,
    float | None,
]:
    """Open source raster and wrap in a WarpedVRT to `target_crs`."""
    if not src_path or not os.path.exists(src_path):
        return None, None, None
    src = rasterio.open(src_path)
    nodata = src.nodata
    if source_crs:
        vrt = _TransformedRasterSampler(src, source_crs, target_crs)
    else:
        vrt = WarpedVRT(src, crs=target_crs, resampling=Resampling.bilinear)
    return src, vrt, nodata


def sample_vrt_points(
    vrt: WarpedVRT | _TransformedRasterSampler | None,
    pts: list[Point],
    nodata: float | None,
) -> list:
    """Sample raster values at point coordinates."""
    if vrt is None or len(pts) == 0:
        return []
    coords = [(p.x, p.y) for p in pts]
    vals = [float(arr[0]) for arr in vrt.sample(coords)]
    sentinels = {-32768.0, -9999.0, -3.4e38}
    out: list[float] = []
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


# ---------------------------
# Shoreline boundary conversion & merging
# ---------------------------
def to_line_boundaries(gdf: GeoDF, target_crs: str) -> list:
    """Flatten polygonal/collection shoreline to EXTERIOR boundaries (LineString parts)."""
    shore = gdf.to_crs(target_crs).copy()
    out: list[LineString] = []
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


def merge_shore_paths(lines: list[LineString]) -> list:
    """Merge shoreline boundary segments into continuous LineStrings (per connected component)."""
    if not lines:
        return []
    u = unary_union(lines)
    if isinstance(u, LineString):
        return [u]
    lm = linemerge(u)
    if isinstance(lm, LineString):
        return [lm]
    elif isinstance(lm, MultiLineString):
        return list(lm.geoms)
    else:
        merged: list[LineString] = []
        if hasattr(lm, "geoms"):
            for g in lm.geoms:
                if isinstance(g, LineString):
                    merged.append(g)
                elif isinstance(g, MultiLineString):
                    merged.extend(list(g.geoms))
        return merged


# ---------------------------
# Densify lines & tangents
# ---------------------------
def densify_line(line: LineString, spacing_m: float) -> list:
    """Return measures along the line at ~spacing_m intervals (including 0 and end)."""
    L = line.length
    if L <= 0:
        return [0.0]
    n = max(1, math.ceil(L / spacing_m))
    return [i * L / n for i in range(n + 1)]


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


def _xy_only(coords: Iterable) -> list[tuple[float, float]]:
    """
    Convert any iterable of coordinates (2D, 3D, or with M) to a list of 2D (x, y) tuples.
    """
    out = []
    for c in coords:
        # c can be a sequence like (x, y), (x, y, z), or (x, y, z, m)
        # We only keep the first two values.
        x = float(c[0])
        y = float(c[1])
        out.append((x, y))
    return out


def force_2d_geom(
    g,
) -> (
    Point
    | LineString
    | LinearRing
    | Polygon
    | MultiPoint
    | MultiLineString
    | MultiPolygon
    | GeometryCollection
    | None
):
    """
    Return a 2D version of a Shapely geometry by dropping Z/M dimensions.
    - Preserves the original geometry type.
    - Handles all core geometry types, including multi- and collections.
    - Returns None if input is None.

    Notes:
    - If the geometry is already 2D, it returns an equivalent 2D geometry.
    - For Polygon, both exterior and interior rings are converted to 2D.
    """
    if g is None:
        return None

    gt = g.geom_type

    if gt == "Point":
        # g.coords[0] is a single coordinate
        xy = _xy_only(g.coords)
        return Point(xy[0])

    elif gt == "LineString":
        return LineString(_xy_only(g.coords))

    elif gt == "LinearRing":
        return LinearRing(_xy_only(g.coords))

    elif gt == "Polygon":
        # Exterior
        exterior_xy = _xy_only(g.exterior.coords) if g.exterior is not None else []
        # Interiors (holes)
        interiors_xy = []
        for ring in g.interiors:
            interiors_xy.append(_xy_only(ring.coords))
        return Polygon(exterior_xy, interiors_xy)

    elif gt == "MultiPoint":
        pts = [Point(xy) for xy in _xy_only([p.coords[0] for p in g.geoms])]
        return MultiPoint(pts)

    elif gt == "MultiLineString":
        lines = []
        for ls in g.geoms:
            lines.append(LineString(_xy_only(ls.coords)))
        return MultiLineString(lines)

    elif gt == "MultiPolygon":
        polys = []
        for p in g.geoms:
            exterior_xy = _xy_only(p.exterior.coords) if p.exterior is not None else []
            interiors_xy = []
            for ring in p.interiors:
                interiors_xy.append(_xy_only(ring.coords))
            polys.append(Polygon(exterior_xy, interiors_xy))
        return MultiPolygon(polys)

    elif gt == "GeometryCollection":
        geoms_2d = []
        for sub in g.geoms:
            geoms_2d.append(force_2d_geom(sub))
        return GeometryCollection(geoms_2d)

    else:
        # Unknown/unsupported type: return as-is (or raise if you prefer)
        return g


# ---------------------------
# Smoothers
# ---------------------------


def _gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """Compute Gaussian weights; zero out tiny values for numerical stability."""
    if sigma <= 0:
        return np.ones_like(distances)
    w = np.exp(-0.5 * (distances / sigma) ** 2)
    w[w < 1e-9] = 0.0
    return w


def smooth_tangents_for_samples(
    line: LineString,
    measures: list[float],
    eps_for_tangent_m: float,
    smooth_m: float,
    max_turn_deg: float | None = None,
) -> list[np.ndarray]:
    """
    Smooth unit tangent vectors along a line using a Gaussian moving-average over distance.
    Optionally cap per-step turn angle to max_turn_deg to prevent abrupt flips and crossings.

    Parameters
    ----------
    line : LineString
        The merged shoreline path.
    measures : list[float]
        Measure positions along the line (e.g., from densify_line).
    eps_for_tangent_m : float
        Small window for local tangent estimation (forward/backward interpolation).
    smooth_m : float
        Gaussian window size in meters; set 0 to disable smoothing.
    max_turn_deg : float, optional
        Maximum allowed change in angle between consecutive smoothed tangents.

    Returns
    -------
    list[np.ndarray]
        Unit tangent vectors for each measure (2D).
    """
    # Raw unit tangents
    raw = [estimate_tangent(line, m, eps=eps_for_tangent_m) for m in measures]
    raw = [t / (np.linalg.norm(t) + 1e-12) for t in raw]

    # Gaussian smoothing
    if smooth_m <= 0:
        smoothed = raw
    else:
        sigma = smooth_m / 2.0
        smoothed = []
        s = np.array(measures, dtype=float)
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

    # Per-step turn cap (optional)
    if max_turn_deg is not None and max_turn_deg > 0 and len(smoothed) >= 2:
        capped = [smoothed[0]]
        max_turn_rad = math.radians(max_turn_deg)
        for i in range(1, len(smoothed)):
            a = capped[-1]
            b = smoothed[i]
            # Angular difference
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            ang = math.acos(dot)
            if ang <= max_turn_rad:
                capped.append(b)
                continue
            # Rotate 'a' toward 'b' by max_turn_rad (signed by cross)
            sign = np.sign(a[0] * b[1] - a[1] * b[0])
            theta = max_turn_rad * sign
            c = math.cos(theta)
            s_ = math.sin(theta)
            R = np.array([[c, -s_], [s_, c]])
            a_rot = R @ a
            capped.append(a_rot / (np.linalg.norm(a_rot) + 1e-12))
        smoothed = capped

    return smoothed


# ---------------------------
# STRtree builders & queries
# ---------------------------
def _only_line_like(gdf: GeoDF | None) -> GeoDF | None:
    if gdf is None or len(gdf) == 0:
        return gdf
    types = gdf.geometry.type.astype(str)
    sel = types.isin(["LineString", "MultiLineString"])
    return gdf.loc[sel].copy()


def build_boundary_strtree_from_polygons_exterior_only(
    polys_gdf: GeoDF | None, crs_str: str, src_label: str
) -> tuple[STRtree | None, dict[int, tuple[str, int]]]:
    """Create a STRtree of polygon exteriors; returns (tree, idmap)."""
    if polys_gdf is None or len(polys_gdf) == 0:
        return None, {}
    polys = polys_gdf.to_crs(crs_str)
    lines: list[LineString] = []
    for geom in polys.geometry:
        if isinstance(geom, Polygon):
            lines.append(LineString(geom.exterior.coords))
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                lines.append(LineString(p.exterior.coords))
    if not lines:
        return None, {}
    tree = STRtree(lines)
    idmap: dict[int, tuple[str, int]] = {}
    for i, g in enumerate(tree.geometries):
        idmap[id(g)] = (src_label, i)
    return tree, idmap


def build_strtree_from_lines(
    lines_gdf: GeoDF | None, crs_str: str, src_label: str
) -> tuple[STRtree | None, dict[int, tuple[str, int]]]:
    """Create a STRtree from line features; returns (tree, idmap)."""
    if lines_gdf is None or len(lines_gdf) == 0:
        return None, {}
    lines = lines_gdf.to_crs(crs_str).geometry.tolist()
    lines = [g for g in lines if g is not None and not g.is_empty]
    if not lines:
        return None, {}
    tree = STRtree(lines)
    idmap = {id(g): (src_label, i) for i, g in enumerate(tree.geometries)}
    return tree, idmap


def build_asset_tree(
    gdf: GeoDF | None, crs_str: str, src_label: str, kind: str = "auto"
) -> tuple[STRtree | None, dict[int, tuple[str, int]]]:
    """Build an STRtree for an asset layer, treating it as lines or polygon exteriors.

    kind: "line", "polygon", or "auto" (decided from the dominant geometry type).
    """
    if gdf is None or len(gdf) == 0:
        return None, {}

    resolved = kind
    if kind == "auto":
        types = gdf.geometry.type.astype(str)
        n_poly = int(types.isin(["Polygon", "MultiPolygon"]).sum())
        n_line = int(types.isin(["LineString", "MultiLineString"]).sum())
        resolved = "polygon" if n_poly > n_line else "line"

    if resolved == "polygon":
        return build_boundary_strtree_from_polygons_exterior_only(
            gdf, crs_str, src_label
        )
    if resolved == "line":
        return build_strtree_from_lines(_only_line_like(gdf), crs_str, src_label)
    raise ValueError(f"Unknown layer kind '{kind}' for '{src_label}'.")


def query_tree_geoms(
    tree: STRtree | None, geom: BaseGeometry, buffer_tol: float = 0.0
) -> list:
    """Query a STRtree with optional buffer around the query geometry."""
    if tree is None or geom is None or geom.is_empty:
        return []
    qgeom = geom.buffer(buffer_tol) if buffer_tol and buffer_tol > 0 else geom
    result = tree.query(qgeom)
    if len(result) == 0:
        return []
    # Shapely 2.x returns geometries directly
    if isinstance(result[0], BaseGeometry):
        return result
    else:
        return [tree.geometries[i] for i in result]


# ---------------------------
# Intersection helper
# ---------------------------
def _decompose_to_points(geom: BaseGeometry) -> list:
    """Extract representative points from an intersection geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, (LineString, LinearRing)):
        return [Point(geom.coords[0]), Point(geom.coords[-1])]
    pts: list[Point] = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            pts.extend(_decompose_to_points(g))
    return pts


def first_intersection_on_transect(
    transect: LineString,
    geoms: Sequence[BaseGeometry],
    src_map: Mapping[int, tuple[str | None, int | None]],
    snap_tol: float = 1.0,
    origin_eps: float = 0.5,
) -> tuple[Point | None, str | None, float | None, int | None]:
    """
    Return the nearest hit along `transect`:
      - Prefer strict intersections.
      - If none, treat any geometry within snap_tol as near-miss and use the
        nearest point ON THE TRANSECT as the hit.
      - Ignore contacts at or extremely close to the origin (epsilon).
    Returns: (hit_point, source_label, distance_m_along, source_index)
    """
    if transect is None or transect.is_empty or len(geoms) == 0:
        return None, None, None, None

    best_dist = None
    best_hit = None
    best_src = None
    best_idx = None

    for g in geoms:
        if g is None or g.is_empty:
            continue

        inter = transect.intersection(g)
        candidates = _decompose_to_points(inter)

        # If no intersection, consider snappable near-miss
        if len(candidates) == 0:
            # nearest point ON transect to geometry
            np_on_g, np_on_t = nearest_points(g, transect)
            if np_on_t.distance(np_on_g) <= snap_tol:
                candidates = [np_on_t]

        for p in candidates:
            d = transect.project(p)
            if d <= origin_eps:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_hit = p
                src = src_map.get(id(g), (None, None))
                best_src, best_idx = src

    if best_hit is None:
        return None, None, None, None

    return best_hit, best_src, best_dist, best_idx


def _dedup_geoms_by_wkb(tagged_geoms):
    """Input: list of (geom, asset_label). Output: list of (geom, asset_label) deduped by WKB."""
    seen = set()
    out = []
    for g, lbl in tagged_geoms:
        w = g.wkb
        if w not in seen:
            seen.add(w)
            out.append((g, lbl))
    return out


def _detect_origin_barrier_hit(
    origin: Point,
    tagged_geoms: list,
    tol_m: float = 0.5,
) -> dict | None:
    """
    Return a dict describing an origin hit if any barrier touches/overlaps at the origin.
    Checks distance <= tol_m, touches(), or intersects() with the origin point.
    Handles both point touches and colinear (line overlap) cases.
    """
    # Prefer armoring over other types if multiple at origin (typical need)
    priority = {"armoring": 0, "levees": 0.5, "buildings": 1, "roads": 2, "rail": 3}
    candidates = []
    for g, lbl in tagged_geoms:
        try:
            if g.distance(origin) <= tol_m or g.touches(origin) or g.intersects(origin):
                candidates.append((priority.get(lbl, 99), g, lbl))
        except GEOSException:
            logger.debug(
                "Skipping invalid geometry during origin-hit test.", exc_info=True
            )
            continue
    if not candidates:
        return None
    # Choose highest priority
    candidates.sort(key=lambda t: t[0])
    _, g, lbl = candidates[0]
    # If overlap returns LineString, we still tag dist=0 at origin
    return {"geometry": origin, "asset_type": lbl, "dist_m": 0.0, "hit_geom": g}
