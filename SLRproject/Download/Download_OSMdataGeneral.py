"""
Created on 07/24/2026

Generalized OSM feature pull for given counties:
- Resolve county relations via Overpass (no Nominatim).
- Compute area_id = 3600000000 + relation_id (Overpass area convention).
- POST Overpass QL directly to /api/interpreter with UA + Accept headers.
- Request full geometry ("out body geom"), classify nodes/ways/relations to
  points/lines/polygons, and write GeoPackage layers.
- Always use tiling across the county bbox to avoid timeouts.
- Include OSM base timestamp (osm3s.timestamp_osm_base) for provenance.

Interim TLS behavior:
- Keep hostname & CA verification ON.
- Relax VERIFY_X509_STRICT (OpenSSL 3+) to tolerate corporate SSL inspection.
- Remove once IT fixes CA or adds SSL-decrypt bypass for Overpass domains.

@author: Kai Parker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# ===============================================================================
# Imports
# ===============================================================================
import re
import json
import time
import random
from pathlib import Path
import ssl
import urllib.request
import urllib.parse
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    LinearRing,
)

from shapely.ops import unary_union, polygonize

# ===============================================================================
# User Inputs
# ===============================================================================

# Output directory
dir_out = r"N:\SEA\Kai\Data\Assets"

# Target geography
state_name = "Washington"
country = "United States"

# Counties to pull (exact OSM 'name' field; set to None to run ALL counties in the state)
county_want = [
    # "Clallam County",
    # "Island County",
    # "Grays Harbor County",
    # "Jefferson County",
    # "King County",
    # "Kitsap County",
    # "Mason County",
    # "Pacific County",
    # "Pierce County",
    # "San Juan County",
    # "Skagit County",
    # "Snohomish County",
    # "Thurston County",
    "Wahkiakum County",
    # "Whatcom County",
]

# county_want = None  # uncomment to run all counties in the state

# Generalized OSM keys (choose any key; optionally restrict by values or regex)
# Examples:
#   {"key": "amenity"}                         -> amenity=*
#   {"key": "amenity", "values": ["school"]}   -> amenity in {school}
#   {"key": "landuse", "regex": "^(residential|industrial)$"}
# KEY_SPECS = [
#    {"key": "building"},
# {"key": "amenity", "values": ["school", "hospital"]},
# {"key": "landuse", "regex": "^(residential|industrial)$"},
# {"key": "power", "values": ["substation", "plant"]},
# {"key": "highway"},  # large payload; consider enabling only when needed
# ]

KEY_SPECS = [
    {"key": "railway"},  # exact value via regex ^(primary)$
]

# Geometry types to include
INCLUDE_NODES = False
INCLUDE_WAYS = True
INCLUDE_RELATIONS = False

# Heuristic: treat closed ways as polygons; otherwise lines unless area=yes
CLOSED_WAY_AS_POLYGON = False

# Projection (None => keep WGS84/EPSG:4326)
TARGET_CRS = (
    None  # e.g., "EPSG:32149" (WA State Plane North, meters) or "EPSG:2927" (US feet)
)

# Tiling settings (enabled for ALL counties)
USE_TILING = True
TILE_SIZE_DEG = 0.25  # 0.25° (~25–30 km). Adjust per need.
INTER_COUNTY_PAUSE_S = 5  # polite pause between counties

# ===============================================================================
# Interim TLS context (relax strict extension checks, but keep verification)
# ===============================================================================
ctx = ssl.create_default_context()
if hasattr(ssl, "VERIFY_X509_STRICT"):
    try:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    except Exception:
        pass  # Continue with default verification

# Install a global opener so urllib inherits the context
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
UA_STR = "ECY-OSMFeaturePull/1.0 (kaip461@ecy.wa.gov)"
opener.addheaders = [
    ("User-Agent", UA_STR),
    ("Accept", "application/json"),
    # ("Referer", "https://ecology.wa.gov/"),  # optional: some mirrors prefer a referer
]
urllib.request.install_opener(opener)
print("[http] Installed global opener with UA/Accept headers.")

# ===============================================================================
# Overpass interpreter endpoints (rotate for resilience)
# ===============================================================================
INTERPRETER_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


# ===============================================================================
# Utility: Shapely make_valid across versions
# ===============================================================================
def _safe_make_valid(geom):
    try:
        # Shapely >= 2
        from shapely.validation import make_valid

        return make_valid(geom)
    except Exception:
        # Robust fallback that fixes many self-intersections
        return geom.buffer(0)


# ===============================================================================
# HTTP: direct POST to Overpass /api/interpreter with rotation/backoff
# ===============================================================================
def _post_overpass_json(
    ql: str, timeout_s: int = 300, max_retries: int = 4, base_sleep: int = 6
):
    """
    Post Overpass QL directly to /api/interpreter using urllib + global HTTPS opener.
    Rotates endpoints and retries with jittered backoff. Returns parsed JSON dict.
    Handles HTTP 429 (Retry-After).
    """
    last_exc = None
    encoded = urllib.parse.urlencode({"data": ql}).encode("utf-8")
    headers = {
        "User-Agent": UA_STR,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    for endpoint in INTERPRETER_ENDPOINTS:
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint, data=encoded, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read()
                    return json.loads(body.decode("utf-8", errors="ignore"))
            except urllib.error.HTTPError as http_err:
                last_exc = http_err
                wait = base_sleep * attempt + random.uniform(0, base_sleep)
                # Honor Retry-After if provided on 429
                if http_err.code == 429:
                    ra = http_err.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except Exception:
                            pass
                print(
                    f"  -> interpreter attempt {attempt} failed on {endpoint}: {http_err} (sleep {wait:.1f}s)"
                )
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                wait = base_sleep * attempt + random.uniform(0, base_sleep)
                print(
                    f"  -> interpreter attempt {attempt} failed on {endpoint}: {exc} (sleep {wait:.1f}s)"
                )
                time.sleep(wait)
        print(f"Switching interpreter endpoint after repeated failures: {endpoint}")
    raise RuntimeError(
        f"Overpass interpreter failed across endpoints. Last error: {last_exc}"
    )


# ===============================================================================
# Overpass helpers: list counties in state (admin_level=6), choose county, bbox
# ===============================================================================
def list_counties_in_state(state_name: str, timeout_s: int = 300):
    """
    Enumerate county boundaries (admin_level=6) within the given state (admin_level=4),
    print a numbered list, and return a sorted list of dicts.
    """
    ql = f"""
    [out:json][timeout:{timeout_s}];
    area[name="{state_name}"]["boundary"="administrative"]["admin_level"="4"]->.state;
    rel(area.state)["boundary"="administrative"]["admin_level"="6"];
    out tags;
    """
    js = _post_overpass_json(ql, timeout_s=timeout_s)
    items = js.get("elements", [])
    counties = []
    for rel in items:
        tags = rel.get("tags", {}) or {}
        nm = tags.get("name")
        nm_en = tags.get("name:en")
        official = tags.get("official_name") or tags.get("long_name")
        if nm:
            counties.append(
                {
                    "name": nm,
                    "name_en": nm_en,
                    "official_name": official,
                    "osm_id": rel.get("id"),  # relation id
                }
            )
    counties = sorted(counties, key=lambda c: c["name"])
    print("\nCounties in", state_name)
    for i, c in enumerate(counties, 1):
        suffix = f"  (official: {c['official_name']})" if c["official_name"] else ""
        print(f"{i:2d}. {c['name']}{suffix}")
    return counties


def get_county_bbox(area_id: int, county_name: str, timeout_s: int = 120):
    """
    Request the county relation and return its bbox via 'out bb' (south, west, north, east).
    Returns (minx, miny, maxx, maxy) in EPSG:4326 or None if not available.
    """
    ql = f"""
    [out:json][timeout:{timeout_s}];
    area({area_id})->.a;
    rel(area.a)["boundary"="administrative"]["admin_level"="6"]["name"="{county_name}"];
    out bb;
    """
    js = _post_overpass_json(ql, timeout_s=timeout_s)
    els = js.get("elements", [])
    if not els:
        return None
    bb = els[0].get("bounds") or {}
    south, west = bb.get("minlat"), bb.get("minlon")
    north, east = bb.get("maxlat"), bb.get("maxlon")
    if None in (south, west, north, east):
        return None
    # Convert to (minx, miny, maxx, maxy)
    return (west, south, east, north)


# ===============================================================================
# Filter builder & query composition
# ===============================================================================
def _build_filter_fragment(spec: dict) -> str:
    """
    Converts a spec like {"key":"amenity", "values":["school","hospital"]} to
    Overpass filter: ["amenity"~"^(school|hospital)$"]
    If only {"key":"amenity"} -> ["amenity"]
    If regex: ["amenity"~"<regex>"]
    """
    key = spec.get("key")
    if not key:
        raise ValueError("KEY_SPECS item missing 'key'")

    values = spec.get("values")
    regex = spec.get("regex")
    if values and regex:
        raise ValueError(
            f"Provide either 'values' or 'regex' for key '{key}', not both."
        )

    if values:
        import re as _re

        pat = "^(" + "|".join(_re.escape(v) for v in values) + ")$"
        return f'["{key}"~"{pat}"]'
    elif regex:
        return f'["{key}"~"{regex}"]'
    else:
        return f'["{key}"]'


def _compose_overpass_union(
    area_id: int,
    specs: list,
    include_nodes=True,
    include_ways=True,
    include_relations=True,
    timeout_s: int = 1200,
    bbox: tuple | None = None,
) -> str:
    """
    Builds a single Overpass QL union across node/way/relation for given specs.
    If bbox is provided, it is applied as a **query filter** on each statement
    (south, west, north, east) to bound selection to the tile. Geometry is
    returned in full (no geom-cropping).
    """
    filters = [_build_filter_fragment(s) for s in specs]
    chunks = []

    # If bbox provided, format as Overpass statement bbox: (south, west, north, east)
    bbox_stmt = ""
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        south, west, north, east = miny, minx, maxy, maxx
        bbox_stmt = f"({south},{west},{north},{east})"

    if include_nodes:
        for f in filters:
            chunks.append(f"node(area.searchArea){f}{bbox_stmt};")
    if include_ways:
        for f in filters:
            chunks.append(f"way(area.searchArea){f}{bbox_stmt};")
    if include_relations:
        for f in filters:
            chunks.append(f"relation(area.searchArea){f}{bbox_stmt};")

    ql = f"""
    [out:json][timeout:{timeout_s}];
    area({area_id})->.searchArea;
    (
      {"\n      ".join(chunks)}
    );
    out body geom;  /* full geometry; no crop here */
    """
    return ql


# ===============================================================================
# Geometry assembly utilities
# ===============================================================================
def _relation_to_multipolygon(raw_rel):
    """
    Assemble a multipolygon from relation members with roles 'outer'/'inner'.
    Expects Overpass 'out geom' members with coordinates present.
    """
    outers, inners = [], []
    for m in raw_rel.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"]]
        if len(coords) < 2:
            continue
        role = m.get("role")
        ls = LineString(coords)
        if role == "outer":
            outers.append(ls)
        elif role == "inner":
            inners.append(ls)

    if not outers and not inners:
        return None

    outer_poly = list(polygonize(unary_union(outers))) if outers else []
    inner_poly = list(polygonize(unary_union(inners))) if inners else []
    holes_union = unary_union(inner_poly) if inner_poly else None

    if holes_union and not holes_union.is_empty:
        outers_valid = [_safe_make_valid(p) for p in outer_poly]
        return (
            unary_union(
                [_safe_make_valid(p.difference(holes_union)) for p in outers_valid]
            )
            if outers_valid
            else None
        )
    return unary_union(outer_poly) if outer_poly else None


def _node_to_point(raw_node):
    lon, lat = raw_node.get("lon"), raw_node.get("lat")
    if lon is None or lat is None:
        return None
    return Point(lon, lat)


def _way_to_geom(raw_way, closed_as_poly=True):
    """
    Returns Polygon for closed ways (or area=yes), otherwise LineString.
    """
    coords = [(pt["lon"], pt["lat"]) for pt in raw_way.get("geometry", [])]
    if not coords:
        return None
    tags = raw_way.get("tags") or {}
    is_closed = coords[0] == coords[-1]
    is_area_tag = tags.get("area") == "yes"

    if (closed_as_poly and is_closed) or is_area_tag:
        if not is_closed:
            coords.append(coords[0])
        try:
            ring = LinearRing(coords)
            return Polygon(ring)
        except Exception:
            repaired = _safe_make_valid(Polygon(coords))
            if repaired and not repaired.is_empty:
                if repaired.geom_type == "MultiPolygon":
                    repaired = max(repaired.geoms, key=lambda g: g.area)
                return repaired
            return None
    else:
        try:
            return LineString(coords)
        except Exception:
            repaired = _safe_make_valid(LineString(coords))
            return repaired if repaired and not repaired.is_empty else None


def _relation_to_geom_generic(raw_rel):
    """
    If relation type is multipolygon or boundary, assemble area geometry.
    Otherwise, merge member ways into MultiLineString/LineString.
    """
    tags = raw_rel.get("tags") or {}
    rel_type = (tags.get("type") or "").lower()

    if rel_type in ("multipolygon", "boundary"):
        g = _relation_to_multipolygon(raw_rel)
        if g and not g.is_empty:
            return g

    # Fallback: treat as linework from member ways
    lines = []
    for m in raw_rel.get("members", []):
        if m.get("type") == "way" and "geometry" in m:
            coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"]]
            if len(coords) >= 2:
                try:
                    lines.append(LineString(coords))
                except Exception:
                    r = _safe_make_valid(LineString(coords))
                    if r and not r.is_empty:
                        lines.append(r)
    if not lines:
        return None
    try:
        return unary_union(lines) if len(lines) > 1 else lines[0]
    except Exception:
        return MultiLineString([ls for ls in lines if ls.is_valid])


# ===============================================================================
# Parse Overpass response into three GeoDataFrames
# ===============================================================================
def _extract_osm_timestamp(js: dict) -> str | None:
    try:
        return js.get("osm3s", {}).get("timestamp_osm_base")
    except Exception:
        return None


def parse_elements_to_layers(elements: list, osm_ts: str | None):
    """
    Splits elements into points, lines, polygons GeoDataFrames.
    """
    point_records, line_records, poly_records = [], [], []

    for raw in elements:
        t = raw.get("type")
        g = None
        tags = raw.get("tags") or {}
        props = {"osm_id": raw.get("id"), "osm_type": t}
        props.update(tags)
        if osm_ts:
            props["osm_base_ts"] = osm_ts

        if t == "node":
            g = _node_to_point(raw)
            if g is None or g.is_empty:
                continue
            point_records.append({**props, "geometry": g})

        elif t == "way":
            g = _way_to_geom(raw, closed_as_poly=CLOSED_WAY_AS_POLYGON)
            if g is None or g.is_empty:
                continue
            rec = {**props, "geometry": g}
            if g.geom_type in ("Polygon", "MultiPolygon"):
                poly_records.append(rec)
            else:
                line_records.append(rec)

        elif t == "relation":
            g = _relation_to_geom_generic(raw)
            if g is None or g.is_empty:
                continue
            rec = {**props, "geometry": g}
            if g.geom_type in ("Polygon", "MultiPolygon"):
                poly_records.append(rec)
            else:
                line_records.append(rec)

    gdf_points = (
        gpd.GeoDataFrame(point_records, crs="EPSG:4326").set_geometry("geometry")
        if point_records
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    gdf_lines = (
        gpd.GeoDataFrame(line_records, crs="EPSG:4326").set_geometry("geometry")
        if line_records
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    gdf_polys = (
        gpd.GeoDataFrame(poly_records, crs="EPSG:4326").set_geometry("geometry")
        if poly_records
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )

    return gdf_points, gdf_lines, gdf_polys


# ===============================================================================
# Range helper for tiling
# ===============================================================================
def _frange(a, b, step):
    vals = []
    x = a
    while x < b:
        vals.append(x)
        x += step
    vals.append(b)
    return vals


# ===============================================================================
# Fetch functions: tiled (always) and single-shot (fallback)
# ===============================================================================
def fetch_osm_by_keys_tiled(
    area_id: int,
    specs: list,
    county_bbox: tuple,
    include_nodes=True,
    include_ways=True,
    include_relations=True,
    timeout_s: int = 1200,
    tile_size_deg: float = 0.25,
):
    """
    Tile the county bbox and run restricted queries per tile.
    bbox: (minx, miny, maxx, maxy) in EPSG:4326.
    Overpass QL bbox order is (south, west, north, east).
    """
    minx, miny, maxx, maxy = county_bbox
    xs = list(_frange(minx, maxx, tile_size_deg))
    ys = list(_frange(miny, maxy, tile_size_deg))

    seen = set()  # (type, id)
    points_all, lines_all, polys_all = [], [], []

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            west, south, east, north = xs[i], ys[j], xs[i + 1], ys[j + 1]
            bbox = (west, south, east, north)
            ql = _compose_overpass_union(
                area_id,
                specs,
                include_nodes=include_nodes,
                include_ways=include_ways,
                include_relations=include_relations,
                timeout_s=timeout_s,
                bbox=bbox,
            )
            try:
                js = _post_overpass_json(ql, timeout_s=timeout_s)
                els = js.get("elements", [])
                ts = _extract_osm_timestamp(js)
                print(f"  Tile ({west},{south},{east},{north}) -> {len(els)} elements")

                # Dedup by (type,id) across tiles
                filtered = [e for e in els if (e.get("type"), e.get("id")) not in seen]
                for e in filtered:
                    seen.add((e.get("type"), e.get("id")))

                gdf_p, gdf_l, gdf_a = parse_elements_to_layers(filtered, ts)
                if len(gdf_p):
                    points_all.append(gdf_p)
                if len(gdf_l):
                    lines_all.append(gdf_l)
                if len(gdf_a):
                    polys_all.append(gdf_a)

            except Exception as exc:
                print(f"  Tile ({west},{south},{east},{north}) failed: {exc}")

    gdf_points = (
        gpd.GeoDataFrame(
            pd.concat(points_all, ignore_index=True), crs="EPSG:4326"
        ).set_geometry("geometry")
        if points_all
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    gdf_lines = (
        gpd.GeoDataFrame(
            pd.concat(lines_all, ignore_index=True), crs="EPSG:4326"
        ).set_geometry("geometry")
        if lines_all
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )
    gdf_polys = (
        gpd.GeoDataFrame(
            pd.concat(polys_all, ignore_index=True), crs="EPSG:4326"
        ).set_geometry("geometry")
        if polys_all
        else gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    )

    return gdf_points, gdf_lines, gdf_polys


def fetch_osm_by_keys(
    area_id: int,
    specs: list,
    include_nodes=True,
    include_ways=True,
    include_relations=True,
    timeout_s: int = 1200,
):
    """
    Non-tiled fetch (fallback). Not used when USE_TILING=True, but kept for completeness.
    """
    ql = _compose_overpass_union(
        area_id,
        specs,
        include_nodes,
        include_ways,
        include_relations,
        timeout_s=timeout_s,
        bbox=None,
    )
    js = _post_overpass_json(ql, timeout_s=timeout_s)
    elements = js.get("elements", [])
    if not elements:
        remark = js.get("remark") or js.get("message")
        raise RuntimeError(
            f"Overpass returned no elements. {('Remark: ' + remark) if remark else ''}"
        )
    ts = _extract_osm_timestamp(js)
    return parse_elements_to_layers(elements, ts)


# ===============================================================================
# Writing layers (points/lines/polygons) and reprojection
# ===============================================================================


def _normalize_gpkg_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize column names for GeoPackage/SQLite:
      - lower case
      - non-alphanumeric -> underscore
      - ensure uniqueness (suffix _1, _2, ...)
    Leaves the geometry column name unchanged.
    """
    if not len(gdf):
        return gdf

    geom_col = gdf.geometry.name
    rename = {}
    used = set([geom_col.lower()])

    for col in gdf.columns:
        if col == geom_col:
            continue

        # 1) lowercase + safe chars
        safe = re.sub(r"[^0-9a-z_]", "_", col.lower())

        # 2) don't allow empty or geometry name collision
        if safe == "" or safe == geom_col.lower():
            safe = f"{safe}_attr" if safe else "attr"

        # 3) de-dupe (case-insensitive world)
        base = safe
        i = 1
        while safe in used:
            safe = f"{base}_{i}"
            i += 1

        used.add(safe)
        rename[col] = safe

    return gdf.rename(columns=rename)


def _write_layers_to_gpkg(
    gpkg_path: Path,
    gdf_points,
    gdf_lines,
    gdf_polys,
    target_crs: str | None,
    layer_prefix: str = "osm",
):
    """
    Writes up to three layers: <prefix>_points, <prefix>_lines, <prefix>_polygons.
    Reprojects if target_crs is provided. Normalizes column names for GPKG.
    """
    layer_points = f"{layer_prefix}_points"
    layer_lines = f"{layer_prefix}_lines"
    layer_polys = f"{layer_prefix}_polygons"

    def _maybe_reproject(gdf):
        if target_crs and len(gdf):
            try:
                return gdf.to_crs(target_crs)
            except Exception as exc:
                print(f"Reprojection to {target_crs} failed (keeping EPSG:4326): {exc}")
        return gdf

    # Normalize names BEFORE write to avoid case-insensitive collisions in SQLite
    gdf_points = _normalize_gpkg_columns(_maybe_reproject(gdf_points))
    gdf_lines = _normalize_gpkg_columns(_maybe_reproject(gdf_lines))
    gdf_polys = _normalize_gpkg_columns(_maybe_reproject(gdf_polys))

    wrote_any = False
    if len(gdf_points):
        gdf_points.to_file(gpkg_path, layer=layer_points, driver="GPKG")
        print(f"  -> wrote {len(gdf_points)} features to layer '{layer_points}'")
        wrote_any = True
    if len(gdf_lines):
        gdf_lines.to_file(gpkg_path, layer=layer_lines, driver="GPKG")
        print(f"  -> wrote {len(gdf_lines)} features to layer '{layer_lines}'")
        wrote_any = True
    if len(gdf_polys):
        gdf_polys.to_file(gpkg_path, layer=layer_polys, driver="GPKG")
        print(f"  -> wrote {len(gdf_polys)} features to layer '{layer_polys}'")
        wrote_any = True

    if not wrote_any:
        print("  -> no features produced; GeoPackage not written.")


# ===============================================================================
# Main
# ===============================================================================
# 1) List counties and pick targets
counties = list_counties_in_state(state_name, timeout_s=300)

if county_want is None:
    target_counties = [c["name"] for c in counties]
else:
    target_counties = county_want

out_dir = Path(dir_out)
out_dir.mkdir(parents=True, exist_ok=True)

for county_name in target_counties:
    print(f"\nUsing county_name: {county_name}\n")

    county_rel_id = next(
        (c["osm_id"] for c in counties if c["name"].lower() == county_name.lower()),
        None,
    )
    if county_rel_id is None:
        print(f"!! County '{county_name}' not found in {state_name} list. Skipping.")
        continue

    # Overpass area id: relation + 3_600_000_000
    area_id = 3600000000 + int(county_rel_id)
    print(f"Resolved area_id via Overpass: {area_id} (relation {county_rel_id})")

    # 2) Get bbox (out bb) for tiling
    county_bbox = get_county_bbox(area_id, county_name, timeout_s=120)
    if not county_bbox:
        print(
            "  -> bbox not available via 'out bb'; attempting fallback via full relation geometry."
        )
        # Fallback: full relation geom (heavier) then bounds
        ql = f"""
        [out:json][timeout:180];
        area({area_id})->.a;
        rel(area.a)["boundary"="administrative"]["admin_level"="6"]["name"="{county_name}"];
        out body geom;
        """
        js = _post_overpass_json(ql, timeout_s=180)
        els = js.get("elements", [])
        if els:
            rel_geom = _relation_to_multipolygon(els[0])
            county_bbox = rel_geom.bounds if rel_geom else None

    if not county_bbox:
        print(
            "  -> unable to determine county bbox. Skipping tiling and doing a single request."
        )
        gdf_pts, gdf_lin, gdf_pol = fetch_osm_by_keys(
            area_id,
            KEY_SPECS,
            include_nodes=INCLUDE_NODES,
            include_ways=INCLUDE_WAYS,
            include_relations=INCLUDE_RELATIONS,
            timeout_s=1200,
        )
    else:
        # 3) Tiled fetch (enabled for ALL counties)
        print(f"Using tiling with {TILE_SIZE_DEG}° tiles.")
        gdf_pts, gdf_lin, gdf_pol = fetch_osm_by_keys_tiled(
            area_id,
            KEY_SPECS,
            county_bbox,
            include_nodes=INCLUDE_NODES,
            include_ways=INCLUDE_WAYS,
            include_relations=INCLUDE_RELATIONS,
            timeout_s=1200,
            tile_size_deg=TILE_SIZE_DEG,
        )

    print(
        f"Collected points={len(gdf_pts)} lines={len(gdf_lin)} polygons={len(gdf_pol)}"
    )

    # 4) Write to GeoPackage (per county)
    layer_prefix = "rail"  # customize per theme if desired
    gpkg_path = out_dir / f"{county_name.replace(' ', '_').lower()}_{layer_prefix}.gpkg"
    _write_layers_to_gpkg(
        gpkg_path, gdf_pts, gdf_lin, gdf_pol, TARGET_CRS, layer_prefix=layer_prefix
    )

    print("Data © OpenStreetMap contributors, ODbL 1.0")
    # Polite pause between counties (even on success)
    time.sleep(INTER_COUNTY_PAUSE_S + random.uniform(0, 3))
