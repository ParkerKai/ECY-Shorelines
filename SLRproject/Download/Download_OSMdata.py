# -*- coding: utf-8 -*-
"""
Created on 07/23/2026

Download OpenStreetMap building footprints for a given county.

Approach:
- Resolve the county relation via Overpass (no Nominatim).
- Compute area_id = 3600000000 + relation_id (standard Overpass convention).
- POST Overpass QL directly to /api/interpreter with UA+Accept headers.
- Request full geometry ("out body geom") and assemble polygons reliably.
- Write GeoPackage output.

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
import json
import re
import time
import random
from pathlib import Path

import ssl
import urllib.request
import urllib.parse

import geopandas as gpd

from shapely.geometry import shape, box, Polygon, LinearRing, LineString
from shapely.ops import unary_union, polygonize

# ===============================================================================
# User Inputs
# ===============================================================================
dir_out = r"N:\SEA\Kai\Data\Assets"
county_want = [
    #"Clallam County", 
            #    "Island County",
            #    "Grays Harbor County",
            #    "Jefferson County",
            #    "King County",
            #    "Kitsap County",
            #    "Mason County",
            #    "Pacific County",
            #    "Pierce County",
            #    "San Juan County",
            #    "Skagit County",
            #    "Snohomish County", 
            #    "Thurston County",
               "Wahkiakum County",
               "Whatcom County"]

state_name  = "Washington"
country     = "United States"

# Optional local projection (None => stay in WGS84/EPSG:4326)
TARGET_CRS = None  # e.g., "EPSG:32149" (WA State Plane North, meters) or "EPSG:2927" (US feet)

# Include building parts? (more detailed, larger result)
INCLUDE_BUILDING_PARTS = True

# Optional tiling to avoid Overpass timeouts for big areas (King/Pierce/Snohomish)
USE_TILING = False             # set True for large counties
TILE_SIZE_DEG = 0.25           # 0.25° (~25–30 km). Adjust per need.

# Optional boundary check before full pull
CHECK_BOUNDARY = True

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
# Harden headers: UA, Accept (and optional Referer)
UA_STR = "ECY-BuildingFootprints/1.0 (kaip461@ecy.wa.gov)"
opener.addheaders = [
    ("User-Agent", UA_STR),
    ("Accept", "application/json"),
    # ("Referer", "https://ecology.wa.gov/"),  # optional, some mirrors like a referer for browser-like clients
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
def _post_overpass_json(ql: str, timeout_s: int = 300, max_retries: int = 4, base_sleep: int = 6):
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
                req = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
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
                print(f"  -> interpreter attempt {attempt} failed on {endpoint}: {http_err} (sleep {wait:.1f}s)")
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                wait = base_sleep * attempt + random.uniform(0, base_sleep)
                print(f"  -> interpreter attempt {attempt} failed on {endpoint}: {exc} (sleep {wait:.1f}s)")
                time.sleep(wait)
        print(f"Switching interpreter endpoint after repeated failures: {endpoint}")
    raise RuntimeError(f"Overpass interpreter failed across endpoints. Last error: {last_exc}")

# ===============================================================================
# Overpass helper: list counties in state (admin_level=6)
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
            counties.append({
                "name": nm,
                "name_en": nm_en,
                "official_name": official,
                "osm_id": rel.get("id"),  # relation id
            })
    counties = sorted(counties, key=lambda c: c["name"])
    print("\nCounties in", state_name)
    for i, c in enumerate(counties, 1):
        suffix = f"  (official: {c['official_name']})" if c["official_name"] else ""
        print(f"{i:2d}. {c['name']}{suffix}")
    return counties

def choose_county(counties: list, idx: int | None = None, name: str | None = None) -> str:
    """
    Pick a county name from the list by index (1-based) or by exact name match.
    """
    if idx is not None:
        if 1 <= idx <= len(counties):
            return counties[idx - 1]["name"]
        raise ValueError(f"Index out of range (1..{len(counties)})")
    if name is not None:
        for c in counties:
            if c["name"].lower() == name.lower():
                return c["name"]
        raise ValueError(f"County name '{name}' not found.")
    raise ValueError("Provide either idx or name to choose a county.")

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
        ls = LineString(coords)
        role = m.get("role")
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
        return unary_union([_safe_make_valid(p.difference(holes_union)) for p in outer_poly]) if outer_poly else None
    return unary_union(outer_poly) if outer_poly else None

def _way_to_polygon(raw_way):
    """
    Build a polygon from a way's geometry (expects 'out geom' coordinates).
    Ensures ring closure and repairs invalid rings.
    """
    coords = [(pt["lon"], pt["lat"]) for pt in raw_way.get("geometry", [])]
    if not coords:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])  # enforce closed ring

    ring = LinearRing(coords)
    if not ring.is_valid:
        try:
            repaired = _safe_make_valid(LineString(coords))
            ring = LinearRing(list(repaired.coords))
        except Exception:
            pass
    try:
        return Polygon(ring)
    except Exception:
        return _safe_make_valid(Polygon(ring))

# ===============================================================================
# Building fetch (area-based), union of building + building:part, returns GeoDataFrame
# ===============================================================================
def fetch_buildings(area_id: int, include_parts: bool = True, timeout_s: int = 1200):
    """
    Run Overpass building query for the given area_id and assemble GeoDataFrame.
    """
    parts_clause = 'way(area.searchArea)["building:part"]; relation(area.searchArea)["building:part"];' if include_parts else ''
    ql = f"""
    [out:json][timeout:{timeout_s}];
    area({area_id})->.searchArea;
    (
      way(area.searchArea)["building"];
      relation(area.searchArea)["building"];
      {parts_clause}
    );
    out body geom;
    """
    js = _post_overpass_json(ql, timeout_s=timeout_s)
    elements = js.get("elements", [])
    if not elements:
        remark = js.get("remark") or js.get("message")
        raise RuntimeError(f"Overpass returned no elements. {('Remark: ' + remark) if remark else ''}")

    records = []
    for raw in elements:
        g = None
        if raw.get("type") == "way":
            g = _way_to_polygon(raw)
        elif raw.get("type") == "relation":
            g = _relation_to_multipolygon(raw)
        if g is None:
            continue

        props = {"osm_id": raw.get("id"), "osm_type": raw.get("type")}
        props.update(raw.get("tags", {}) or {})
        records.append({**props, "geometry": g})

    if not records:
        raise RuntimeError("No building geometries assembled from Overpass output. Check filters and endpoint health.")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").set_geometry("geometry")
    return gdf

# ===============================================================================
# Optional: Tile-based fetch for big areas (bbox slicing)
# ===============================================================================
def _frange(a, b, step):
    vals = []
    x = a
    while x < b:
        vals.append(x)
        x += step
    vals.append(b)
    return vals

def fetch_buildings_tiled(area_id: int, county_bbox: tuple[float, float, float, float],
                          include_parts: bool = True, timeout_s: int = 1200, tile_size_deg: float = 0.25):
    """
    Tile the county bbox and run restricted queries per tile.
    bbox: (minx, miny, maxx, maxy) in EPSG:4326.
    Overpass QL bbox order is (south, west, north, east).
    """
    minx, miny, maxx, maxy = county_bbox
    xs = list(_frange(minx, maxx, tile_size_deg))
    ys = list(_frange(miny, maxy, tile_size_deg))

    all_records = []
    seen_ids = set()

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            west, south, east, north = xs[i], ys[j], xs[i+1], ys[j+1]
            parts_union = ''
            if include_parts:
                parts_union = f"""
                  way(area.searchArea)["building:part"];
                  relation(area.searchArea)["building:part"];
                """

            # Restrict geometry output to the tile bbox to reduce payload
            ql = f"""
            [out:json][timeout:{timeout_s}];
            area({area_id})->.searchArea;
            (
              way(area.searchArea)["building"];
              relation(area.searchArea)["building"];
              {parts_union}
            );
            out body geom ({south},{west},{north},{east});
            """
            try:
                js = _post_overpass_json(ql, timeout_s=timeout_s)
                elements = js.get("elements", [])
                print(f"  Tile ({west},{south},{east},{north}) -> {len(elements)} elements")
                for raw in elements:
                    oid = raw.get("id")
                    if oid in seen_ids:
                        continue
                    g = None
                    if raw.get("type") == "way":
                        g = _way_to_polygon(raw)
                    elif raw.get("type") == "relation":
                        g = _relation_to_multipolygon(raw)
                    if g is None:
                        continue
                    props = {"osm_id": oid, "osm_type": raw.get("type")}
                    props.update(raw.get("tags", {}) or {})
                    all_records.append({**props, "geometry": g})
                    seen_ids.add(oid)
            except Exception as exc:
                print(f"  Tile ({west},{south},{east},{north}) failed: {exc}")

    if not all_records:
        raise RuntimeError("Tiled fetch produced no building geometries.")
    gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326").set_geometry("geometry")
    return gdf

# ===============================================================================
# Main flow
# ===============================================================================
# 1) List counties and pick the target
counties = list_counties_in_state(state_name, timeout_s=300)

for county_name in county_want:

    # Use the pre-set county_name
    print(f"\nUsing county_name: {county_name}\n")

    county_rel_id = next((c["osm_id"] for c in counties if c["name"].lower() == county_name.lower()), None)
    if county_rel_id is None:
        raise RuntimeError(f"County '{county_name}' not found in {state_name} list.")

    # Overpass area id: relation + 3_600_000_000
    area_id = 3600000000 + int(county_rel_id)
    print(f"Resolved area_id via Overpass: {area_id} (relation {county_rel_id})")

    # Optional: boundary sanity check
    county_bbox = None
    if CHECK_BOUNDARY:
        test_ql = f"""
        [out:json][timeout:120];
        area({area_id})->.a;
        rel(area.a)["boundary"="administrative"]["admin_level"="6"]["name"="{county_name}"];
        out body geom;
        """
        js = _post_overpass_json(test_ql, timeout_s=120)
        els = js.get("elements", [])
        ok = bool(els)
        remark = js.get("remark")
        print(f"[Boundary check] OK={ok} Remark={remark or 'None'}")
        if ok:
            # derive bbox from returned relation geometry
            # (Use union of member way geometries if needed)
            # For simplicity, take bounds of the union of outers
            rel_geom = _relation_to_multipolygon(els[0])
            if rel_geom:
                county_bbox = rel_geom.bounds  # (minx, miny, maxx, maxy) in EPSG:4326

    # 2) Run building query (area-based or tiled)
    if USE_TILING and county_bbox:
        print(f"Using tiling with {TILE_SIZE_DEG}° tiles.")
        gdf = fetch_buildings_tiled(area_id, county_bbox, include_parts=INCLUDE_BUILDING_PARTS,
                                    timeout_s=1200, tile_size_deg=TILE_SIZE_DEG)
    else:
        gdf = fetch_buildings(area_id, include_parts=INCLUDE_BUILDING_PARTS, timeout_s=1200)

    print(f"Collected {len(gdf)} features.")

    # 3) Optional reprojection
    if TARGET_CRS:
        try:
            gdf = gdf.to_crs(TARGET_CRS)
            print(f"Reprojected to {TARGET_CRS}.")
        except Exception as exc:
            print(f"Reprojection to {TARGET_CRS} failed (keeping EPSG:4326): {exc}")

    # 4) Write to GeoPackage
    safe_layer = re.sub(r"[^A-Za-z0-9_]", "_", "buildings")
    out_dir = Path(dir_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{county_name.replace(' ', '_').lower()}_buildings.gpkg"

    gdf.to_file(out_path, layer=safe_layer, driver="GPKG")
    print(f"Wrote {len(gdf)} features to {out_path} (layer={safe_layer})")
    print("Data © OpenStreetMap contributors, ODbL 1.0")
