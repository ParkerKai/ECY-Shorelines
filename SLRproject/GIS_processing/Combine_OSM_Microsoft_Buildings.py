# -*- coding: utf-8 -*-
"""
County-parameterized union of OSM + Microsoft building footprints.

- Merges only overlapping polygons (non-zero area intersection).
- Excludes pure boundary-touching unless MERGE_WHEN_TOUCHES=True (default=False).
- Tags source: 'OSM', 'Microsoft', 'OSM+Microsoft'.
- Outputs to GeoPackage with county-driven file and layer names.
- Robust to geographic CRS inputs (EPSG:4326); avoids precision snapping in degrees.
- Faster pair assembly via GeoPandas sjoin (vectorized), with safe fallback.

Author: Kai Parker (ECY) | Regenerated: 2026-08-14
"""

import os
import sys
import argparse
import warnings
from typing import List, Dict, Tuple

import geopandas as gpd
import shapely
from shapely import set_precision

# ---------------------------
# Configuration (defaults; override via CLI if desired)
# ---------------------------

COUNTY_LIST = ["Clallam", "Grays_Harbor","Island",
               "Jefferson","King","Kitsap","Mason",
               "Pacific","Pierce","San Juan","Skagit",
               "Snohomish","Thurston","Wahkiakum",
               "Whatcom"]  
# Base directories (adjust if you ever reorganize)
BASE_DIR = r"N:\SEA\Kai\Data\Assets"
OSM_DIR = os.path.join(BASE_DIR, "OSM")
MS_DIR  = os.path.join(BASE_DIR, "Microsoft")
OUT_DIR = os.path.join(BASE_DIR, "Unified")

# Optional: precision grid to reduce micro-slivers (units in CRS)
# NOTE: Applied ONLY when CRS is projected (meters/feet). Skipped for geographic (degrees).
TARGET_PRECISION = 0.001  # set None to disable

# Merge behavior for boundary touches:
MERGE_WHEN_TOUCHES = False  

# Optional performance toggle: use vectorized sjoin to build candidate pairs
USE_SJOIN_FOR_EDGES = True

# Optional overlap threshold (only if CRS is projected); set None to disable
# Example: MIN_OVERLAP_RATIO = 0.01 means intersection area / min(area_i, area_j) must be ≥ 1%
MIN_OVERLAP_RATIO = None

# ---------------------------
# Helpers
# ---------------------------

def _list_first_layer(path: str) -> str:
    """Return the first layer name in a GPKG using pyogrio (preferred) or fiona."""
    try:
        import pyogrio
        layers = pyogrio.list_layers(path)
        if not layers:
            raise RuntimeError(f"No layers found in {path}")
        first = layers[0]
        return first["name"] if isinstance(first, dict) else first
    except Exception:
        import fiona
        layers = fiona.listlayers(path)
        if not layers:
            raise RuntimeError(f"No layers found in {path}")
        return layers[0]

def read_gpkg_first_layer(path: str) -> gpd.GeoDataFrame:
    """Read the first layer in a GeoPackage into a GeoDataFrame."""
    layer = _list_first_layer(path)
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        # Do NOT overwrite; fail loudly to avoid wrong assumptions
        raise ValueError(f"Layer '{layer}' in {os.path.basename(path)} has no CRS defined. "
                         "Please set CRS in the source file before running.")
    return gdf

def _is_geographic(gdf: gpd.GeoDataFrame) -> bool:
    try:
        return bool(gdf.crs and gdf.crs.is_geographic)
    except Exception:
        return False

def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid, drop empties, keep polygons, and apply precision ONLY if projected."""
    gdf = gdf.copy()

    # Drop null/empty
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # Make valid (Shapely 2.0: shapely.make_valid; fallback if needed)
    try:
        gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    except Exception:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)

    # Keep Polygon/MultiPolygon only
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # Optional precision reduction (ONLY when projected; never snap in degrees)
    if TARGET_PRECISION is not None and not _is_geographic(gdf):
        gdf["geometry"] = set_precision(gdf.geometry.values, grid_size=TARGET_PRECISION)
        # Re-validate after snapping (snapping can create tiny invalidities)
        try:
            gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
        except Exception:
            from shapely.validation import make_valid
            gdf["geometry"] = gdf.geometry.apply(make_valid)

    return gdf

def ensure_same_crs(g1: gpd.GeoDataFrame, g2: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Project g2 to g1's CRS if needed. Do NOT overwrite CRS if missing—raise earlier."""
    if g1.crs is None or g2.crs is None:
        # read_gpkg_first_layer already raises if None, so this is defensive
        raise ValueError("One of the inputs has no CRS. Ensure both have CRS set before processing.")
    if g1.crs != g2.crs:
        g2 = g2.to_crs(g1.crs)
    return g1, g2

def _apply_overlap_threshold_if_enabled(geom_a, geom_b, is_geographic: bool) -> bool:
    """Return True if overlap ratio meets threshold, or if threshold disabled."""
    if MIN_OVERLAP_RATIO is None or is_geographic:
        return True
    inter = geom_a.intersection(geom_b)
    if inter.is_empty:
        return False
    try:
        denom = min(geom_a.area, geom_b.area)
        if denom == 0:
            return False
        ratio = inter.area / denom
        return ratio >= MIN_OVERLAP_RATIO
    except Exception:
        return False

def build_overlap_edges(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    """
    Return (idx_a, idx_b) for features that overlap with area.
    Uses spatial indices; excludes pure boundary touches unless MERGE_WHEN_TOUCHES is True.
    Fallback method (loop-based).
    """
    edges: List[Tuple[int, int]] = []
    sidx_b = gdf_b.sindex
    is_geog = _is_geographic(gdf_a)  # assume same CRS after ensure_same_crs
    for i, geom_a in zip(gdf_a.index, gdf_a.geometry.values):
        # Broad-phase: any intersects
        cand_int = list(gdf_b.index[sidx_b.query(geom_a, predicate="intersects")])
        if not cand_int:
            continue
        cand_touch = set()
        if not MERGE_WHEN_TOUCHES:
            try:
                cand_touch = set(gdf_b.index[sidx_b.query(geom_a, predicate="touches")])
            except Exception:
                cand_touch = set()
        for j in cand_int:
            if not MERGE_WHEN_TOUCHES and j in cand_touch:
                continue
            geom_b = gdf_b.at[j, "geometry"]
            # Final check
            if MERGE_WHEN_TOUCHES:
                if geom_a.intersects(geom_b) and _apply_overlap_threshold_if_enabled(geom_a, geom_b, is_geog):
                    edges.append((i, j))
            else:
                if geom_a.intersects(geom_b) and not geom_a.touches(geom_b) and _apply_overlap_threshold_if_enabled(geom_a, geom_b, is_geog):
                    edges.append((i, j))
    return edges

def build_overlap_edges_sjoin(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame, intra_source: bool = False) -> List[Tuple[int, int]]:
    """
    Return (idx_a, idx_b) for features that overlap in area using vectorized sjoin,
    then filter out pure touches unless MERGE_WHEN_TOUCHES=True.
    Set intra_source=True for A vs A to dedupe and enforce i<j.
    """
    edges: List[Tuple[int, int]] = []
    is_geog = _is_geographic(gdf_a)
    pairs = gpd.sjoin(
        gdf_a[["geometry"]].copy(),
        gdf_b[["geometry"]].copy(),
        predicate="intersects",
        how="inner"
    )
    # sjoin returns left index as pairs.index and right in 'index_right'
    for idx_a, idx_b in zip(pairs.index.values, pairs["index_right"].values):
        if intra_source and idx_a == idx_b:
            continue  # skip self-pairs in intra-source joins
        # Enforce order for intra-source to avoid duplicates
        if intra_source and idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a
        geom_a = gdf_a.at[idx_a, "geometry"]
        geom_b = gdf_b.at[idx_b, "geometry"]
        if MERGE_WHEN_TOUCHES:
            if geom_a.intersects(geom_b) and _apply_overlap_threshold_if_enabled(geom_a, geom_b, is_geog):
                edges.append((idx_a, idx_b))
        else:
            if geom_a.intersects(geom_b) and not geom_a.touches(geom_b) and _apply_overlap_threshold_if_enabled(geom_a, geom_b, is_geog):
                edges.append((idx_a, idx_b))
    if intra_source:
        # Deduplicate pairs that may have arisen due to multi-part relationships
        edges = sorted(set(edges))
    return edges

class UnionFind:
    """Disjoint-set union with path compression and union by rank."""
    def __init__(self, nodes: List):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb: return
        ra, rb = self.rank[pa], self.rank[pb]
        if ra < rb:
            self.parent[pa] = pb
        elif ra > rb:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1

# ---------------------------
# Main routine
# ---------------------------

def build_union(osm_path: str, ms_path: str, output_gpkg: str, output_layer: str):
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)

    print("=== Building OSM + Microsoft union ===")
    print(f"OSM input:        {osm_path}")
    print(f"Microsoft input:  {ms_path}")
    print(f"Output:           {output_gpkg} (layer='{output_layer}')")
    print(f"Settings: MERGE_WHEN_TOUCHES={MERGE_WHEN_TOUCHES}, "
          f"USE_SJOIN_FOR_EDGES={USE_SJOIN_FOR_EDGES}, "
          f"TARGET_PRECISION={TARGET_PRECISION}, "
          f"MIN_OVERLAP_RATIO={MIN_OVERLAP_RATIO}")

    # 1) Read
    g_osm = read_gpkg_first_layer(osm_path)
    g_ms  = read_gpkg_first_layer(ms_path)

    print(f"Read OSM: {len(g_osm)} features | CRS={g_osm.crs.to_string()}")
    print(f"Read MS:  {len(g_ms)} features | CRS={g_ms.crs.to_string()}")

    # Tag source; preserve original row indices for traceability
    g_osm = g_osm.reset_index().rename(columns={"index": "orig_idx"}).assign(source="OSM")
    g_ms  = g_ms.reset_index().rename(columns={"index": "orig_idx"}).assign(source="Microsoft")

    # 2) Clean geometries
    before_osm = len(g_osm)
    before_ms  = len(g_ms)
    g_osm = fix_geometries(g_osm)
    g_ms  = fix_geometries(g_ms)
    print(f"Cleaned OSM: {len(g_osm)} polygons (dropped {before_osm - len(g_osm)})")
    print(f"Cleaned MS:  {len(g_ms)} polygons (dropped {before_ms  - len(g_ms)})")

    # 3) Align CRS (no overwrites; only project g_ms to g_osm if different)
    g_osm, g_ms = ensure_same_crs(g_osm, g_ms)

    # 4) Build overlap edges
    try:
        if USE_SJOIN_FOR_EDGES:
            edges_in_osm = build_overlap_edges_sjoin(g_osm, g_osm, intra_source=True)
            edges_in_ms  = build_overlap_edges_sjoin(g_ms,  g_ms,  intra_source=True)
            edges_cross  = build_overlap_edges_sjoin(g_osm, g_ms,  intra_source=False)
        else:
            edges_in_osm = [(i, j) for (i, j) in build_overlap_edges(g_osm, g_osm) if i < j]
            edges_in_ms  = [(i, j) for (i, j) in build_overlap_edges(g_ms,  g_ms)  if i < j]
            edges_cross  = build_overlap_edges(g_osm, g_ms)
    except Exception as e:
        print(f"Vectorized sjoin failed ({e}); falling back to loop-based edge builder.")
        edges_in_osm = [(i, j) for (i, j) in build_overlap_edges(g_osm, g_osm) if i < j]
        edges_in_ms  = [(i, j) for (i, j) in build_overlap_edges(g_ms,  g_ms)  if i < j]
        edges_cross  = build_overlap_edges(g_osm, g_ms)

    print(f"Edges within OSM: {len(edges_in_osm)}")
    print(f"Edges within MS:  {len(edges_in_ms)}")
    print(f"Edges across OSM↔MS: {len(edges_cross)}")

    # 5) Union-find across all features
    nodes_osm = [("OSM", i) for i in g_osm.index]
    nodes_ms  = [("Microsoft", j) for j in g_ms.index]
    uf = UnionFind(nodes_osm + nodes_ms)

    for i, j in edges_in_osm:
        uf.union(("OSM", i), ("OSM", j))
    for i, j in edges_in_ms:
        uf.union(("Microsoft", i), ("Microsoft", j))
    for i, j in edges_cross:
        uf.union(("OSM", i), ("Microsoft", j))

    # 6) Collect components
    components: Dict[Tuple[str, int], Dict[str, List[int]]] = {}
    for key in nodes_osm + nodes_ms:
        root = uf.find(key)
        components.setdefault(root, {"OSM": [], "Microsoft": []})
        src, idx = key
        components[root][src].append(idx)

    print(f"Connected components: {len(components)}")

    # 7) Union geometry per component & tag source
    records = []
    is_geog = _is_geographic(g_osm)  # CRS aligned already
    for root, parts in components.items():
        osm_ix = parts["OSM"]
        ms_ix  = parts["Microsoft"]

        geoms = []
        if osm_ix:
            geoms.extend(g_osm.loc[osm_ix, "geometry"].tolist())
        if ms_ix:
            geoms.extend(g_ms.loc[ms_ix, "geometry"].tolist())
        if not geoms:
            continue

        # Shapely 2.0: union_all is fast; fallback to unary_union if needed
        try:
            union_geom = shapely.union_all(geoms)
        except Exception:
            from shapely.ops import unary_union
            union_geom = unary_union(geoms)

        if union_geom.is_empty:
            continue

        source_tag = (
            "OSM+Microsoft" if (osm_ix and ms_ix)
            else "OSM" if osm_ix
            else "Microsoft"
        )

        osm_ids = g_osm.loc[osm_ix, "orig_idx"].tolist() if osm_ix else []
        ms_ids  = g_ms.loc[ms_ix, "orig_idx"].tolist()  if ms_ix else []

        rec = {
            "source": source_tag,
            "n_osm_parts": len(osm_ix),
            "n_ms_parts": len(ms_ix),
            "osm_ids": ";".join(map(str, osm_ids)) if osm_ids else None,
            "ms_ids":  ";".join(map(str, ms_ids))  if ms_ids else None,
            "geometry": union_geom
        }
        # Optional area (only if projected)
        if not is_geog:
            rec["area"] = union_geom.area  # units = CRS units (meters/feet)
        records.append(rec)

    g_out = gpd.GeoDataFrame(records, geometry="geometry", crs=g_osm.crs)

    # 8) (Optional) explode multiparts & add stable component_id
    g_out = g_out.explode(index_parts=False).reset_index(drop=True)
    g_out["component_id"] = g_out.index.astype("int64")

    print(f"Output features (post-explode): {len(g_out)}")

    # 9) Export (prefer pyogrio if available; else geopandas.to_file)
    try:
        import pyogrio
        pyogrio.write_dataframe(g_out, output_gpkg, layer=output_layer, driver="GPKG")
    except Exception as e:
        print(f"pyogrio write failed ({e}); using GeoPandas/Fiona.")
        g_out.to_file(output_gpkg, layer=output_layer, driver="GPKG")

    print(f"✅ Wrote {len(g_out)} features to:\n  {output_gpkg} (layer='{output_layer}')")
    print("Columns:", list(g_out.columns))

def _paths_from_county(county: str) -> Tuple[str, str, str, str]:
    """Build standard paths and layer names from the county token."""
    county_token = county.strip().lower().replace(" ", "_")
    osm_path     = os.path.join(OSM_DIR, f"{county_token}_buildings.gpkg")
    ms_path      = os.path.join(MS_DIR,  f"{county_token}_building_footprints.gpkg")
    output_gpkg  = os.path.join(OUT_DIR, f"{county_token}_buildings_union.gpkg")
    output_layer = f"{county_token}_buildings_union"
    return osm_path, ms_path, output_gpkg, output_layer




## Main function
for COUNTY in COUNTY_LIST:
    print(f'Processing county: {COUNTY}')
    osm_path, ms_path, output_gpkg, output_layer = _paths_from_county(COUNTY)
    build_union(osm_path, ms_path, output_gpkg, output_layer)
