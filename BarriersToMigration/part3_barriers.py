# barriers_to_migration/part3_barriers.py
import logging
import os

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, substring

from config import BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import (
    _decompose_to_points,
    _dedup_geoms_by_wkb,
    _detect_origin_barrier_hit,
    build_asset_tree,
    build_strtree_from_lines,
    first_intersection_on_transect,
    merge_shore_paths,
    query_tree_geoms,
    safe_write_gpkg,
    to_line_boundaries,
)

logger = logging.getLogger("barriers_to_migration")


def _transect_origin_and_landward(row) -> tuple[Point, LineString]:
    """Transects run seaward -> origin -> landward; barrier search uses the landward half."""
    coords = list(row.geometry.coords)
    ox = row.get("origin_x")
    oy = row.get("origin_y")
    if ox is None or oy is None or np.isnan(ox) or np.isnan(oy):
        ox, oy = coords[len(coords) // 2][:2]
    ox, oy = float(ox), float(oy)
    return Point(ox, oy), LineString([(ox, oy), coords[-1][:2]])


def _build_behind_barrier_trees(
    county_layers: dict[str, gpd.GeoDataFrame], config: BarrierConfig
) -> dict[str, STRtree]:
    """One STRtree per "behind barrier" layer using full geometries (polygons stay filled)."""
    trees: dict[str, STRtree] = {}
    for name in config.behind_barrier_layers:
        gdf = county_layers.get(name)
        if gdf is None or gdf.empty:
            continue
        geoms = [
            g
            for g in gdf.to_crs(config.target_crs).geometry
            if g is not None and not g.is_empty
        ]
        if geoms:
            trees[name] = STRtree(geoms)
    return trees


def _feature_behind_barrier(
    tran: LineString,
    hit_dist_m: float,
    trees: dict[str, STRtree],
    search_m: float,
    eps_m: float = 0.01,
) -> str | None:
    """Return the layer name of the nearest feature found along `tran` within
    (hit_dist_m, hit_dist_m + search_m], or None.

    Features whose contact with the search segment starts at the hit point
    itself (i.e. the barrier that was hit) are ignored.
    """
    if not trees or search_m <= 0:
        return None
    end_m = min(hit_dist_m + search_m, tran.length)
    if end_m - hit_dist_m <= eps_m:
        return None
    seg = substring(tran, hit_dist_m, end_m)

    best_name, best_d = None, None
    for name, tree in trees.items():
        for g in query_tree_geoms(tree, seg):
            inter = seg.intersection(g)
            pts = _decompose_to_points(inter)
            if not pts:
                continue
            d = min(seg.project(p) for p in pts)
            if d <= eps_m:
                continue
            if best_d is None or d < best_d:
                best_name, best_d = name, d
    return best_name


def find_barriers_and_export(
    county_layers: dict[str, gpd.GeoDataFrame],
    transect_layers: dict[str, gpd.GeoDataFrame],
    config: BarrierConfig,
    out_dir: str,
    county_name: str,
    asset_layers: dict[str, LayerSpec | str | dict] | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Part 3:
    - For each transect, find first intersection with any barrier layer supplied
      via `asset_layers` (line layers directly, polygon layers via their exteriors).
    - Emit:
        * barrier_hits (POINTS); `behind_barrier` names the nearest road/rail/
          building layer found within `config.behind_barrier_search_m` landward
          of the hit along the transect (None if nothing found)
        * shoreline_barrier_distance_lines (short segments along shoreline between bases, with mean hit dists)
        * transects_hits (origin->hit lines)
        * transects_all (re-export for convenience / QA)
    - Export everything as ONE GeoPackage (e.g., `<out_dir>/barriers_hard.gpkg`)
    """
    shoreline_gdf = county_layers.get("shoreline")

    RESERVED = {"shoreline", "county_boundary", "barrier_region", "water"}
    if asset_layers is None:
        specs = {
            name: LayerSpec(path="") for name in county_layers if name not in RESERVED
        }
    else:
        specs = normalize_asset_specs(asset_layers)

    transects_all = transect_layers["transects_all"]
    transect_bases = transect_layers["transect_bases"]

    # One STRtree per barrier layer, keyed by layer name (used as asset_type)
    barrier_trees: dict[str, object] = {}
    src_map: dict = {}
    for name, spec in specs.items():
        if not spec.barrier:
            continue
        tree, idmap = build_asset_tree(
            county_layers.get(name), config.target_crs, name, spec.kind
        )
        if tree is None:
            logger.info("Barrier layer '%s' is empty; skipping.", name)
            continue
        barrier_trees[name] = tree
        src_map.update(idmap)

    logger.info(
        "%s: using %d barrier layer(s): %s",
        county_name,
        len(barrier_trees),
        ", ".join(barrier_trees) or "none",
    )

    # Match the exact boundary representation used to generate the transect
    # origins in Part 2. The source shoreline may contain polygons or multipart
    # geometries, which must not be indexed as filled shoreline areas here.
    shoreline_paths = merge_shore_paths(
        to_line_boundaries(shoreline_gdf, config.target_crs)
    )
    shoreline_paths_gdf = gpd.GeoDataFrame(
        geometry=shoreline_paths, crs=config.target_crs
    )
    tree_shoreline, shoreline_map = build_strtree_from_lines(
        shoreline_paths_gdf, config.target_crs, "shoreline"
    )

    behind_trees = _build_behind_barrier_trees(county_layers, config)
    logger.info(
        "%s: behind-barrier search (%.1f m) using layer(s): %s",
        county_name,
        config.behind_barrier_search_m,
        ", ".join(behind_trees) or "none",
    )

    # WKB keyed lookup to recover (asset_type, asset_id) for explicit origin hits
    src_wkb_lookup = {}
    for tree in barrier_trees.values():
        for g in tree.geometries:
            meta = src_map.get(id(g))
            if meta is not None:
                src_wkb_lookup[g.wkb] = meta

    barrier_hits_records = []
    tran_hits_records = []
    crosses_shoreline_flags: dict = {}
    barrier_hit_flags: dict = {}

    # For each transect, combine nearby candidate geometries (via spatial index) and pick first intersection
    for idx, row in transects_all.iterrows():
        origin, tran = _transect_origin_and_landward(row)
        x0, y0 = origin.x, origin.y
        crosses_shoreline_flags[idx] = False
        barrier_hit_flags[idx] = False

        # Collect candidates with labels
        tagged_cand = []
        for tag, tree in barrier_trees.items():
            for g in query_tree_geoms(tree, tran, buffer_tol=config.buffer_distance_m):
                tagged_cand.append((g, tag))

        # Dedup by WKB
        tagged_unique = _dedup_geoms_by_wkb(tagged_cand)

        # 1) Explicit origin-hit detection (touch or overlap at shoreline)
        origin_hit = _detect_origin_barrier_hit(
            origin,
            tagged_unique,
            tol_m=float(getattr(config, "origin_hit_tolerance_m", 0.5)),
        )

        hit_pt = src_label = dist_m = src_idx = None

        if origin_hit:
            hit_pt = origin_hit["geometry"]
            src_label = origin_hit["asset_type"]
            dist_m = 0.0
            # Try to recover an asset_id via WKB lookup
            key_wkb = origin_hit["hit_geom"].wkb
            meta = src_wkb_lookup.get(key_wkb)
            # 'meta' could be (asset_type, asset_id) or just asset_id; adapt if needed
            src_idx = (
                meta[1]
                if (isinstance(meta, (tuple, list)) and len(meta) >= 2)
                else meta
            )

        else:
            # 2) Fall back to "first intersection along transect"
            #    Set origin_eps to 0.0 so true origin hits aren't discarded
            #    (you can keep 0.5 here if you want only the explicit origin logic to tag 0-m hits)
            unique_geoms_only = [g for (g, _) in tagged_unique]
            hit_pt, src_label, dist_m, src_idx = first_intersection_on_transect(
                tran,
                unique_geoms_only,
                src_map,
                snap_tol=config.snap_tolerance_m,
                origin_eps=0.0,
            )

        # --- Shoreline-first crossing check ---
        if tree_shoreline is not None:
            # Query nearby shoreline geometries
            shore_cand = query_tree_geoms(
                tree_shoreline, tran, buffer_tol=config.buffer_distance_m
            )

            shore_tagged_unique = _dedup_geoms_by_wkb(
                [(g, "shoreline") for g in shore_cand]
            )

            # Find the first true shoreline intersection on the landward half,
            # skipping the expected contact at the transect origin. Unlike barrier
            # detection, proximity within snap_tolerance_m is not a crossing.
            shore_dist_m = None
            if shore_tagged_unique:
                _, _, shore_dist_m, _ = first_intersection_on_transect(
                    tran,
                    [g for (g, _) in shore_tagged_unique],
                    shoreline_map,
                    snap_tol=0.0,
                    origin_eps=float(getattr(config, "origin_hit_tolerance_m", 0.5)),
                )

            # If shoreline is intersected before any barrier, mark this transect as "no hit"
            EPS = 1e-9
            if (
                (shore_dist_m is not None)
                and not (isinstance(shore_dist_m, float) and np.isnan(shore_dist_m))
                and ((dist_m is None) or (shore_dist_m + EPS < dist_m))
            ):
                logger.debug(
                    f"{county_name}: path_id={row['path_id']}, sample_id={row['sample_id']} "
                    f"crosses shoreline first at {shore_dist_m:.3f} m; barrier hit suppressed."
                )
                crosses_shoreline_flags[idx] = True
                continue

        if hit_pt is not None and not hit_pt.is_empty:
            # Normalize hit_pt to a 2D Point (handles MultiPoint, LineString, GeometryCollection)
            if isinstance(hit_pt, Point):
                hx, hy = hit_pt.coords[0][:2]
                hit_point_geom = Point(hx, hy)
            elif hit_pt.geom_type == "MultiPoint" and len(hit_pt.geoms) > 0:
                hx, hy = hit_pt.geoms[0].coords[0][:2]
                hit_point_geom = Point(hx, hy)
            else:
                _, np_hit = nearest_points(origin, hit_pt)
                hx, hy = np_hit.coords[0][:2]
                hit_point_geom = Point(hx, hy)

            barrier_hit_flags[idx] = True

            behind = _feature_behind_barrier(
                tran,
                float(dist_m),
                behind_trees,
                config.behind_barrier_search_m,
            )

            barrier_hits_records.append(
                {
                    "path_id": row["path_id"],
                    "sample_id": row["sample_id"],
                    "asset_type": src_label,
                    "asset_id": src_idx,
                    "dist_m": float(dist_m),
                    "behind_barrier": behind,
                    "geometry": hit_point_geom,
                }
            )

            # Use coordinate tuples to avoid dimensionality/mixed-geometry issues
            tran_hits_records.append(
                {
                    "path_id": row["path_id"],
                    "sample_id": row["sample_id"],
                    "asset_type": src_label,
                    "asset_id": src_idx,
                    "geometry": LineString([(x0, y0), (hx, hy)]),
                }
            )

    hit_columns = [
        "path_id",
        "sample_id",
        "asset_type",
        "asset_id",
        "dist_m",
        "behind_barrier",
        "geometry",
    ]
    transect_hit_columns = [
        "path_id",
        "sample_id",
        "asset_type",
        "asset_id",
        "geometry",
    ]
    barrier_hits = gpd.GeoDataFrame(
        barrier_hits_records,
        columns=hit_columns,
        geometry="geometry",
        crs=config.target_crs,
    )
    transects_hits = gpd.GeoDataFrame(
        tran_hits_records,
        columns=transect_hit_columns,
        geometry="geometry",
        crs=config.target_crs,
    )

    transects_all = transects_all.copy()
    transects_all["crosses_shoreline"] = [
        bool(crosses_shoreline_flags.get(i, False)) for i in transects_all.index
    ]
    transects_all["has_barrier_hit"] = [
        bool(barrier_hit_flags.get(i, False)) for i in transects_all.index
    ]

    # -----------------------------
    # Shoreline lines: exactly one segment per transect sample, centred on its
    # base point and spanning the midpoints to its neighbours.
    # -----------------------------
    # Map: path_id -> {sample_id: dist_m} (min if duplicates in barrier_hits)
    hits_by_path: dict = {}
    if not barrier_hits.empty:
        for pid, group in barrier_hits.groupby("path_id"):
            hits_by_path[pid] = group.groupby("sample_id")["dist_m"].min().to_dict()

    line_records = []
    bases_sorted = transect_bases.drop_duplicates(
        subset=["path_id", "sample_id"]
    ).sort_values(["path_id", "measure_m"])

    for pid, group in bases_sorted.groupby("path_id"):
        if len(group) < 2:
            logger.warning(
                f"{county_name}: path_id={pid} has <2 base points; skipping shoreline line."
            )
            continue

        coords = np.column_stack(
            (group.geometry.x.to_numpy(), group.geometry.y.to_numpy())
        )
        measures = group["measure_m"].to_numpy(dtype=float)
        sample_ids = group["sample_id"].to_numpy()
        dist_map = hits_by_path.get(pid, {})

        mid_xy = 0.5 * (coords[:-1] + coords[1:])
        mid_m = 0.5 * (measures[:-1] + measures[1:])
        n = len(coords)

        for i in range(n):
            start_xy = coords[0] if i == 0 else mid_xy[i - 1]
            end_xy = coords[-1] if i == n - 1 else mid_xy[i]
            start_m = measures[0] if i == 0 else mid_m[i - 1]
            end_m = measures[-1] if i == n - 1 else mid_m[i]

            verts = [tuple(start_xy), tuple(coords[i]), tuple(end_xy)]
            verts = [v for j, v in enumerate(verts) if j == 0 or v != verts[j - 1]]
            if len(verts) < 2:
                continue
            seg_geom = LineString(verts)
            if seg_geom.length == 0:
                continue

            seg_dist = dist_map.get(sample_ids[i], np.nan)

            line_records.append(
                {
                    "path_id": pid,
                    "segment_index": i,  # order along shoreline
                    "start_m": float(start_m),
                    "end_m": float(end_m),
                    "sample_id": sample_ids[i],
                    "dist_m": float(seg_dist),
                    "geometry": seg_geom,
                }
            )

    shoreline_line_columns = [
        "path_id",
        "segment_index",
        "start_m",
        "end_m",
        "sample_id",
        "dist_m",
        "geometry",
    ]
    shoreline_barrier_distance_lines = gpd.GeoDataFrame(
        line_records,
        columns=shoreline_line_columns,
        geometry="geometry",
        crs=config.target_crs,
    )

    # Export single GPKG for QA/QC
    out_gpkg = os.path.join(out_dir, "barriers_hard.gpkg")
    safe_write_gpkg(barrier_hits, out_gpkg, "barrier_hits", mode="w")
    safe_write_gpkg(
        shoreline_barrier_distance_lines,
        out_gpkg,
        "shoreline_barrier_distance_lines",
        mode="a",
    )
    safe_write_gpkg(transects_hits, out_gpkg, "transects_hits", mode="a")
    # Include transects_all for completeness
    safe_write_gpkg(transects_all, out_gpkg, "transects_all", mode="a")

    logger.info(f"Part 3 complete: {out_gpkg}")

    return {
        "barrier_hits": barrier_hits,
        "shoreline_barrier_distance_lines": shoreline_barrier_distance_lines,
        "transects_hits": transects_hits,
        "transects_all": transects_all,
    }
