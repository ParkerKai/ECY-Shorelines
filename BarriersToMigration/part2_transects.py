# barriers_to_migration/part2_transects.py
import logging
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from config import BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import (
    build_asset_tree,
    densify_line,
    force_2d_geom,
    left_normal,
    line_point,
    merge_shore_paths,
    open_warped_vrt,
    query_tree_geoms,
    safe_write_gpkg,
    sample_vrt_points,
    smooth_tangents_for_samples,
    to_line_boundaries,
)

logger = logging.getLogger("barriers_to_migration")

RESERVED_LAYERS = {"shoreline", "county_boundary", "barrier_region", "water"}


def determine_landward_side_from_water(
    path: LineString,
    measures: list[float],
    tangents_sm: list[np.ndarray],
    config: BarrierConfig,
    water_tree: STRtree | None,
) -> tuple[int, str] | None:
    """
    Decide landward side (+1 LEFT, -1 RIGHT) from a visible surface water polygon.

    Probe points are offset to both sides at `config.water_probe_dists_m`; the side
    with more probes inside water is seaward, so landward is the opposite side.
    Returns None when the layer is missing or the vote margin is below
    `config.water_margin_min`, so the caller can fall back to DEM/assets.
    """
    if water_tree is None or len(measures) == 0:
        return None

    stride = max(1, int(config.side_probe_stride_m / max(1.0, config.spacing_m)))
    stride = min(stride, max(1, len(measures) - 1))
    dists = tuple(config.water_probe_dists_m)

    probes: list[Point] = []
    for i in range(0, len(measures), stride):
        base = line_point(path, measures[i])
        n_left = left_normal(tangents_sm[i])
        for side_sign in (+1.0, -1.0):
            for d in dists:
                probes.append(
                    Point(
                        base.x + side_sign * n_left[0] * d,
                        base.y + side_sign * n_left[1] * d,
                    )
                )
    if not probes:
        return None

    hit = np.zeros(len(probes), dtype=bool)
    found = water_tree.query(np.array(probes, dtype=object), predicate="intersects")
    if found.size:
        hit[found[0]] = True

    # Rows are sampled locations; columns are [left distances..., right distances...]
    hit = hit.reshape(-1, 2 * len(dists))
    left_water = hit[:, : len(dists)].sum(axis=1)
    right_water = hit[:, len(dists) :].sum(axis=1)

    # More water on the left means left is seaward, so landward is the right side.
    votes_landward_right = int(np.sum(left_water > right_water))
    votes_landward_left = int(np.sum(right_water > left_water))
    total = votes_landward_left + votes_landward_right
    if total == 0:
        logger.info("Water check inconclusive: no probes distinguished the two sides.")
        return None

    margin = abs(votes_landward_left - votes_landward_right) / float(total)
    logger.info(
        "Water votes landward L/R=%d/%d (margin=%.2f) over %d sampled locations.",
        votes_landward_left,
        votes_landward_right,
        margin,
        hit.shape[0],
    )
    if margin < float(config.water_margin_min):
        logger.info(
            "Water vote margin below %.2f; falling back to DEM/assets.",
            float(config.water_margin_min),
        )
        return None

    sign = +1 if votes_landward_left >= votes_landward_right else -1
    label = (
        "landward_left_water_polygon" if sign == +1 else "landward_right_water_polygon"
    )
    return sign, label


def determine_landward_side_for_path(
    path: LineString,
    measures: list[float],
    tangents_sm: list[np.ndarray],
    config: BarrierConfig,
    dem_vrt,
    dem_nodata: float | None,
    asset_trees: list[tuple[object, float]],
) -> tuple[int, str]:
    """
    Decide landward side (+1 LEFT, -1 RIGHT) for the entire merged path.

    RESET HEURISTIC (elevation-first):
      1) Elevation primary: sample inland on left/right at configurable distances.
         Compare median elevation; majority vote across samples decides landward.
      2) Assets secondary (confirm/tie-break):
         Distance-weighted, de-duplicated counts of nearby features from
         `asset_trees`, a list of (STRtree, weight) pairs.
         Used when elevation margin is weak or DEM is missing.
      3) Island consistency check:
         Path-level elevation difference sanity-check; flip if global elevation contradicts local votes by a meaningful margin.
      4) Default: LEFT if tie or insufficient signal.

    Returns:
        sign: +1 for LEFT, -1 for RIGHT
        label: decision rationale label
    """
    if len(measures) == 0:
        return +1, "path_landward_fixed_left_default"

    # Configurable thresholds/distances with sensible defaults
    elev_probe_dists_m: tuple[int, ...] = tuple(
        getattr(config, "elev_probe_dists_m", (10, 30, 60))
    )
    elev_margin_min: float = float(
        getattr(config, "elev_margin_min", 0.20)
    )  # fraction of elevation votes difference
    asset_margin_min: float = float(
        getattr(config, "asset_margin_min", 0.20)
    )  # fraction of asset votes difference
    asset_score_delta_min: float = float(
        getattr(config, "asset_score_delta_min", 0.10)
    )  # per-sample score delta
    elev_consistency_dists_m: tuple[int, ...] = tuple(
        getattr(config, "elev_consistency_dists_m", (30, 60, 120))
    )
    elev_consistency_delta_m: float = float(
        getattr(config, "elev_consistency_delta_m", 0.25)
    )  # median diff threshold (m)

    # Probe along the path at stride (skip dense measures for speed)
    stride = max(1, int(config.side_probe_stride_m / max(1.0, config.spacing_m)))
    stride = min(stride, max(1, len(measures) - 1))

    # Collect per-sample metrics
    elev_votes_left = 0
    elev_votes_right = 0
    elev_diffs: list[
        float
    ] = []  # per-sample (median(L) - median(R)), for consistency check

    asset_left_hits = 0
    asset_right_hits = 0
    asset_diffs: list[float] = []  # per-sample (score(L) - score(R))

    def score_assets(seg_line: LineString) -> float:
        """
        Distance-weighted and de-duplicated asset scoring.
        Sources weighted by indicative strength of landward side.
        """
        score = 0.0
        for tree, w in asset_trees:
            if tree is None:
                continue
            nearby = query_tree_geoms(
                tree, seg_line, buffer_tol=config.side_assets_buffer_m
            )
            seen = set()
            for g in nearby:
                # Robust de-duplication key; fallback to id if needed
                try:
                    key = g.wkb
                except AttributeError:
                    key = id(g)
                if key in seen:
                    continue
                seen.add(key)
                d = max(seg_line.distance(g), 1.0)  # avoid division by zero
                score += w * (1.0 / d)
        return score

    # Loop over sampled indices
    for i in range(0, len(measures), stride):
        m = measures[i]
        t = tangents_sm[i]
        n_left = left_normal(t)
        n_right = -n_left

        base = line_point(path, m)

        # Short test segments for asset queries
        def seg(nvec, base=base):
            end = Point(
                base.x + nvec[0] * config.dem_test_len_m,
                base.y + nvec[1] * config.dem_test_len_m,
            )
            return LineString([base, end])

        seg_left = seg(n_left)
        seg_right = seg(n_right)

        # Elevation sampling (primary)
        zL_med = np.nan
        zR_med = np.nan
        if dem_vrt is not None:
            ptsL = [
                Point(base.x + n_left[0] * d, base.y + n_left[1] * d)
                for d in elev_probe_dists_m
            ]
            ptsR = [
                Point(base.x + n_right[0] * d, base.y + n_right[1] * d)
                for d in elev_probe_dists_m
            ]
            zL = np.array(sample_vrt_points(dem_vrt, ptsL, dem_nodata))
            zR = np.array(sample_vrt_points(dem_vrt, ptsR, dem_nodata))
            zL = zL[~np.isnan(zL)]
            zR = zR[~np.isnan(zR)]
            if zL.size and zR.size:
                zL_med = float(np.nanmedian(zL))
                zR_med = float(np.nanmedian(zR))
                elev_diffs.append(zL_med - zR_med)
                if zL_med > zR_med:
                    elev_votes_left += 1
                elif zR_med > zL_med:
                    elev_votes_right += 1
                # equal -> no vote

        # Asset scoring (secondary)
        sL = score_assets(seg_left)
        sR = score_assets(seg_right)
        asset_diffs.append(sL - sR)
        if (sL - sR) > asset_score_delta_min:
            asset_left_hits += 1
        elif (sR - sL) > asset_score_delta_min:
            asset_right_hits += 1
        # small delta -> no asset vote

    # Elevation-first decision
    elev_votes_total = elev_votes_left + elev_votes_right
    if elev_votes_total > 0:
        sign_elev = +1 if elev_votes_left >= elev_votes_right else -1
        elev_margin = abs(elev_votes_left - elev_votes_right) / float(elev_votes_total)
    else:
        sign_elev = +1  # default LEFT
        elev_margin = 0.0

    # Asset summary (for confirmation / tie-break)
    asset_votes_total = asset_left_hits + asset_right_hits
    assets_sign = +1 if asset_left_hits >= asset_right_hits else -1
    assets_margin = (
        (abs(asset_left_hits - asset_right_hits) / float(asset_votes_total))
        if asset_votes_total > 0
        else 0.0
    )

    # Initial decision from elevation
    sign = sign_elev
    decision = (
        "landward_left_elevation_primary"
        if sign == +1
        else "landward_right_elevation_primary"
    )

    # Use assets to confirm or flip only if elevation is weak/inconclusive
    if elev_margin < elev_margin_min or elev_votes_total == 0:
        if assets_margin >= asset_margin_min and assets_sign != sign_elev:
            sign = assets_sign
            decision = (
                "landward_left_assets_confirm_flip"
                if sign == +1
                else "landward_right_assets_confirm_flip"
            )
        else:
            # Keep elevation decision but note assets were considered
            decision += "_assets_considered"

    # Island consistency check (path-level elevation sanity)
    # Aggregate median difference across samples at longer distances.
    if dem_vrt is not None and len(measures) > 0:
        # Re-sample at a coarser set to avoid recomputing per-point tangents
        stride_cons = max(1, int(3 * stride))
        diffs_consistency: list[float] = []
        for i in range(0, len(measures), stride_cons):
            m = measures[i]
            t = tangents_sm[i]
            n_left = left_normal(t)
            n_right = -n_left
            base = line_point(path, m)
            ptsL = [
                Point(base.x + n_left[0] * d, base.y + n_left[1] * d)
                for d in elev_consistency_dists_m
            ]
            ptsR = [
                Point(base.x + n_right[0] * d, base.y + n_right[1] * d)
                for d in elev_consistency_dists_m
            ]
            zL = np.array(sample_vrt_points(dem_vrt, ptsL, dem_nodata))
            zR = np.array(sample_vrt_points(dem_vrt, ptsR, dem_nodata))
            zL = zL[~np.isnan(zL)]
            zR = zR[~np.isnan(zR)]
            if zL.size and zR.size:
                diffs_consistency.append(float(np.nanmedian(zL) - np.nanmedian(zR)))

        if diffs_consistency:
            path_median_diff = float(
                np.nanmedian(diffs_consistency)
            )  # + => LEFT higher
            # If global elevation strongly contradicts the chosen sign, flip
            if (sign == +1 and path_median_diff < -elev_consistency_delta_m) or (
                sign == -1 and path_median_diff > elev_consistency_delta_m
            ):
                sign = -sign
                decision = (
                    "landward_right_elevation_consistency_flip"
                    if sign == -1
                    else "landward_left_elevation_consistency_flip"
                )

    # Final log detail
    logger.info(
        "Path decision: elev_votes L/R=%d/%d (margin=%.2f), asset_votes L/R=%d/%d (margin=%.2f) -> %s",
        elev_votes_left,
        elev_votes_right,
        elev_margin,
        asset_left_hits,
        asset_right_hits,
        assets_margin,
        decision,
    )

    return sign, decision


def generate_transects(
    county_layers: dict[str, gpd.GeoDataFrame],
    config: BarrierConfig,
    dem_path: str | None,
    out_dir: str,
    county_name: str,
    asset_layers: dict[str, LayerSpec | str | dict] | None = None,
    dem_source_crs: str | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Part 2:
    - Convert shoreline polygons/lines to exterior boundaries
    - Merge into continuous paths
    - Smooth orientation and determine landward side per path, preferring the
      visible surface water polygon (`county_layers["water"]`) and falling back
      to the elevation/asset heuristic when water is missing or inconclusive
    - Emit 'transects_all' (full-length) and 'transect_bases' (origin points)

    `asset_layers` is the same mapping passed to Part 1; layers with
    `landward=True` contribute to the landward-side vote using their
    `landward_weight`. If omitted, every non-reserved layer in `county_layers`
    is used with unit weight.
    """
    shoreline_gdf = county_layers["shoreline"]

    if asset_layers is None:
        specs = {
            name: LayerSpec(path="")
            for name in county_layers
            if name not in RESERVED_LAYERS
        }
    else:
        specs = normalize_asset_specs(asset_layers)

    # STRtrees for assets used in landward-side voting
    asset_trees: list[tuple[object, float]] = []
    for name, spec in specs.items():
        if not spec.landward:
            continue
        gdf = county_layers.get(name)
        tree, _ = build_asset_tree(gdf, config.target_crs, name, spec.kind)
        if tree is not None:
            asset_trees.append((tree, float(spec.landward_weight)))
    logger.info(
        "Landward voting uses %d asset layer(s): %s",
        len(asset_trees),
        ", ".join(n for n, s in specs.items() if s.landward),
    )

    # Visible surface water polygons (primary landward/seaward signal)
    water_tree: STRtree | None = None
    water_gdf = county_layers.get("water")
    if water_gdf is not None and not water_gdf.empty:
        water_geoms = [
            g
            for g in water_gdf.to_crs(config.target_crs).geometry
            if g is not None and not g.is_empty
        ]
        if water_geoms:
            water_tree = STRtree(water_geoms)
            logger.info("Water polygon available: %d feature(s).", len(water_geoms))
    if water_tree is None:
        logger.warning(
            "No visible surface water layer available; using DEM/asset heuristic only."
        )

    # DEM (warped to target CRS) — guarded
    src = dem_vrt = None
    dem_nodata = None
    if dem_path:
        src, dem_vrt, dem_nodata = open_warped_vrt(
            dem_path, config.target_crs, dem_source_crs
        )

    # Shoreline boundaries & merging
    shore_lines = to_line_boundaries(shoreline_gdf, config.target_crs)

    # Ensure 2D geometries regardless of the container type returned
    if isinstance(shore_lines, gpd.GeoDataFrame):
        geom2d = shore_lines.geometry.apply(force_2d_geom)
        shore_lines = shore_lines.set_geometry(geom2d, crs=shore_lines.crs)
    elif isinstance(shore_lines, gpd.GeoSeries):
        shore_lines = shore_lines.apply(force_2d_geom)
    elif isinstance(shore_lines, list):
        shore_lines = [
            force_2d_geom(g)
            for g in shore_lines
            if g is not None and not getattr(g, "is_empty", False)
        ]
    else:
        raise TypeError(
            f"Unexpected type from to_line_boundaries: {type(shore_lines)}; "
            "expected GeoDataFrame, GeoSeries, or list of geometries."
        )

    # Merge into continuous paths (works for GeoSeries and list)
    shore_paths = merge_shore_paths(shore_lines)

    # Ensure paths are 2D (defensive)
    shore_paths = [
        force_2d_geom(p)
        for p in shore_paths
        if p is not None and not getattr(p, "is_empty", False)
    ]

    transects_records = []
    bases_records = []

    for path_idx, path in enumerate(shore_paths):
        measures = densify_line(path, config.spacing_m)
        if not measures:
            logger.warning(
                "Path %d: no measures produced by densify_line; skipping.", path_idx
            )
            continue

        tang_sm = smooth_tangents_for_samples(
            path,
            measures,
            eps_for_tangent_m=config.eps_for_tangent_m,
            smooth_m=config.orientation_smooth_m,
            max_turn_deg=config.max_turn_deg,
        )

        logger.info(
            "Path %d: %d samples (spacing=%.2f m); starting landward determination.",
            path_idx,
            len(measures),
            config.spacing_m,
        )

        water_result = determine_landward_side_from_water(
            path, measures, tang_sm, config, water_tree
        )
        if water_result is not None:
            sign, decision = water_result
        else:
            sign, decision = determine_landward_side_for_path(
                path,
                measures,
                tang_sm,
                config,
                dem_vrt,
                dem_nodata,
                asset_trees,
            )

        logger.info(
            "Path %d: landward side is %s (%s).",
            path_idx,
            "LEFT" if sign == +1 else "RIGHT",
            decision,
        )

        # Build transects
        for i, m in enumerate(measures):
            origin = line_point(path, m)
            normal = left_normal(tang_sm[i]) * sign
            landward_end = Point(
                origin.x + normal[0] * config.max_search_m,
                origin.y + normal[1] * config.max_search_m,
            )
            seaward_end = Point(
                origin.x - normal[0] * config.max_search_m,
                origin.y - normal[1] * config.max_search_m,
            )
            # Vertex order is always seaward -> origin -> landward
            tran = LineString([seaward_end, origin, landward_end])
            landward_azimuth_deg = float(
                np.degrees(np.arctan2(normal[0], normal[1])) % 360.0
            )

            transects_records.append(
                {
                    "path_id": path_idx,
                    "sample_id": i,
                    "measure_m": m,
                    "landward_decision": decision,
                    "landward_side": "left" if sign == +1 else "right",
                    "landward_sign": int(sign),
                    "landward_azimuth_deg": landward_azimuth_deg,
                    "origin_x": float(origin.x),
                    "origin_y": float(origin.y),
                    "search_length_m": float(config.max_search_m),
                    "geometry": tran,
                }
            )
            bases_records.append(
                {
                    "path_id": path_idx,
                    "sample_id": i,
                    "measure_m": m,
                    "geometry": origin,
                }
            )

    # Build GeoDataFrames
    transect_columns = [
        "path_id",
        "sample_id",
        "measure_m",
        "landward_decision",
        "landward_side",
        "landward_sign",
        "landward_azimuth_deg",
        "origin_x",
        "origin_y",
        "search_length_m",
        "geometry",
    ]
    base_columns = ["path_id", "sample_id", "measure_m", "geometry"]
    transects_all = gpd.GeoDataFrame(
        transects_records,
        columns=transect_columns,
        geometry="geometry",
        crs=config.target_crs,
    )
    transect_bases = gpd.GeoDataFrame(
        bases_records,
        columns=base_columns,
        geometry="geometry",
        crs=config.target_crs,
    )

    # Export
    out_gpkg = os.path.join(out_dir, "transects.gpkg")
    safe_write_gpkg(transects_all, out_gpkg, "transects_all", mode="w")
    safe_write_gpkg(transect_bases, out_gpkg, "transect_bases", mode="a")

    logger.info(f"Part 2 complete: {out_gpkg}")
    if dem_vrt:
        dem_vrt.close()
    if src:
        src.close()
    return {"transects_all": transects_all, "transect_bases": transect_bases}
