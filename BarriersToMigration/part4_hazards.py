# barriers_to_migration/part4_hazards.py
import logging
import os

import geopandas as gpd

from config import BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import ensure_crs, finalize_clean, safe_write_gpkg

logger = logging.getLogger("barriers_to_migration")


def _read_hazard_layer(name: str, spec: LayerSpec, target_crs: str) -> gpd.GeoDataFrame:
    logger.info("Reading hazard layer '%s' from %s", name, spec.path)
    gdf = (
        gpd.read_file(spec.path, layer=spec.layer)
        if spec.layer
        else gpd.read_file(spec.path)
    )
    return finalize_clean(ensure_crs(gdf, target_crs))


def flag_hazard_hits_and_export(
    barrier_layers: dict[str, gpd.GeoDataFrame],
    config: BarrierConfig,
    out_dir: str,
    county_name: str,
    hazard_layers: dict[str, LayerSpec | str | dict],
    *,
    out_name: str = "barriers_hazards.gpkg",
) -> dict[str, gpd.GeoDataFrame]:
    """
    Part 4:
    - Load hazard polygon layers (e.g., CoSMoS overland flooding extents).
    - For each transect, test whether its first barrier hit falls inside each
      hazard polygon.
    - Emit `transects_all` and `barrier_hits` with one boolean column per
      hazard, named `Hit_IN_<hazard name>` (e.g., `Hit_IN_SLR050`).
    """
    transects_all = barrier_layers["transects_all"].copy()
    barrier_hits = barrier_layers["barrier_hits"].copy()

    specs = normalize_asset_specs(hazard_layers)
    if not specs:
        raise ValueError("No hazard layers supplied to Part 4.")

    key_cols = ["path_id", "sample_id"]

    for name, spec in specs.items():
        col = f"Hit_IN_{name}"
        hazard = _read_hazard_layer(name, spec, config.target_crs)

        if barrier_hits.empty or hazard.empty:
            logger.warning(
                "%s: hazard '%s' or barrier_hits is empty; %s set to False.",
                county_name,
                name,
                col,
            )
            barrier_hits[col] = False
            transects_all[col] = False
            continue

        joined = gpd.sjoin(
            barrier_hits[key_cols + ["geometry"]],
            hazard[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        inside_keys = set(
            map(tuple, joined[key_cols].drop_duplicates().to_numpy().tolist())
        )

        barrier_hits[col] = [
            (pid, sid) in inside_keys
            for pid, sid in zip(
                barrier_hits["path_id"], barrier_hits["sample_id"], strict=True
            )
        ]
        transects_all[col] = [
            (pid, sid) in inside_keys
            for pid, sid in zip(
                transects_all["path_id"], transects_all["sample_id"], strict=True
            )
        ]

        logger.info(
            "%s: %d of %d barrier hits fall within hazard '%s'.",
            county_name,
            int(barrier_hits[col].sum()),
            len(barrier_hits),
            name,
        )

    out_gpkg = os.path.join(out_dir, out_name)
    safe_write_gpkg(transects_all, out_gpkg, "transects_all", mode="w")
    safe_write_gpkg(barrier_hits, out_gpkg, "barrier_hits", mode="a")

    logger.info("Part 4 complete: %s", out_gpkg)

    return {"transects_all": transects_all, "barrier_hits": barrier_hits}
