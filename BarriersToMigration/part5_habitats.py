import logging
import os

import geopandas as gpd
import pandas as pd

from config import BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import ensure_crs, finalize_clean, safe_write_gpkg

logger = logging.getLogger("barriers_to_migration")


def _read_habitat_layer(
    name: str, spec: LayerSpec, target_crs: str
) -> gpd.GeoDataFrame:
    logger.info("Reading habitat layer '%s' from %s", name, spec.path)
    gdf = (
        gpd.read_file(spec.path, layer=spec.layer)
        if spec.layer
        else gpd.read_file(spec.path)
    )
    return finalize_clean(ensure_crs(gdf, target_crs)).reset_index(drop=True)


def _aggregate_values(values: pd.Series) -> str | None:
    distinct = sorted({str(value) for value in values if not pd.isna(value)})
    return " | ".join(distinct) if distinct else None


def tag_habitat_intersections_and_export(
    transect_layers: dict[str, gpd.GeoDataFrame],
    config: BarrierConfig,
    out_dir: str,
    county_name: str,
    habitat_layers: dict[str, LayerSpec | str | dict],
    *,
    out_name: str = "transects_habitats.gpkg",
) -> dict[str, gpd.GeoDataFrame]:
    """Tag full transects with intersections and attributes from habitat layers.

    The output keeps one row per transect. For each habitat layer it adds a
    ``Hit_IN_<name>`` flag, a ``<name>_hit_count`` field, and one prefixed field
    per non-geometry habitat attribute. Multiple distinct attribute values are
    joined with `` | ``.
    """
    transects_all = transect_layers["transects_all"].copy().reset_index(drop=True)
    specs = normalize_asset_specs(habitat_layers)
    if not specs:
        raise ValueError("No habitat layers supplied to Part 5.")

    base_cols = list(transects_all.columns)
    hit_cols: list[str] = []
    transects_all["_transect_row_id"] = range(len(transects_all))

    for name, spec in specs.items():
        habitat = _read_habitat_layer(name, spec, config.target_crs)
        hit_col = f"Hit_IN_{name}"
        count_col = f"{name}_hit_count"
        hit_cols.append(hit_col)
        attribute_cols = [
            col for col in habitat.columns if col != habitat.geometry.name
        ]

        transects_all[hit_col] = False
        transects_all[count_col] = 0
        for col in attribute_cols:
            transects_all[f"{name}_{col}"] = None

        if transects_all.empty or habitat.empty:
            logger.warning(
                "%s: habitat '%s' or transects_all is empty; no intersections found.",
                county_name,
                name,
            )
            continue

        habitat_join = habitat[attribute_cols + [habitat.geometry.name]].copy()
        habitat_join["_habitat_feature_id"] = range(len(habitat_join))
        joined = gpd.sjoin(
            transects_all[["_transect_row_id", "geometry"]],
            habitat_join,
            how="inner",
            predicate="intersects",
        )

        if joined.empty:
            logger.info("%s: no transects intersect habitat '%s'.", county_name, name)
            continue

        grouped = joined.groupby("_transect_row_id", sort=False)
        row_ids = grouped.size().index
        transects_all.loc[row_ids, hit_col] = True
        counts = grouped["_habitat_feature_id"].nunique()
        transects_all.loc[counts.index, count_col] = counts.astype(int)

        for col in attribute_cols:
            aggregated = grouped[col].apply(_aggregate_values)
            transects_all.loc[aggregated.index, f"{name}_{col}"] = aggregated

        logger.info(
            "%s: %d of %d transects intersect habitat '%s'.",
            county_name,
            len(row_ids),
            len(transects_all),
            name,
        )

    transects_all = transects_all.drop(columns="_transect_row_id")
    transects_hits = transects_all[base_cols + hit_cols].copy()

    out_gpkg = os.path.join(out_dir, out_name)
    safe_write_gpkg(transects_all, out_gpkg, "transects_all", mode="w")
    safe_write_gpkg(transects_hits, out_gpkg, "transects_hits_only", mode="a")
    logger.info("Part 5 complete: %s", out_gpkg)

    return {"transects_all": transects_all, "transects_hits_only": transects_hits}
