import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from config import BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import ensure_crs, finalize_clean, safe_write_gpkg

logger = logging.getLogger("barriers_to_migration")


def _transect_origins(transects: gpd.GeoDataFrame) -> np.ndarray:
    """Shoreline origin point per transect (origin_x/y, else the middle vertex)."""
    coords = shapely.get_coordinates(transects.geometry.values)
    counts = shapely.get_num_coordinates(transects.geometry.values)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    mid = coords[starts + counts // 2] if len(coords) else np.empty((0, 2))
    ox = mid[:, 0].copy()
    oy = mid[:, 1].copy()
    if "origin_x" in transects.columns and "origin_y" in transects.columns:
        tx = pd.to_numeric(transects["origin_x"], errors="coerce").to_numpy(float)
        ty = pd.to_numeric(transects["origin_y"], errors="coerce").to_numpy(float)
        ok = ~(np.isnan(tx) | np.isnan(ty))
        ox[ok], oy[ok] = tx[ok], ty[ok]
    return shapely.points(ox, oy)


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


def _aggregate_values(joined: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Per `key`, join the sorted distinct non-null string values of `col` with ' | '."""
    vals = joined[[key, col]].dropna(subset=[col])
    if vals.empty:
        return pd.Series(dtype=object)
    vals = vals.assign(**{col: vals[col].astype(str)}).drop_duplicates()
    vals = vals.sort_values([key, col], kind="stable")
    # Group boundaries on the sorted keys; avoids pandas' per-group Python overhead.
    keys = vals[key].to_numpy()
    strs = vals[col].to_numpy(dtype=object)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    out = [
        strs[s] if e - s == 1 else " | ".join(strs[s:e])
        for s, e in zip(starts, ends)
    ]
    return pd.Series(out, index=keys[starts], dtype=object)


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
    ``Hit_IN_<name>`` flag and one prefixed field per non-geometry habitat
    attribute. Multiple distinct attribute values are joined with `` | ``. In
    the ``transects_hits_only`` layer the flags are named just ``<name>``.

    Layers whose spec sets ``columns`` (e.g. alongshore ShoreZone lines) are
    instead appended as attributes: each transect takes the values of the
    intersecting feature nearest its shoreline origin. The output column is
    ``<name>`` for a single column, otherwise ``<name>_<column>``; transects with
    no intersecting feature get null.
    """
    transects_all = transect_layers["transects_all"].copy().reset_index(drop=True)
    specs = normalize_asset_specs(habitat_layers)
    if not specs:
        raise ValueError("No habitat layers supplied to Part 5.")

    base_cols = list(transects_all.columns)
    clashes = [n for n in specs if n in base_cols]
    if clashes:
        raise ValueError(
            f"Habitat layer key(s) {clashes} clash with existing transect columns."
        )
    hit_cols: list[str] = []
    append_cols: list[str] = []
    origins: np.ndarray | None = None
    # Small, unfragmented frame used for every spatial join.
    transects_geom = gpd.GeoDataFrame(
        {"_transect_row_id": range(len(transects_all))},
        geometry=transects_all.geometry.values,
        crs=transects_all.crs,
    )

    # Build each layer's new columns in a dict and attach them all with a single
    # concat; inserting them one by one fragments the frame and slows later ops.
    new_columns: dict[str, pd.Series] = {}

    for name, spec in specs.items():
        habitat = _read_habitat_layer(name, spec, config.target_crs)
        append_mode = spec.columns is not None
        if append_mode:
            missing = [c for c in spec.columns if c not in habitat.columns]
            if missing:
                raise ValueError(
                    f"Habitat layer '{name}' is missing column(s) {missing}; "
                    f"available: {[c for c in habitat.columns if c != habitat.geometry.name]}"
                )
            attribute_cols = list(spec.columns)
        else:
            attribute_cols = [
                col for col in habitat.columns if col != habitat.geometry.name
            ]

        joined = None
        if transects_all.empty or habitat.empty:
            logger.warning(
                "%s: habitat '%s' or transects_all is empty; no intersections found.",
                county_name,
                name,
            )
        else:
            habitat_join = habitat[attribute_cols + [habitat.geometry.name]].copy()
            habitat_join["_habitat_feature_id"] = range(len(habitat_join))
            joined = gpd.sjoin(
                transects_geom,
                habitat_join,
                how="inner",
                predicate="intersects",
            )
            if joined.empty:
                logger.info(
                    "%s: no transects intersect habitat '%s'.", county_name, name
                )
                joined = None

        key = "_transect_row_id"

        if append_mode:
            out_names = (
                {attribute_cols[0]: name}
                if len(attribute_cols) == 1
                else {c: f"{name}_{c}" for c in attribute_cols}
            )
            if joined is None:
                for col, out_col in out_names.items():
                    new_columns[out_col] = pd.Series(
                        None, index=transects_all.index, dtype=object
                    )
            else:
                if origins is None:
                    origins = _transect_origins(transects_all)
                feat_ids = joined["_habitat_feature_id"].to_numpy()
                dists = shapely.distance(
                    origins[joined[key].to_numpy()],
                    habitat.geometry.values[feat_ids],
                )
                picked = (
                    joined.assign(_dist=dists, _fid=feat_ids)
                    .sort_values([key, "_dist", "_fid"], kind="stable")
                    .drop_duplicates(key)
                    .set_index(key)
                )
                for col, out_col in out_names.items():
                    series = picked[col].reindex(range(len(transects_all)))
                    series.index = transects_all.index
                    new_columns[out_col] = series
                logger.info(
                    "%s: %d of %d transects take attributes from '%s'.",
                    county_name,
                    len(picked),
                    len(transects_all),
                    name,
                )
            append_cols.extend(out_names.values())
            continue

        hit_col = f"Hit_IN_{name}"
        hit_cols.append(hit_col)
        hits = pd.Series(False, index=transects_all.index, dtype=bool)
        attrs = {
            col: pd.Series(None, index=transects_all.index, dtype=object)
            for col in attribute_cols
        }

        if joined is not None:
            hit_rows = joined[key].unique()
            hits.iloc[hit_rows] = True

            for col in attribute_cols:
                aggregated = _aggregate_values(joined, key, col)
                if not aggregated.empty:
                    attrs[col].iloc[aggregated.index.to_numpy()] = (
                        aggregated.to_numpy()
                    )

            logger.info(
                "%s: %d of %d transects intersect habitat '%s'.",
                county_name,
                len(hit_rows),
                len(transects_all),
                name,
            )

        new_columns[hit_col] = hits
        for col, series in attrs.items():
            new_columns[f"{name}_{col}"] = series

    if new_columns:
        new_df = pd.DataFrame(new_columns, index=transects_all.index)
        transects_all = transects_all.drop(
            columns=[c for c in new_df.columns if c in transects_all.columns]
        )
        transects_all = gpd.GeoDataFrame(
            pd.concat([transects_all, new_df], axis=1),
            geometry=transects_all.geometry.name,
            crs=transects_all.crs,
        )
    # In the hits-only layer the flags are named by the bare layer key.
    transects_hits = transects_all[base_cols + hit_cols + append_cols].rename(
        columns={c: c.removeprefix("Hit_IN_") for c in hit_cols}
    )

    out_gpkg = os.path.join(out_dir, out_name)
    safe_write_gpkg(transects_all, out_gpkg, "transects_all", mode="w")
    safe_write_gpkg(transects_hits, out_gpkg, "transects_hits_only", mode="a")
    logger.info("Part 5 complete: %s", out_gpkg)

    return {"transects_all": transects_all, "transects_hits_only": transects_hits}
