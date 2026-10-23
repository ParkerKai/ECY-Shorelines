# barriers_to_migration/part1_trim.py
import logging
import os
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from config import COUNTY_CRS_EPSG, BarrierConfig, LayerSpec, normalize_asset_specs
from helpers import (
    ensure_crs,
    ensure_gpkg_writable,
    finalize_clean,
    normalize_county_name,
    write_layers_gpkg,
)

logger = logging.getLogger("barriers_to_migration")


def _buffer_distance_in_crs_units(
    target_crs: str | int, buffer_distance_m: float
) -> float:
    """
    Convert a buffer distance expressed in meters to the units of the target CRS.
    Supports meters and (US/international) feet. Falls back to meters if unknown.
    """
    crs = CRS.from_user_input(target_crs)
    # Default: meters
    distance = float(buffer_distance_m)
    try:
        unit_name = (crs.axis_info[0].unit_name or "").lower()
    except (IndexError, AttributeError):
        unit_name = ""

    if "metre" in unit_name or "meter" in unit_name:
        # Already meters
        pass
    elif (
        "us survey foot" in unit_name
        or "survey foot" in unit_name
        or "us foot" in unit_name
    ):
        distance *= 3.280833333  # meters -> US survey feet
    elif "foot" in unit_name or "feet" in unit_name or "ft" in unit_name:
        distance *= 3.28084  # meters -> international feet
    else:
        logger.warning(
            "Unknown CRS linear unit '%s' for %s; assuming meters for buffer distance.",
            unit_name,
            crs.to_string(),
        )
    return distance


def _read_layer_spec(name: str, spec: LayerSpec) -> gpd.GeoDataFrame:
    """Read one asset layer, honoring an optional layer name inside a GeoPackage."""
    logger.info("Reading asset layer '%s' from %s", name, spec.path)
    try:
        if spec.layer:
            return gpd.read_file(spec.path, layer=spec.layer)
        return gpd.read_file(spec.path)
    except Exception as exc:  # layer absent (e.g., no such assets in this county)
        logger.warning(
            "Asset layer '%s' could not be read (path=%s, layer=%s): %s. "
            "Continuing with an empty layer.",
            name,
            spec.path,
            spec.layer,
            exc,
        )
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{COUNTY_CRS_EPSG}")


def trim_and_export(
    county_name_lower: str,
    shoreline_path: str,
    county_bounds_path: str,
    asset_layers: dict[str, LayerSpec | str | dict],
    target_crs: str | int,
    out_dir: str,
    *,
    barrier_config: BarrierConfig,
    county_name_field: str | None = None,
    water_path: str | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Part 1:
    - Read the shoreline, county bounds, and an arbitrary set of asset layers
    - Set/transform CRS
    - Clip shoreline to the county boundary
    - Create a shoreline buffer (using BarrierConfig.buffer_distance_m) as the *barrier region*
    - Use the barrier region to clip every asset layer
    - Export everything to ONE GeoPackage (e.g., `<out_dir>/county_data.gpkg`)

    Parameters
    ----------
    county_name_lower : str
        County name (lowercase) for matching, e.g., 'thurston'.
    shoreline_path : str
        Path to the shoreline dataset (supports file formats readable by GeoPandas).
    county_bounds_path : str
        Path to county boundaries dataset.
    asset_layers : dict
        Mapping of layer name -> LayerSpec (or a plain path string / dict of
        LayerSpec fields). Any number and type of layers is supported; the names
        become the output GeoPackage layer names and the barrier `asset_type`
        labels used in Parts 2 and 3.
    target_crs : Union[str, int]
        Target CRS (e.g., 'EPSG:32149' or 32149). Should be projected; buffering assumes planar units.
    out_dir : str
        Output directory for the resulting GeoPackage.
    barrier_config : BarrierConfig
        Configuration containing `buffer_distance_m` (meters) to create the barrier region.
    county_name_field : Optional[str]
        Explicit county name field; if None, auto-detect among {'name','county','county_name','county_nm'}.
    water_path : Optional[str]
        Path to a visible surface water polygon layer. When supplied it is clipped
        to the shoreline buffer (not the county boundary, so offshore water is kept)
        and exported as the 'water' layer used by Part 2 for seaward determination.

    Returns
    -------
    Dict[str, gpd.GeoDataFrame]
        Dictionary of layers written to the GeoPackage, including 'county_boundary',
        'shoreline', 'barrier_region', and one entry per supplied asset layer.
    """
    # --- Setup & I/O ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_gpkg = os.path.join(out_dir, "county_data.gpkg")

    specs = normalize_asset_specs(asset_layers)
    for reserved in ("shoreline", "county_boundary", "barrier_region", "water"):
        if reserved in specs:
            raise ValueError(f"Asset layer name '{reserved}' is reserved.")

    logger.info("Reading input datasets...")
    shoreline = gpd.read_file(shoreline_path)
    county_bnds = gpd.read_file(county_bounds_path)
    assets = {name: _read_layer_spec(name, spec) for name, spec in specs.items()}
    water = None
    if water_path:
        logger.info("Reading visible surface water from %s", water_path)
        water = gpd.read_file(water_path)

    # --- CRS handling ---
    # Ensure county bounds CRS first, then clean
    county_bnds = finalize_clean(ensure_crs(county_bnds, COUNTY_CRS_EPSG))

    # Auto-detect county name field if not provided
    if county_name_field is None:
        candidate_cols = [
            c
            for c in county_bnds.columns
            if c.lower() in {"name", "county", "county_name", "county_nm"}
        ]
        if len(candidate_cols) == 0:
            raise ValueError(
                "County bounds must have a county name column (e.g., NAME/County/County_Name/County_Nm)."
            )
        county_name_field = candidate_cols[0]

    # Robust selection: coerce to string, strip, lowercase, and treat spaces/underscores alike
    county_key = normalize_county_name(county_name_lower)
    county_series = county_bnds[county_name_field].map(normalize_county_name)
    county_poly = county_bnds[county_series == county_key]
    if len(county_poly) == 0:
        raise ValueError(f"County '{county_name_lower}' not found in county bounds.")
    logger.info("Selected county rows: %d", len(county_poly))

    # Project county to target CRS
    county_poly_target = county_poly.to_crs(target_crs)
    county_union: BaseGeometry = county_poly_target.unary_union

    # Reproject & clean all operational layers
    shoreline = finalize_clean(ensure_crs(shoreline, target_crs))
    assets = {
        name: finalize_clean(ensure_crs(gdf, target_crs))
        for name, gdf in assets.items()
    }
    if water is not None:
        water = finalize_clean(ensure_crs(water, target_crs))

    # --- Desired behavior: shoreline -> county clip -> buffer -> clip other layers ---
    logger.info("Clipping shoreline to county boundary...")
    shoreline_clip = gpd.clip(shoreline, county_union)

    # If the shoreline clip is empty, warn and proceed with empty barrier
    if shoreline_clip.empty:
        logger.warning(
            "Shoreline clip produced no features inside county '%s'. Barrier region will be empty.",
            county_name_lower,
        )

    # Buffer distance handling (meters to CRS units)
    buffer_distance_m: float = float(getattr(barrier_config, "buffer_distance_m", 0.0))
    if buffer_distance_m <= 0:
        logger.warning(
            "BarrierConfig.buffer_distance_m is <= 0; barrier region will be empty and no clipping will occur."
        )

    barrier_buffer_distance = _buffer_distance_in_crs_units(
        target_crs, buffer_distance_m
    )

    logger.info(
        "Creating barrier region from shoreline buffer (%.3f in CRS units)...",
        barrier_buffer_distance,
    )
    shoreline_union: BaseGeometry = shoreline_clip.unary_union
    if buffer_distance_m <= 0 or shoreline_clip.empty or shoreline_union is None:
        # Create an empty GeoDataFrame with county schema
        barrier_region = shoreline_clip.iloc[0:0].copy()
    else:
        barrier_geom = shoreline_union.buffer(barrier_buffer_distance)
        # Keep barrier inside county (intersection ensures we don't leak outside)
        barrier_geom = barrier_geom.intersection(county_union)
        barrier_region = gpd.GeoDataFrame(
            geometry=[barrier_geom], crs=shoreline_clip.crs
        )

    # --- Clip other layers to barrier region ---
    def _clip_if_nonempty(
        gdf: gpd.GeoDataFrame, mask_geom: BaseGeometry
    ) -> gpd.GeoDataFrame:
        if gdf.empty or mask_geom is None or barrier_region.empty:
            return gdf.iloc[0:0].copy()
        return gpd.clip(gdf, mask_geom)

    barrier_mask = None if barrier_region.empty else barrier_region.geometry.unary_union

    assets_clip = {
        name: _clip_if_nonempty(gdf, barrier_mask) for name, gdf in assets.items()
    }

    # Water is clipped to the raw shoreline buffer so offshore water is retained
    # even where the county boundary stops at the shoreline.
    if water is None:
        water_clip = None
    elif water.empty or shoreline_clip.empty or shoreline_union is None:
        water_clip = water.iloc[0:0].copy()
    else:
        water_clip = gpd.clip(water, shoreline_union.buffer(barrier_buffer_distance))

    # --- Logging counts for edge-case visibility ---
    logger.info(
        "Post-clip feature counts — shoreline: %d, barrier_region polygons: %d, water: %s, %s",
        len(shoreline_clip),
        0 if barrier_region.empty else len(barrier_region),
        "n/a" if water_clip is None else len(water_clip),
        ", ".join(f"{name}: {len(gdf)}" for name, gdf in assets_clip.items()),
    )

    # --- Prepare export layers ---
    layers: dict[str, gpd.GeoDataFrame] = {
        "county_boundary": county_poly_target,  # full county in target CRS
        "shoreline": shoreline_clip,
        "barrier_region": barrier_region,  # buffered shoreline within county
        **({} if water_clip is None else {"water": water_clip}),
        **assets_clip,
    }

    # --- Export to a single GeoPackage ---
    ensure_gpkg_writable(out_gpkg)
    write_layers_gpkg(layers, out_gpkg)
    logger.info("Part 1 complete: %s", out_gpkg)

    return layers
