"""
Barriers to Shoreline Migration
Build barrier hit points by sampling landward transects from a shoreline,
with merged shoreline paths and smoothed transect orientations to prevent crossing
and shore-parallel artifacts. Landward side is determined once per merged path
and applied uniformly to all transects on that path.

Outputs:
- `barrier_hits` (points at hit locations)
- `shoreline_barrier_distance_lines` (shoreline segments with the nearest
    transect hit distance)
- `transects_hits` and `transects_all` (QA layers)

Author: Kai Parker (WA Dept. of Ecology)
Date: 2026-08-05
"""

# ---------------------------
# Import
# ---------------------------
import logging
import os

from config import BarrierConfig, LayerSpec
from part1_trim import trim_and_export
from part2_transects import generate_transects
from part3_barriers import find_barriers_and_export
from part4_hazards import flag_hazard_hits_and_export
from part5_habitats import tag_habitat_intersections_and_export

# ---------------------------
# User configuration
# ---------------------------

county_list = [
#    "clallam",
#    "jefferson",
    "grays_harbor",
    "kitsap",
    "pacific",
    "wahkiakum",
]


# county = "wahkiakum"  # lower case

# Projection
TARGET_CRS = "EPSG:26910"  # NAD83 / UTM zone 10N


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
cfg = BarrierConfig(
    target_crs=TARGET_CRS,
    spacing_m=2.0,
    max_search_m=150.0,
    snap_tolerance_m=1.0,
    buffer_distance_m=150.0,
    orientation_smooth_m=30.0,
    max_turn_deg=25.0,
    elev_probe_dists_m=(10, 30, 60),
    elev_margin_min=0.20,
    asset_margin_min=0.20,
    asset_score_delta_min=0.10,
    elev_consistency_dists_m=(30, 60, 120),
    elev_consistency_delta_m=0.25,
)

for county in county_list:
    print(f"Processing county: {county}")

    # Output directory
    out_dir = os.path.join(
        r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\BarriersToMigration\CountyData_Outer",
        county,
    )

    # ---------------------------
    # Asset layers
    # ---------------------------
    # Add/remove entries freely: the key becomes the output layer name and the
    # barrier `asset_type` label. `kind` may be "line", "polygon", or "auto".
    ASSET_LAYERS: dict[str, LayerSpec] = {
        "armoring": LayerSpec(
            path=r"N:\SEA\Kai\Data\BarriersToMigration\OuterCoast\WA_OuterCoast_armoring.shp",
            kind="line",
            landward_weight=2.0,
        ),
        "buildings": LayerSpec(
            path=os.path.join(
                r"N:\SEA\Kai\Data\Assets\Unified", f"{county}_buildings_union.gpkg"
            ),
            kind="polygon",
            landward_weight=0.75,
        ),
        "roads": LayerSpec(
            path=os.path.join(r"N:\SEA\Kai\Data\Assets\OSM", f"{county}_roads.gpkg"),
            layer="osm_lines",
            kind="line",
            landward_weight=1.0,
        ),
        "rail": LayerSpec(
            path=os.path.join(r"N:\SEA\Kai\Data\Assets\OSM", f"{county}_rail.gpkg"),
            layer="rail_lines",
            kind="line",
            landward_weight=1.5,
        ),
        "levees": LayerSpec(
            path=r"N:\SEA\Kai\Data\Levees\Levees_combined.gpkg",
            kind="line",
            landward_weight=2.0,
        ),
        "rail_ecopia": LayerSpec(
            path=os.path.join(
                r"N:\SEA\Kai\Data\CCAP\Ecopia_ImperviousPolygons",
                f"{county}_landuse.gpkg",
            ),
            layer="railroad",
            kind="polygon",
            landward=False,
        ),
        "impervious": LayerSpec(
            path=os.path.join(
                r"N:\SEA\Kai\Data\CCAP\Ecopia_ImperviousPolygons",
                f"{county}_landuse.gpkg",
            ),
            layer="impervious",
            kind="polygon",
            landward=False,
        ),
    }

    """Run all three processing stages (trim/export, transect generation, barrier finding) for the configured county."""
    os.makedirs(out_dir, exist_ok=True)
    county_layers = trim_and_export(
        county_name_lower=county,
        shoreline_path=r"N:\SEA\Kai\Data\Shorelines\Outer_ShorelineMerged_WA_simp.shp",
        county_bounds_path=r"N:\SEA\Kai\Data\BarriersToMigration\CountyBounds.shp",
        asset_layers=ASSET_LAYERS,
        target_crs=TARGET_CRS,
        out_dir=out_dir,
        barrier_config=cfg,
        water_path=r"D:\Data\GIS_general\VisibleSurfaceWater_OpenWater.shp",
    )

    transects = generate_transects(
        county_layers=county_layers,
        config=cfg,
        dem_path=os.path.join(r"N:\SEA\Kai\Data\DEM_ByCounty", f"{county}_dem"),
        dem_source_crs="EPSG:2927",
        out_dir=out_dir,
        county_name=county,
        asset_layers=ASSET_LAYERS,
    )

    barriers = find_barriers_and_export(
        county_layers=county_layers,
        transect_layers=transects,
        config=cfg,
        out_dir=out_dir,
        county_name=county,
        asset_layers=ASSET_LAYERS,
    )

    # ---------------------------
    # Hazard layers (Part 4)
    # ---------------------------
    flood_dir = os.path.join(
        r"D:\Data\CoSMoS_Overland_flooding",
        f"{county.capitalize()}_flood_extent",
        f"CoSMoS-PS_flood_extent_projections_1-year_storm_{county.capitalize()}",
    )
    HAZARD_LAYERS: dict[str, LayerSpec] = {
        "SLR050": LayerSpec(
            path=os.path.join(
                flood_dir, f"{county.capitalize()}_fldPoly_slr050_rp001.shp"
            ),
            kind="polygon",
        ),
    }

    missing = [
        name for name, spec in HAZARD_LAYERS.items() if not os.path.exists(spec.path)
    ]
    tagged_transects = barriers
    if missing:
        logging.getLogger("barriers_to_migration").warning(
            "%s: skipping Part 4; hazard layer(s) not found: %s",
            county,
            ", ".join(missing),
        )
    else:
        tagged_transects = flag_hazard_hits_and_export(
            barrier_layers=barriers,
            config=cfg,
            out_dir=out_dir,
            county_name=county,
            hazard_layers=HAZARD_LAYERS,
        )

    # ---------------------------
    # Habitat layers (Part 5)
    # ---------------------------
    habitat_dir = r"D:\Data\Habitat"
    HABITAT_LAYERS: dict[str, LayerSpec] = {
        "shellfish_growing": LayerSpec(
            path=os.path.join(
                habitat_dir,
                "commercial shellfish growing area",
                "growingareas.shp",
            ),
            kind="polygon",
        ),
        "shellfish_harvest": LayerSpec(
            path=os.path.join(
                habitat_dir, "commercial shellfish harvest", "harvest_sites.shp"
            ),
            kind="auto",
        ),
        "eelgrass": LayerSpec(
            path=os.path.join(
                habitat_dir, "eelgrass", "generalized_eelgrass_polygons.shp"
            ),
            kind="polygon",
        ),
        "smelt_spawning": LayerSpec(
            path=os.path.join(habitat_dir, "Forage fish", "Smelt_Spawning.shp"),
            kind="line",
        ),
        "recreational_shellfish": LayerSpec(
            path=os.path.join(habitat_dir, "recreational shellfish", "beaches.shp"),
            kind="line",
        ),
        "sasi": LayerSpec(
            path=os.path.join(habitat_dir, "SASI", "SaSI.shp"),
            kind="line",
        ),
        "seal_haulout": LayerSpec(
            path=os.path.join(habitat_dir, "seal haul-out", "sealion_haulout.shp"),
            kind="polygon",
        ),
        "shorezone_line": LayerSpec(
            path=os.path.join(habitat_dir, "ShoreZone", "szline.shp"),
            kind="line",
        ),
        "shorezone_polygon": LayerSpec(
            path=os.path.join(habitat_dir, "ShoreZone", "szpoly.shp"),
            kind="polygon",
        ),
    }

    missing = [
        name for name, spec in HABITAT_LAYERS.items() if not os.path.exists(spec.path)
    ]
    if missing:
        logging.getLogger("barriers_to_migration").warning(
            "%s: skipping Part 5; habitat layer(s) not found: %s",
            county,
            ", ".join(missing),
        )
    else:
        tag_habitat_intersections_and_export(
            transect_layers=tagged_transects,
            config=cfg,
            out_dir=out_dir,
            county_name=county,
            habitat_layers=HABITAT_LAYERS,
        )

    print("Workflow complete.")
