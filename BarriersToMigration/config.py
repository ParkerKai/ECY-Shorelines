# barriers_to_migration/config.py
from dataclasses import dataclass, replace

TARGET_CRS = "EPSG:26910"  # NAD83 / UTM zone 10N
COUNTY_CRS_EPSG = 2927  # WA State Plane North (HARN)


@dataclass
class LayerSpec:
    """Describes one optional asset input layer (roads, rail, armoring, ...).

    `kind` controls how the layer is turned into barrier geometry:
    "line" uses the features directly, "polygon" uses exterior boundaries,
    "auto" picks based on the dominant geometry type of the layer.
    """

    path: str
    layer: str | None = None
    kind: str = "auto"
    barrier: bool = True  # used as a barrier in Part 3
    landward: bool = True  # used for landward-side voting in Part 2
    landward_weight: float = 1.0
    # Part 5 only: if set, copy these attribute columns onto each transect
    # (value from the intersecting feature nearest the transect origin) instead
    # of adding a Hit_IN_<name> flag.
    columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.columns, str):
            self.columns = (self.columns,)
        elif self.columns is not None:
            self.columns = tuple(self.columns)
            if not self.columns:
                raise ValueError("LayerSpec.columns must not be empty when set.")

def normalize_asset_specs(
    asset_layers: dict[str, "LayerSpec | str | dict"] | None,
) -> dict[str, LayerSpec]:
    """Coerce a user-supplied asset mapping into {name: LayerSpec}."""
    specs: dict[str, LayerSpec] = {}
    for name, value in (asset_layers or {}).items():
        if isinstance(value, LayerSpec):
            specs[name] = replace(value)
        elif isinstance(value, str):
            specs[name] = LayerSpec(path=value)
        elif isinstance(value, dict):
            specs[name] = LayerSpec(**value)
        else:
            raise TypeError(
                f"Asset layer '{name}' must be a LayerSpec, path string, or dict; "
                f"got {type(value)!r}."
            )
    return specs


# @dataclass
# class Paths:
#     condition_survey: str
#     county_bounds: str
#     buildings_gpkg: str
#     roads_gpkg: str
#     rail_gpkg: str
#     levees_gpkg: str
#     dem_tif: str | None
#     out_dir: str


@dataclass
class BarrierConfig:
    target_crs: str = TARGET_CRS
    spacing_m: float = 50.0
    max_search_m: float = 1000.0
    snap_tolerance_m: float = 1.0
    dem_test_len_m: float = 25.0
    min_hit_distance_m: float = 0.0
    segment_fixed_onshore: bool = True
    buffer_distance_m: float = 150.0
    # "Behind barrier" check: after the first hit, look this far further along the
    # transect for any of these asset layers (keys of the asset layer mapping).
    behind_barrier_search_m: float = 6.0
    behind_barrier_layers: tuple[str, ...] = (
        "roads",
        "rail",
        "rail_ecopia",
        "buildings",
    )
    # Orientation smoothing
    orientation_smooth_m: float = 40.0
    eps_for_tangent_m: float = 1.0
    max_turn_deg: float | None = 25.0
    # Landward side probe spacing
    side_probe_stride_m: float = 25.0
    side_assets_offset_m: float = 25.0
    side_assets_buffer_m: float = 20.0
    # Visible surface water (primary landward/seaward signal)
    water_probe_dists_m: tuple[float, ...] = (
        5,
        15,
        30,
        60,
    )  # offshore/onshore probe distances (m)
    water_margin_min: float = (
        0.20  # min water vote margin before falling back to DEM/assets
    )
    elev_probe_dists_m: tuple[float, ...] = (
        10,
        30,
        60,
    )  # primary elevation probe distances (m)
    elev_margin_min: float = 0.20  # min elevation vote margin to consider "strong"
    asset_margin_min: float = 0.20  # min asset vote margin to permit elevation override
    asset_score_delta_min: float = 0.10  # per-sample asset score difference threshold
    elev_consistency_dists_m: tuple[float, ...] = (
        30,
        60,
        120,
    )  # path-level consistency probe distances (m)
    elev_consistency_delta_m: float = (
        0.25  # min median elevation diff (m) to trigger flip
    )

    def __post_init__(self) -> None:
        """Reject settings that would create invalid or non-terminating geometry work."""
        positive = {
            "spacing_m": self.spacing_m,
            "max_search_m": self.max_search_m,
            "dem_test_len_m": self.dem_test_len_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These distances must be > 0: {', '.join(invalid)}")
        if self.snap_tolerance_m < 0 or self.buffer_distance_m < 0:
            raise ValueError("snap_tolerance_m and buffer_distance_m must be >= 0")
        if self.behind_barrier_search_m < 0:
            raise ValueError("behind_barrier_search_m must be >= 0")
        if not self.elev_probe_dists_m or any(d <= 0 for d in self.elev_probe_dists_m):
            raise ValueError("elev_probe_dists_m must contain only positive distances")
        if not self.water_probe_dists_m or any(
            d <= 0 for d in self.water_probe_dists_m
        ):
            raise ValueError("water_probe_dists_m must contain only positive distances")
        if not self.elev_consistency_dists_m or any(
            d <= 0 for d in self.elev_consistency_dists_m
        ):
            raise ValueError(
                "elev_consistency_dists_m must contain only positive distances"
            )


# Asset labels (kept for attributes/QA)
ASSETS_ROADS = "roads"
ASSETS_RAIL = "rail"
ASSETS_BUILDINGS = "buildings"
ASSETS_ARMORING = "armoring"
ASSETS_SHORELINE = "shoreline"
ASSETS_RAIL_ECOPIA = "rail_ecopia"
ASSETS_IMPERVIOUS = "impervious"
ASSETS_LEVEES = "levees"
