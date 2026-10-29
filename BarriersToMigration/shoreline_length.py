"""
Clip the raw shoreline to a county or city extent and report its length (miles).
Follows the clipping workflow in part1_trim.py.
"""

import os

import geopandas as gpd
import pandas as pd

from config import COUNTY_CRS_EPSG
from helpers import ensure_crs, finalize_clean, normalize_county_name

# ---------------------------
# User configuration
# ---------------------------
EXTENT_TYPE = "city"  # "county" or "city"
EXTENT_NAMES = [
    "Edmonds",
    "Mukilteo",
    "Bainbridge",
    "Burien",
    "Stanwood",
    "Des Moines",
]  # county or city names (case-insensitive)

SHORELINE_PATH = (
    r"D:\Data\SMU_ConditionSurvey_V1\OpenSourceVersions\ConditionSurvey.shp"
)
BOUNDS_PATHS = {
    "county": r"D:\Data\GIS_general\CountyBounds.shp",
    "city": r"N:\SEA\Kai\Data\Gis_General\City_Boundaries.shp",
}
# Set to override auto-detection of the boundary name column.
NAME_FIELD: str | None = None
NAME_FIELD_CANDIDATES = {
    "county": {"name", "county", "county_name", "county_nm"},
    "city": {"name", "CITY_DISSO", "city_name", "cityname", "city_nm", ""},
}

TARGET_CRS = "EPSG:26910"  # NAD83 / UTM zone 10N (meters)
METERS_PER_MILE = 1609.344
OUT_CSV = os.path.join(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Counties",
    f"shoreline_length_miles_{EXTENT_TYPE}.csv",
)


def _name_field(bounds: gpd.GeoDataFrame) -> str:
    if NAME_FIELD:
        return NAME_FIELD
    candidates = [
        c for c in bounds.columns if c.lower() in NAME_FIELD_CANDIDATES[EXTENT_TYPE]
    ]
    if not candidates:
        raise ValueError(
            f"No {EXTENT_TYPE} name column found; set NAME_FIELD. "
            f"Columns: {list(bounds.columns)}"
        )
    return candidates[0]


def main() -> None:

    NAME_FIELD = "CITY_DISSO"
    if EXTENT_TYPE not in BOUNDS_PATHS:
        raise ValueError("EXTENT_TYPE must be 'county' or 'city'.")

    shoreline = finalize_clean(ensure_crs(gpd.read_file(SHORELINE_PATH), TARGET_CRS))
    bounds = finalize_clean(
        ensure_crs(gpd.read_file(BOUNDS_PATHS[EXTENT_TYPE]), COUNTY_CRS_EPSG)
    )
    field = _name_field(bounds)
    keys = bounds[field].map(normalize_county_name)

    rows = []
    for name in EXTENT_NAMES:
        poly = bounds[keys == normalize_county_name(name)]
        if poly.empty:
            raise ValueError(
                f"{EXTENT_TYPE.capitalize()} '{name}' not found in '{field}'."
            )
        extent = poly.to_crs(TARGET_CRS).unary_union
        clipped = gpd.clip(shoreline, extent)
        length_mi = clipped.geometry.length.sum() / METERS_PER_MILE
        rows.append({EXTENT_TYPE: name, "shoreline_miles": round(length_mi, 2)})
        print(f"{name}: {length_mi:,.2f} mi")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
