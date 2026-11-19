"""Download CORA water levels to a folder, using tide gauge metadata for coordinates.

CORA data is stored in an S3 bucket made freely available via NOAA's Open Data
Dissemination (NODD) service: https://noaa-nos-cora-pds.s3.amazonaws.com/index.html

Parts of this code were guided by Signell, R. (2024). Reproducible Notebooks/
hurricane-ike-water-levels.
https://github.com/reproducible-notebooks/hurricane-ike-water-levels/tree/master

Available CORA datasets (named CORA-{domain}-{product}-{chunking}):
    fort.63        Hourly water levels
    swan_TPS.63    Hourly spectral peak wave periods
    swan_DIR.63    Hourly mean wave directions
    swan_HS.63     Hourly significant wave heights
    Grid           Water levels interpolated to a uniform 500 m grid
    {domain}       GEC or PAC
    {chunking}     "timeseries" for long series at few points,
                   "map" for many points over a few days
"""

import json
import sys
from datetime import datetime, timezone
import fsspec
import intake
import numpy as np
import pandas as pd
import urllib3
import zarr
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

# --- Configuration ---------------------------------------------------------

# Tide gauge archive and output paths
OUTPUT_FOLDER = Path(
    r"N:\SEA\Kai\Data\WLs\Cora"
)

DATASET = "CORA-PAC-fort.63-timeseries"
MAX_NODE_DISTANCE_KM = 10  # warn if nearest CORA node is farther than this
CORA_DATUM = "MSL"  # CORA water levels are referenced to NAVD88
CORA_SOURCE = (
    "NOAA CORA, https://noaa-nos-cora-pds.s3.amazonaws.com/index.html#cora_pac/"
)

# Certificate verification is disabled for S3 (as in the original notebook).
VERIFY_SSL = False
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def open_cora(dataset=DATASET):
    """Lazily open a CORA dataset (multi-TB; only requested pieces are loaded)."""
    # Separate catalogs exist for zarr 2 and zarr 3.
    catalog_file = (
        "CORA_intake_zarr3.yml" if int(zarr.__version__[0]) > 2 else "CORA_intake.yml"
    )
    storage_options = {"anon": True, "client_kwargs": {"verify": VERIFY_SSL}}

    # Applies to every s3fs instance, including those created inside zarr/xarray.
    fsspec.config.conf["s3"] = storage_options

    catalog = intake.open_catalog(
        "s3://noaa-nos-cora-pds/" + catalog_file, storage_options=storage_options
    )
    return catalog[dataset].to_dask()


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


################# Main #######

print(f"Output folder: {OUTPUT_FOLDER}")

print("\nOpening CORA dataset...")
ds = open_cora()


x_coords = ds['x'].compute()
y_coords = ds['y'].compute()
z = ds['depth'].compute()


# Create a DataFrame
df = pd.DataFrame({"x": x_coords, "y": y_coords, 'depth': z})

# Create geometry column using shapely Points
geometry = [Point(xy) for xy in zip(df["x"], df["y"])]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

#export
gdf.to_file(OUTPUT_FOLDER / 'Cora_grid.gpkg', driver="GPKG", layer="my_layer")

