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

# --- Configuration ---------------------------------------------------------

# Tide gauge archive and output paths
TIDE_GAUGE_ARCHIVE = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\6_min"
)
OUTPUT_FOLDER = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WLs\Cora\TideGauges"
)

DATASET = "CORA-PAC-fort.63-timeseries"
START_DATE = "1979-01-01"
END_DATE = "2026-01-01"
CORA_TO_M = 1 / 1000  # CORA water levels are in mm; convert to m
MAX_NODE_DISTANCE_KM = 10  # warn if nearest CORA node is farther than this
CORA_DATUM = "MSL"  # CORA water levels are referenced to NAVD88
CORA_SOURCE = (
    "NOAA CORA, https://noaa-nos-cora-pds.s3.amazonaws.com/index.html#cora_pac/"
)

# Certificate verification is disabled for S3 (as in the original notebook).
VERIFY_SSL = False
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- Metadata extraction ---------------------------------------------------


def load_tide_gauge_metadata(station_id):
    """Load latitude and longitude from tide gauge metadata file."""
    metadata = load_tide_gauge_metadata_dict(station_id)
    lat = metadata.get("Latitude")
    lon = metadata.get("Longitude")

    if lat is None or lon is None:
        raise ValueError(f"Missing latitude/longitude in metadata for {station_id}")

    return float(lat), float(lon)


def load_tide_gauge_metadata_dict(station_id):
    """Load the full tide gauge metadata dictionary."""
    metadata_file = list(
        (TIDE_GAUGE_ARCHIVE / f"NOAA_TG_{station_id}").glob(
            f"metadata_*{station_id}.json"
        )
    )[0]

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found for station {station_id}")

    with open(metadata_file, "r") as f:
        return json.load(f)


def discover_tide_gauges():
    """Discover all tide gauge station IDs from metadata files in archive."""
    station_ids = set()
    for metadata_file in TIDE_GAUGE_ARCHIVE.rglob("metadata_*.json"):
        # Extract station ID from filename: metadata_Name_StationID.json
        parts = metadata_file.stem.replace("metadata_", "").split("_")
        if len(parts) >= 2:
            station_id = parts[-1]
            station_ids.add(station_id)
    return sorted(list(station_ids))


# --- CORA ------------------------------------------------------------------


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


# --- Download CORA data to parquet -----------------------------------------


def download_cora_for_station(ds, station_id, station_name, lat, lon, start, end):
    """Download CORA data for a station and save to parquet plus a JSON sidecar."""
    try:
        # Find nearest node to the station coordinates
        dist_km = haversine(lat, lon, ds.y, ds.x).data
        node = int(np.argmin(dist_km))
        if dist_km[node] > MAX_NODE_DISTANCE_KM:
            print(
                f"! Warning: nearest node for {station_id} ({station_name}) is {dist_km[node]:.1f} km away"
            )

        # Extract data for this node and time period
        cora = ds["zeta"].sel(node=node, time=slice(start, end)).load()

        if bool(np.isnan(cora.values).all()):
            print(f"✗ Skipping {station_id} ({station_name}): all values are NaN")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(cora["time"].values),
                "water_level_m": cora.values * CORA_TO_M,
            }
        )

        # Save to parquet with a JSON metadata sidecar
        output_file = OUTPUT_FOLDER / f"{station_id}_{station_name}.parquet"
        df.to_parquet(output_file, index=False)

        tg_meta = load_tide_gauge_metadata_dict(station_id)
        metadata = {
            "date_processed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "station_name": station_name,
            "station_number": station_id,
            "units": "metric",
            "datum": CORA_DATUM,
            "time_zone": "gmt",
            "Latitude_TG": lat,
            "Latitude_Cora": float(ds.y[node].values),
            "Longitude_TG": lon,
            "Longitude_Cora": float(ds.x[node].values),
            "State": tg_meta.get("state"),
            "Begin_date": df["time"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "End_date": df["time"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": CORA_SOURCE,
        }
        with open(
            output_file.with_name(f"metadata_{station_name}_{station_id}.json"), "w"
        ) as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Downloaded {len(df)} records for {station_id} ({station_name})")
        return True

    except Exception as err:
        print(f"✗ Error downloading {station_id} ({station_name}): {err}")
        return False


def download_all(ds, stations, start, end):
    """Download CORA data for all stations."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for station_id, station_name in stations.items():
        try:
            lat, lon = load_tide_gauge_metadata(station_id)
            if download_cora_for_station(
                ds, station_id, station_name, lat, lon, start, end
            ):
                successful += 1
            else:
                failed += 1
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
            print(f"✗ Skipping {station_id}: {err}")
            failed += 1

    print(f"\nDownload complete: {successful} successful, {failed} failed")
    return successful, failed


################# MAin #######

print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Tide gauge archive: {TIDE_GAUGE_ARCHIVE}")
print(f"Date range: {START_DATE} to {END_DATE}\n")

# Discover stations from metadata files
print("Discovering tide gauge stations...")
stations_dict = {}
for station_id in discover_tide_gauges():
    try:
        lat, lon = load_tide_gauge_metadata(station_id)
        # Extract station name from metadata file if possible
        metadata_files = list(TIDE_GAUGE_ARCHIVE.rglob(f"metadata_*_{station_id}.json"))
        if metadata_files:
            # Extract name from filename: metadata_Name_StationID.json
            name = metadata_files[0].stem.replace("metadata_", "").rsplit("_", 1)[0]
            stations_dict[station_id] = name
        else:
            stations_dict[station_id] = station_id
    except Exception as err:
        print(f"Warning: Could not load metadata for {station_id}: {err}")


if not stations_dict:
    print("No tide gauge metadata files found!")
    sys.exit()

print(f"Found {len(stations_dict)} tide gauge stations")
print("\nOpening CORA dataset...")
ds = open_cora()

print("Downloading CORA data for each station...\n")
download_all(ds, stations_dict, START_DATE, END_DATE)
