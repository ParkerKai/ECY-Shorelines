"""
Created on 07/13/2026

Create a pandas DataFrame from Cascadia Tide Gauge Stations,
append Lat/Lon from station metadata JSON, and export to CSV.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# -------------------------------
# Packages
# -------------------------------
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------------------
# User Input
# -------------------------------

Stations = {
    "9419750": "CrescentCity",
    "9431647": "PortOrford",
    "9432780": "Charleston",
    "9435380": "SouthBeach",
    "9437540": "Garibaldi",
    "9439099": "Wauna",
    "9440569": "Skamokawa",
    "9439040": "Astoria",
    "9440581": "CapeDisappointment",
    "9440910": "TokePoint",
    "9441102": "Westport",
    "9442396": "LaPush",
    "9443090": "NeahBay",
    "9444090": "PortAngeles",
    "9444900": "PortTownsend",
    "9449880": "FridayHarbor",
    "9449424": "CherryPoint",
    "9447130": "Seattle",
    "9445958": "Bremerton",
    "9446484": "Tacoma",
}

# Base directories (adjust if needed)
dir_in_TG = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"
)
dir_out = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"
)

# -------------------------------
# Helpers
# -------------------------------


def parse_float(value: Any) -> float:
    """Safely parse a float from mixed types (str/num/None). Returns np.nan on failure."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        if isinstance(value, str):
            # Extract numeric part, preserving sign and decimal
            import re

            cleaned = "".join(re.findall(r"[-+]?\d*\.?\d+", value))
            try:
                return float(cleaned) if cleaned else np.nan
            except ValueError:
                return np.nan
        return np.nan


def first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional:
    """Return the first present, non-empty value among candidate keys."""
    for k in keys:
        if k in d and d[k] not in (None, "", "null"):
            return d[k]
    return None


def find_latest_metadata_file(base_dir: Path, station_id: str) -> Optional[Path]:
    """Find the most recent metadata_*_{station}.json inside base_dir/NOAA_TG_{station}."""
    tg_dir = base_dir / f"NOAA_TG_{station_id}"
    if not tg_dir.exists():
        return None
    matches = list(tg_dir.glob(f"metadata_*_{station_id}.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_lat_lon_from_json(meta_path: Path) -> Tuple[float, float]:
    """Load Latitude/Longitude from a metadata JSON file; returns (lat, lon) as floats."""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow list roots (take first element) but prefer dict
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return np.nan, np.nan

    lat_raw = first_present(data, ["Latitude", "Lat", "lat", "latitude"])
    lon_raw = first_present(data, ["Longitude", "Lon", "lon", "longitude"])

    lat = parse_float(lat_raw)
    lon = parse_float(lon_raw)

    return lat, lon


# -------------------------------
# Main
# -------------------------------

# 1) Create a DataFrame from the Stations dict
df = pd.DataFrame.from_dict(Stations, orient="index", columns=["StationName"])
df.index.name = "StationID"
df = df.reset_index()

# 2) Append Latitude/Longitude by reading station metadata JSON
latitudes = []
longitudes = []
meta_paths = []
missing_meta = []

for station_id in df["StationID"]:
    meta_path = find_latest_metadata_file(dir_in_TG, station_id)
    if meta_path is None:
        latitudes.append(np.nan)
        longitudes.append(np.nan)
        meta_paths.append(None)
        missing_meta.append(station_id)
        continue

    lat, lon = load_lat_lon_from_json(meta_path)
    latitudes.append(lat)
    longitudes.append(lon)
    meta_paths.append(str(meta_path))

df["Latitude"] = latitudes
df["Longitude"] = longitudes

# 3) Export to CSV
out_path = dir_out / "Cascadia_Tide_Gauges_LonLat.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

# Optional console output
print(f"Exported {len(df)} stations to: {out_path}")
if missing_meta:
    print(f"Missing metadata files for {len(missing_meta)} stations: {missing_meta}")

# Optional quick peek
# print(df.head())
