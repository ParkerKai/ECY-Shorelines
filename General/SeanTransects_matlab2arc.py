import mat73
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
import geopandas as gpd
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional


# ---------- Helpers (same as before) ----------
def matlab_datenum_to_datetime(dnums: np.ndarray) -> List:
    dnums = np.asarray(dnums, dtype=float).ravel()
    out = []
    for d in dnums:
        dt = (
            datetime.fromordinal(int(d))
            + timedelta(days=float(d % 1))
            - timedelta(days=366)
        )
        out.append(dt)
    return out


def safe_float(x, default=np.nan):
    try:
        a = np.asarray(x)
        if a.size == 0:
            return default
        return float(np.atleast_1d(a).ravel()[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return default


def safe_string(x, default=""):
    if x is None:
        return default
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    a = np.asarray(x)
    if a.dtype.kind in ("U", "S") and a.size > 0:
        return str(a.ravel()[0])
    return str(x)


def parse_epsg_from_utmzone(utmzone: Optional[str], *, wgs84_default=True) -> Optional:
    if utmzone is None:
        return None
    z = utmzone.strip().upper()
    digits = "".join(ch for ch in z if ch.isdigit())
    hemi = "N" if "N" in z and "S" not in z else ("S" if "S" in z else None)
    if not digits:
        return None
    zone = int(digits)
    if wgs84_default:
        if hemi == "S":
            return 32700 + zone
        return 32600 + zone
    else:
        return 26900 + zone


# --- Normalize to a list of dicts ---


def normalize_transects(obj: Any) -> List[Dict[str, Any]]:
    """
    Return a list of dicts, one dict per transect.
    Handles:
      - list of dicts
      - numpy object arrays containing dicts
      - dict of field arrays (columnar) -> row-wise dicts
      - singleton dict (1x1 struct) -> list of length 1
    """
    # Case A: already a list of dicts
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return obj

    # Case B: numpy object array of dicts
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.size > 0 and isinstance(obj.flat[0], dict):
            return obj.tolist()
        flat = obj.flatten().tolist()
        dicts = [x for x in flat if isinstance(x, dict)]
        if dicts:
            return dicts

    # Case C: dict of field arrays (columnar) -> build row dicts
    elif isinstance(obj, dict):

        def col_len(v) -> int:
            """Robust length for ragged/heterogeneous columns."""
            try:
                arr = np.array(v, dtype=object)
                return arr.shape[0] if arr.ndim > 0 else 1
            except Exception:
                try:
                    return len(v)  # fall back if it's a plain list/tuple
                except Exception:
                    return 1

        lengths = [col_len(v) for v in obj.values()]
        N = max(lengths) if lengths else 1

        rows: List[Dict[str, Any]] = []
        for i in range(N):
            row = {}
            for k, v in obj.items():
                arr = np.array(v, dtype=object)  # <-- key change: dtype=object
                if arr.ndim == 0:
                    # scalar or singleton
                    row[k] = arr.item() if arr.size == 1 else arr
                else:
                    if arr.shape[0] > i:
                        elem = arr[i]
                        # Unwrap trivial (length-1) nested arrays/lists
                        try:
                            if isinstance(elem, (list, np.ndarray)):
                                a = np.asarray(elem)
                                if a.size == 1:
                                    elem = a.ravel()[0]
                        except Exception:
                            pass
                        row[k] = elem
                    else:
                        row[k] = np.nan
            rows.append(row)
        return rows

    # Case D: single dict -> wrap in list
    elif isinstance(obj, dict):
        return [obj]

    raise TypeError(f"Unsupported transects object type: {type(obj)}")


# ---------- Load with mat73 ----------
MAT_FILE = r"N:\SEA\Kai\Data\Transects\transects_with_shorelines_final.mat"
data = mat73.loadmat(MAT_FILE)

# Identify variable names (adjust if needed)
TRANSECT_VAR_CANDIDATES = ["transects", "Transects", "T"]
UTMZONE_VAR_CANDIDATES = ["UTMZONE", "utmzone", "UTMZone"]

# --- Locate transects object ---
transects_obj = None
for cand in TRANSECT_VAR_CANDIDATES:
    if cand in data:
        transects_obj = data[cand]
        break

if transects_obj is None:
    # Fallback: pick the first plausible struct-like object
    for k, v in data.items():
        # Heuristics for "struct array"
        if isinstance(v, list) and v and isinstance(v[0], dict):
            transects_obj = v
            break
        if isinstance(v, dict) and any(
            isinstance(val, (list, np.ndarray)) for val in v.values()
        ):
            transects_obj = v
            break

if transects_obj is None:
    raise RuntimeError("Could not locate the transects struct array in the .mat file.")

transects_list = normalize_transects(transects_obj)

# Optional: quick sanity print (helpful for debugging)
print(
    f"Normalized transects: {len(transects_list)} records; first type: {type(transects_list[0])}"
)

# --- UTMZONE / CRS ---
utmzone = None
for cand in UTMZONE_VAR_CANDIDATES:
    if cand in data:
        utmzone = safe_string(data[cand])
        break
epsg = parse_epsg_from_utmzone(utmzone, wgs84_default=True)

# ---------- Build GeoDataFrame ----------
rows = []
ts_records = []

for tdict in transects_list:
    # Ensure we really have a dict (skip bad entries)
    if not isinstance(tdict, dict):
        continue

    # Scalars
    ID = safe_float(tdict.get("ID"), default=np.nan)
    mtype = safe_string(tdict.get("model_type"), default="")
    x_off = safe_float(tdict.get("x_off"))
    y_off = safe_float(tdict.get("y_off"))
    x_on = safe_float(tdict.get("x_on"))
    y_on = safe_float(tdict.get("y_on"))
    angle = safe_float(tdict.get("angle"))
    tanBeta = safe_float(tdict.get("tanBeta"))
    LTER = safe_float(tdict.get("LTER"))
    Y0 = safe_float(tdict.get("Y0"))
    Ymin = safe_float(tdict.get("Ymin"))

    # Geometry
    geom = LineString([(x_on, y_on), (x_off, y_off)])

    # Arrays
    Tarr = np.asarray(tdict.get("t")) if tdict.get("t") is not None else np.array([])
    Yarr = np.asarray(tdict.get("Y")) if tdict.get("Y") is not None else np.array([])
    SATarr = (
        np.asarray(tdict.get("SAT"))
        if tdict.get("SAT") is not None
        else np.array([], dtype=bool)
    )

    Tarr = Tarr.ravel()
    Yarr = Yarr.ravel()
    SATarr = SATarr.ravel().astype(bool) if SATarr.size else np.array([], dtype=bool)

    dt_list = matlab_datenum_to_datetime(Tarr) if Tarr.size else []
    n = (
        min(len(dt_list), len(Yarr), len(SATarr))
        if SATarr.size
        else min(len(dt_list), len(Yarr))
    )

    for k in range(n):
        ts_records.append(
            {
                "ID": ID,
                "time": dt_list[k],
                "Y": float(Yarr[k]) if np.isfinite(Yarr[k]) else np.nan,
                "SAT": bool(SATarr[k]) if SATarr.size else np.nan,
            }
        )

    rows.append(
        {
            "geometry": geom,
            "ID": ID,
            "model_type": mtype,
            "x_off": x_off,
            "y_off": y_off,
            "x_on": x_on,
            "y_on": y_on,
            "angle": angle,
            "tanBeta": tanBeta,
            "LTER": LTER,
            "Y0": Y0,
            "Ymin": Ymin,
            "utmzone": utmzone if utmzone else "",
        }
    )

# GeoDataFrame and outputs (same as before)
gdf = gpd.GeoDataFrame(rows, geometry="geometry")
if epsg is not None:
    gdf.set_crs(epsg=epsg, inplace=True)

# Keep shapefile-safe names
gdf = gdf.rename(columns={"model_type": "model_typ"})

num_cols = [
    "ID",
    "x_off",
    "y_off",
    "x_on",
    "y_on",
    "angle",
    "tanBeta",
    "LTER",
    "Y0",
    "Ymin",
]
for c in num_cols:
    gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
gdf["model_typ"] = gdf["model_typ"].astype(str)
gdf["utmzone"] = gdf["utmzone"].astype(str)

print("Writing shapefile: transects.shp")
gdf.to_file(r"N:\SEA\Kai\Data\Transects\transects.shp")

if ts_records:
    ts_df = pd.DataFrame(ts_records)
    print("Writing timeseries CSV: transect_timeseries.csv")
    ts_df.to_csv(r"N:\SEA\Kai\Data\Transects\transect_timeseries.csv", index=False)
