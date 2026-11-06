"""Download the full NDBC bulk wave record for a buoy and save as NetCDF.

Uses ndbc-api to request the standard meteorological (stdmet) files and keeps the
bulk wave parameters (WVHT, DPD, APD, MWD).
"""

import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import xarray as xr
from ndbc_api import NdbcApi
from ndbc_api.api.parsers.http.stdmet import StdmetParser

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46088"
OUT_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\BulkWaves")
VERIFY_SSL = False  # False is needed behind the PNNL firewall; set True elsewhere

AVAIL_LABEL = "Standard meteorological data"
# NDBC column: (variable name, missing-value flag, attributes)
WAVE_VARS = {
    "WVHT": (
        "Hs",
        99.0,
        {"long_name": "significant wave height", "units": "m"},
    ),
    "DPD": (
        "Tp",
        99.0,
        {"long_name": "dominant (peak) wave period", "units": "s"},
    ),
    "APD": (
        "Tm",
        99.0,
        {"long_name": "average wave period", "units": "s"},
    ),
    "MWD": (
        "Dp",
        999.0,
        {
            "long_name": "wave direction at the dominant period, direction waves come from",
            "units": "degree",
            "convention": "nautical, clockwise from true north",
        },
    ),
}
TIME_COLS = ["year", "month", "day", "hour", "minute"]


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ndbc-api's text parser is incompatible with pandas>=3 and with pre-2005 headers; replace it.
def _read_response(cls, response, use_timestamp=True):
    lines = [ln for ln in response.get("body", "").splitlines() if ln.strip()]
    is_header = [ln.startswith("#") or not _is_number(ln.split()[0]) for ln in lines]
    header = [ln for ln, h in zip(lines, is_header) if h]
    data = [ln for ln, h in zip(lines, is_header) if not h]
    if not header or not data:
        return pd.DataFrame()

    names = header[0].lstrip("#").split()
    # some archive years list trailing columns (e.g. TIDE) that the data rows omit
    df = pd.read_csv(
        StringIO("\n".join(data)),
        sep=r"\s+",
        header=None,
        names=names,
        index_col=False,
        na_values=["MM"],
    )

    # time columns: YY/YYYY, MM, DD, hh, [mm]; minute column absent before 2005
    n_time = 5 if names[4] == "mm" else 4
    t = df.iloc[:, :n_time].astype(int)
    t.columns = TIME_COLS[:n_time]
    t["year"] = t["year"].where(t["year"] >= 100, t["year"] + 1900)

    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(t), name="timestamp"))
    for col, (_, missing, _) in WAVE_VARS.items():
        vals = df[col].astype(float).to_numpy() if col in df else np.nan
        out[col] = vals
        out[col] = out[col].mask(out[col] >= missing)
    return out


StdmetParser._read_response = classmethod(_read_response)


def available_years(api, station_id):
    avail = api.available_historical(station_id=station_id)
    years = set()
    for key in avail.get(AVAIL_LABEL, {}):
        match = re.search(r"\d{4}", key)  # keys look like '1996', '2006b', 'Jan 2026'
        if match:
            years.add(int(match.group()))
    return sorted(years)


def download_stdmet(api, station_id, years):
    now = utcnow()
    parts = []
    for yr in sorted(set(years) | {now.year}):
        end = min(datetime(yr, 12, 31, 23, 59), now)
        try:
            df = api.get_data(
                station_id=station_id,
                mode="stdmet",
                start_time=datetime(yr, 1, 1),
                end_time=end,
            )
        except Exception as e:  # missing years raise various ndbc-api exceptions
            print(f"  stdmet {yr}: no data ({type(e).__name__})")
            continue
        print(f"  stdmet {yr}: {len(df)} records")
        parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.dropna(how="all")


def main():
    if not VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    api = NdbcApi(verify_https=VERIFY_SSL)

    years = available_years(api, STATION_ID)
    if not years:
        raise SystemExit(f"No historical stdmet data listed for station {STATION_ID}")
    print(f"Station {STATION_ID}: data listed for {years[0]}-{years[-1]}")

    print("Downloading stdmet ...")
    df = download_stdmet(api, STATION_ID, years)
    if df is None or df.empty:
        raise SystemExit("No bulk wave data downloaded.")

    times = pd.DatetimeIndex(df.index, name="time")
    ds = xr.Dataset(coords={"time": times})
    for col, (name, _, attrs) in WAVE_VARS.items():
        ds[name] = ("time", df[col].to_numpy(np.float32), {**attrs, "ndbc_name": col})
    ds.attrs = {
        "title": f"NDBC station {STATION_ID} bulk wave parameters",
        "station_id": STATION_ID,
        "source": "NOAA National Data Buoy Center (https://www.ndbc.noaa.gov), via ndbc-api",
        "history": f"Created {utcnow():%Y-%m-%d %H:%M} UTC by DownloadNDBC_BulkWaves.py",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = (
        OUT_DIR / f"ndbc{STATION_ID}_bulk_waves_{times[0]:%Y%m%d}_{times[-1]:%Y%m%d}.nc"
    )
    encoding = {
        v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
        for v in ds.data_vars
    }
    encoding["time"] = {"units": "seconds since 1970-01-01 00:00:00", "dtype": "int64"}
    ds.to_netcdf(out_file, encoding=encoding)
    print(f"Saved {out_file}")
    print(ds)


if __name__ == "__main__":
    main()
