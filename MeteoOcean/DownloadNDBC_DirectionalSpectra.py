"""Download the full NDBC directional wave spectrum record for a buoy and save as NetCDF.

Uses ndbc-api to discover/request the spectral files (swden, swdir, swdir2, swr1, swr2)
and combines them into a single (time, frequency) dataset.
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
from ndbc_api.api.parsers.http._base import BaseParser

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46088"
OUT_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\FullSpectral")
VERIFY_SSL = False  # False is needed behind the PNNL firewall; set True elsewhere

# mode: (variable name, available_historical label, attributes)
MODES = {
    "swden": (
        "spectral_density",
        "Spectral wave density data",
        {"long_name": "spectral wave energy density", "units": "m2 Hz-1"},
    ),
    "swdir": (
        "alpha1",
        "Spectral wave (alpha1) direction data",
        {
            "long_name": "mean wave direction (alpha1), direction waves come from",
            "units": "degree",
            "convention": "nautical, clockwise from true north",
        },
    ),
    "swdir2": (
        "alpha2",
        "Spectral wave (alpha2) direction data",
        {
            "long_name": "principal wave direction (alpha2), direction waves come from",
            "units": "degree",
            "convention": "nautical, clockwise from true north",
        },
    ),
    "swr1": (
        "r1",
        "Spectral wave (r1) direction data",
        {
            "long_name": "first normalized polar coordinate of the Fourier coefficients",
            "units": "1",
        },
    ),
    "swr2": (
        "r2",
        "Spectral wave (r2) direction data",
        {
            "long_name": "second normalized polar coordinate of the Fourier coefficients",
            "units": "1",
        },
    ),
}
MISSING_VALUES = [999.0, 9999.0]


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ndbc-api's text parser is incompatible with pandas>=3 and with pre-2005 headers; replace it.
def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_response(cls, response, use_timestamp=True):
    lines = [ln for ln in response.get("body", "").splitlines() if ln.strip()]
    is_header = [ln.startswith("#") or not _is_number(ln.split()[0]) for ln in lines]
    header = [ln for ln, h in zip(lines, is_header) if h]
    data = [ln for ln, h in zip(lines, is_header) if not h]
    if not header or not data:
        return pd.DataFrame()

    freqs = [float(n) for n in header[0].lstrip("#").split() if _is_number(n)]
    if not freqs:
        # realtime2 45-day files use a different format and have no matching swden file
        print("    skipping realtime file (no frequency header)")
        return pd.DataFrame()
    df = pd.read_csv(StringIO("\n".join(data)), sep=r"\s+", header=None)
    n_time = df.shape[1] - len(freqs)
    if n_time not in (4, 5):
        print(
            f"    skipping file with unexpected layout ({df.shape[1]} cols, {len(freqs)} freqs)"
        )
        return pd.DataFrame()

    t = df.iloc[:, :n_time].astype(int)
    t.columns = ["year", "month", "day", "hour", "minute"][:n_time]
    t["year"] = t["year"].where(t["year"] >= 100, t["year"] + 1900)
    out = df.iloc[:, n_time:].astype(float)
    out.columns = freqs
    out.index = pd.DatetimeIndex(pd.to_datetime(t), name="timestamp")
    return out.mask(out.isin(MISSING_VALUES))


BaseParser._read_response = classmethod(_read_response)


def available_years(api, station_id):
    avail = api.available_historical(station_id=station_id)
    years = set()
    for _, label, _ in MODES.values():
        for key in avail.get(label, {}):
            match = re.search(
                r"\d{4}", key
            )  # keys look like '1996', '2006b', 'Jan 2026'
            if match:
                years.add(int(match.group()))
    return sorted(years)


def download_mode(api, station_id, mode, years):
    now = utcnow()
    parts = []
    for yr in range(years[0], now.year + 1):
        end = min(datetime(yr, 12, 31, 23, 59), now)
        try:
            df = api.get_data(
                station_id=station_id,
                mode=mode,
                start_time=datetime(yr, 1, 1),
                end_time=end,
            )
        except Exception as e:  # missing years raise various ndbc-api exceptions
            print(f"  {mode} {yr}: no data ({type(e).__name__})")
            continue
        print(f"  {mode} {yr}: {len(df)} records")
        parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="first")].sort_index()


def main():
    if not VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    api = NdbcApi(verify_https=VERIFY_SSL)

    years = available_years(api, STATION_ID)
    if not years:
        raise SystemExit(f"No historical spectral data listed for station {STATION_ID}")
    print(f"Station {STATION_ID}: data listed for {years[0]}-{years[-1]}")

    frames = {}
    for mode in MODES:
        print(f"Downloading {mode} ...")
        df = download_mode(api, STATION_ID, mode, years)
        if df is None:
            continue
        if mode in ("swr1", "swr2"):
            # archived files store r1/r2 x100; values > 1 are non-physical
            df = df / 100.0
            df = df.where(df <= 1.0)
        frames[mode] = df
    if not frames:
        raise SystemExit("No spectral data downloaded.")

    times = pd.DatetimeIndex(
        sorted(set().union(*(df.index for df in frames.values()))), name="time"
    )
    freqs = np.array(sorted(set().union(*(df.columns for df in frames.values()))))

    ds = xr.Dataset(
        coords={
            "time": times,
            "frequency": (
                "frequency",
                freqs,
                {"long_name": "center frequency of band", "units": "Hz"},
            ),
        }
    )
    for mode, df in frames.items():
        name, _, attrs = MODES[mode]
        ds[name] = (
            ("time", "frequency"),
            df.reindex(index=times, columns=freqs).to_numpy(np.float32),
            attrs,
        )
    ds.attrs = {
        "title": f"NDBC station {STATION_ID} directional wave spectra",
        "station_id": STATION_ID,
        "source": "NOAA National Data Buoy Center (https://www.ndbc.noaa.gov), via ndbc-api",
        "directional_spectrum": (
            "E(f,theta) = S(f)/pi * [0.5 + r1*cos(theta-alpha1) "
            "+ r2*cos(2*(theta-alpha2))]"
        ),
        "history": f"Created {utcnow():%Y-%m-%d %H:%M} UTC by DownloadNDBC_DirectionalSpectra.py",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = (
        OUT_DIR
        / f"ndbc{STATION_ID}_directional_spectra_{times[0]:%Y%m%d}_{times[-1]:%Y%m%d}.nc"
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
