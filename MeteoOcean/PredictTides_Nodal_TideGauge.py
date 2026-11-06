# -*- coding: utf-8 -*-
"""
Harmonic analysis (UTide) of pre-downloaded NOAA tide gauge data and comparison
of predicted tides for December 2025 vs December 2026.

Nodal corrections (18.61-yr satellite modulation of f/u) are applied in UTide.
UTide has no 18.61-yr long-period constituent ("N"/Node), so it is fit separately
by least squares together with the mean and a linear trend, before UTide is run on
the remainder. The nodal term is phase-locked to cos(N), N = longitude of the lunar
ascending node (equilibrium phase), so only its amplitude/sign is estimated.

UTide predictions are compared against NOAA CO-OPS harmonic predictions.

@author: kaparker
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import os
import json
import urllib.parse
import urllib.request
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utide

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
dir_in_tg = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\6_min"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WLs\TidePredictions\Nodal"
os.makedirs(dir_out, exist_ok=True)

Gauges = {
    "9443090": "Neah Bay",
    "9444090": "Port Angeles",
    "9444900": "Port Townsend",
    "9447130": "Seattle",
    "9446484": "Tacoma",
    "9449880": "Friday Harbor",
    "9449424": "Cherry Point",
    "9442396": "La Push",
    "9440910": "Toke Point",
    "9441102": "Westport",
    "9440581": "Cape Disappointment",
}

# Analysis window; use >= 18.61 years so the nodal constituent is resolvable
ANALYSIS_START = "2006-01-01"
ANALYSIS_END = "2025-12-01"

# Prediction windows to compare: PRED_N_MONTHS starting PRED_START_MONTH of each year
PRED_YEARS = [2025, 2026]
PRED_START_MONTH = 10
PRED_N_MONTHS = 5  # Oct-Feb
PRED_FREQ = "6min"

# Fit the 18.61-yr nodal constituent (equilibrium long-period tide)
FIT_NODAL_CONSTITUENT = True
# Include the fitted linear (sea-level) trend in predictions; off = astronomical tide only
INCLUDE_TREND_IN_PREDICTION = False

# Records shorter than this skip the 18.61-yr fit
MIN_RECORD_YEARS = 18.61

# NOAA CO-OPS predictions; datum/time zone match the tide gauge parquet (MSL, GMT)
NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_DATUM = "MSL"
dir_noaa = os.path.join(dir_out, "NOAA_Predictions")
os.makedirs(dir_noaa, exist_ok=True)

# ===============================================================================
# %% Functions
# ===============================================================================


def Read_TG_Parquet(dir_in_tg, gauge_ID, start_date=None, end_date=None):
    """Read observed water level (v_obs) from combined_resolved.parquet as a Series."""
    parquet_file = os.path.join(
        dir_in_tg, f"NOAA_TG_{gauge_ID}", "combined_resolved.parquet"
    )
    if not os.path.exists(parquet_file):
        print(f"Warning: {parquet_file} not found")
        return None

    df = pd.read_parquet(parquet_file)
    time_col = "t" if "t" in df.columns else "time"
    if time_col not in df.columns or "v_obs" not in df.columns:
        raise ValueError(f"Missing time or 'v_obs' column in {parquet_file}")

    s = pd.Series(
        df["v_obs"].values.astype(float),
        index=pd.to_datetime(df[time_col].values),
        name="wl",
    ).sort_index()
    s = s[~s.index.duplicated()]
    if start_date is not None:
        s = s[s.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        s = s[s.index < pd.to_datetime(end_date)]
    return s


def Read_TG_Latitude(dir_in_tg, gauge_ID):
    """Read station latitude from the gauge metadata JSON."""
    meta_files = glob(os.path.join(dir_in_tg, f"NOAA_TG_{gauge_ID}", "metadata_*.json"))
    if len(meta_files) == 0:
        raise FileNotFoundError(f"No metadata JSON found for gauge {gauge_ID}")
    with open(meta_files[0], "r") as f:
        return float(json.load(f)["Latitude"])


def lunar_node_longitude(t):
    """Mean longitude of the Moon's ascending node, N (radians)."""
    T = (pd.to_datetime(t) - pd.Timestamp("2000-01-01 12:00")) / pd.Timedelta(
        days=36525
    )
    return np.radians(125.04452 - 1934.136261 * np.asarray(T, dtype=float))


def low_freq_design(t, t0, fit_nodal):
    """Design matrix: [1, trend (yr), cos(N)]."""
    days = (pd.to_datetime(t) - t0) / pd.Timedelta(days=1)
    days = np.asarray(days, dtype=float)
    cols = [np.ones_like(days), days / 365.2422]
    if fit_nodal:
        cols.append(np.cos(lunar_node_longitude(t)))
    return np.column_stack(cols)


def fit_tides(wl_hourly, lat, fit_nodal):
    """
    Fit mean + trend + 18.61-yr constituent by least squares, then run UTide
    (nodal corrections on) on the remainder.
    """
    wl = wl_hourly.dropna()
    # Mid-record reference so that, without trend, the intercept is the record-mean level
    t0 = wl.index[0] + (wl.index[-1] - wl.index[0]) / 2

    # Daily means suppress the tidal signal so it does not leak into the low-frequency fit
    daily = wl.resample("1D").mean().dropna()
    X = low_freq_design(daily.index, t0, fit_nodal)
    beta, *_ = np.linalg.lstsq(X, daily.values, rcond=None)

    resid = wl.values - low_freq_design(wl.index, t0, fit_nodal) @ beta
    coef = utide.solve(
        wl.index,
        resid,
        lat=lat,
        nodal=True,
        trend=False,
        constit="auto",
        method="ols",
        conf_int="none",
        verbose=False,
    )
    return {"coef": coef, "beta": beta, "t0": t0, "fit_nodal": fit_nodal}


def predict_tides(model, t_pred):
    """Reconstruct tide at t_pred, adding back the low-frequency terms."""
    h = utide.reconstruct(t_pred, model["coef"], verbose=False).h
    beta = model["beta"].copy()
    if not INCLUDE_TREND_IN_PREDICTION:
        beta[1] = 0.0
    return h + low_freq_design(t_pred, model["t0"], model["fit_nodal"]) @ beta


def Download_NOAA_Predictions(gauge_ID, t_pred):
    """Download NOAA 6-min predictions covering t_pred, one calendar month per request."""
    parts = []
    for _, t_month in pd.Series(t_pred, index=t_pred).groupby(t_pred.to_period("M")):
        s = Download_NOAA_Month(gauge_ID, t_month.index)
        if s is None:
            return None
        parts.append(s)
    return pd.concat(parts)


def Download_NOAA_Month(gauge_ID, t_pred):
    """Download (or load cached) NOAA 6-min harmonic predictions covering t_pred."""
    begin, end = t_pred[0], t_pred[-1]
    cache = os.path.join(
        dir_noaa, f"{gauge_ID}_{NOAA_DATUM}_{begin:%Y%m%d}_{end:%Y%m%d}.csv"
    )
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["time"])
        return pd.Series(df["tide_m"].values, index=df["time"])

    params = {
        "product": "predictions",
        "station": gauge_ID,
        "begin_date": f"{begin:%Y%m%d %H:%M}",
        "end_date": f"{end:%Y%m%d %H:%M}",
        "datum": NOAA_DATUM,
        "units": "metric",
        "time_zone": "gmt",
        "interval": "6",
        "format": "json",
        "application": "ECY_Shorelines",
    }
    url = f"{NOAA_API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
    except (OSError, ValueError) as err:
        print(f"  NOAA download failed for {gauge_ID}: {err}")
        return None
    if "predictions" not in data:
        print(f"  NOAA returned no predictions for {gauge_ID}: {data.get('error')}")
        return None

    df = pd.DataFrame(data["predictions"])
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(df["t"]),
            "tide_m": pd.to_numeric(df["v"], errors="coerce"),
        }
    )
    df.to_csv(cache, index=False)
    return pd.Series(df["tide_m"].values, index=df["time"])


def season_window(year, freq):
    start = pd.Timestamp(year=year, month=PRED_START_MONTH, day=1)
    return pd.date_range(
        start, start + pd.DateOffset(months=PRED_N_MONTHS), freq=freq, inclusive="left"
    )


def season_label(year, sep=" - "):
    t = season_window(year, "1D")
    return f"{t[0]:%b %Y}{sep}{t[-1]:%b %Y}"


def tide_stats(h):
    return {"max": np.max(h), "range": np.ptp(h)}


def plot_comparison(gauge_ID, gauge_name, preds, noaa, fname):
    """Overlay seasonal predictions by elapsed days, with year-difference and UTide-NOAA panels."""
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    days = {}
    for i, (yr, (t, h)) in enumerate(preds.items()):
        c = f"C{i}"
        days[yr] = (t - t[0]) / pd.Timedelta(days=1)
        axes[0].plot(
            days[yr],
            h,
            color=c,
            lw=0.6,
            label=f"UTide {season_label(yr)} (max {h.max():.2f} m)",
        )
        if noaa.get(yr) is not None:
            nd = (noaa[yr].index - t[0]) / pd.Timedelta(days=1)
            axes[0].plot(
                nd,
                noaa[yr].values,
                color=c,
                lw=0.6,
                ls="--",
                alpha=0.7,
                label=f"NOAA {season_label(yr)} (max {noaa[yr].max():.2f} m)",
            )
            ut = pd.Series(h, index=t).reindex(noaa[yr].index)
            axes[2].plot(
                nd, ut.values - noaa[yr].values, color=c, lw=0.8, label=str(yr)
            )
    axes[0].set_ylabel(f"Predicted tide (m, {NOAA_DATUM})")
    axes[0].set_title(
        f"{gauge_name} ({gauge_ID}): UTide prediction with nodal corrections"
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    y0, y1 = PRED_YEARS[0], PRED_YEARS[1]
    n = min(len(preds[y0][1]), len(preds[y1][1]))
    axes[1].plot(
        days[y1][:n],
        preds[y1][1][:n] - preds[y0][1][:n],
        color="k",
        lw=0.8,
        label="UTide",
    )
    if noaa.get(y0) is not None and noaa.get(y1) is not None:
        m = min(len(noaa[y0]), len(noaa[y1]))
        nd = (noaa[y1].index[:m] - preds[y1][0][0]) / pd.Timedelta(days=1)
        axes[1].plot(
            nd,
            noaa[y1].values[:m] - noaa[y0].values[:m],
            color="r",
            lw=0.8,
            ls="--",
            label="NOAA",
        )
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel(f"{y1}/{str(y1 + 1)[-2:]} - {y0}/{str(y0 + 1)[-2:]} (m)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("UTide - NOAA (m)")
    axes[2].set_xlabel(f"Days since 1 {pd.Timestamp(2000, PRED_START_MONTH, 1):%B}")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# ===============================================================================
# %% Main loop
# ===============================================================================
summary = []
for gauge_ID, gauge_name in Gauges.items():
    print(f"\n=== {gauge_name} ({gauge_ID}) ===")
    wl = Read_TG_Parquet(dir_in_tg, gauge_ID, ANALYSIS_START, ANALYSIS_END)
    if wl is None or wl.dropna().empty:
        continue

    # On-the-hour samples; hourly bin means are labelled at bin start, which lags the phase ~27 min
    wl_hourly = wl[(wl.index.minute == 0) & (wl.index.second == 0)].asfreq("1h")
    record_yrs = (wl.index[-1] - wl.index[0]) / pd.Timedelta(days=365.2422)
    fit_nodal = FIT_NODAL_CONSTITUENT and record_yrs >= MIN_RECORD_YEARS
    if FIT_NODAL_CONSTITUENT and not fit_nodal:
        print(
            f"  Record is {record_yrs:.1f} yr (< {MIN_RECORD_YEARS}); skipping 18.61-yr constituent fit"
        )

    lat = Read_TG_Latitude(dir_in_tg, gauge_ID)
    model = fit_tides(wl_hourly, lat, fit_nodal)
    # Signed amplitude of the cos(N) term (positive = high when N = 0, i.e. major lunar standstill)
    amp_N = model["beta"][2] if fit_nodal else np.nan
    print(
        f"  Record: {record_yrs:.1f} yr, {len(model['coef'].name)} UTide constituents"
    )
    print(f"  18.61-yr constituent amplitude (cos N): {amp_N * 1000:.1f} mm")

    preds, noaa = {}, {}
    tag = {yr: season_label(yr, "-").replace(" ", "") for yr in PRED_YEARS}
    stem = f"{gauge_ID}_{gauge_name.replace(' ', '')}"
    for yr in PRED_YEARS:
        t_pred = season_window(yr, PRED_FREQ)
        preds[yr] = (t_pred, predict_tides(model, t_pred))
        pd.DataFrame({"time": t_pred, "tide_m": preds[yr][1]}).to_csv(
            os.path.join(dir_out, f"{stem}_{tag[yr]}.csv"), index=False
        )
        noaa[yr] = Download_NOAA_Predictions(gauge_ID, t_pred)

    row = {"gauge_ID": gauge_ID, "gauge_name": gauge_name}
    y0, y1 = PRED_YEARS
    for src, data in (
        ("UTide", {yr: preds[yr][1] for yr in PRED_YEARS}),
        ("NOAA", noaa),
    ):
        if any(data[yr] is None for yr in PRED_YEARS):
            continue
        for yr in PRED_YEARS:
            for k, v in tide_stats(np.asarray(data[yr])).items():
                row[f"{src}_{k}_{tag[yr]}_m"] = v
        for k in ("max", "range"):
            row[f"{src}_{k}_change_m"] = (
                row[f"{src}_{k}_{tag[y1]}_m"] - row[f"{src}_{k}_{tag[y0]}_m"]
            )
        print(
            f"  {src}: "
            + ", ".join(
                f"{tag[yr]} max = {row[f'{src}_max_{tag[yr]}_m']:.3f} m, "
                f"range = {row[f'{src}_range_{tag[yr]}_m']:.3f} m"
                for yr in PRED_YEARS
            )
        )
    summary.append(row)

    plot_comparison(
        gauge_ID,
        gauge_name,
        preds,
        noaa,
        os.path.join(dir_out, f"{stem}_{tag[y0]}_vs_{tag[y1]}.png"),
    )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    os.path.join(dir_out, "Summary_Season_TidePredictions.csv"), index=False
)
print(summary_df)
