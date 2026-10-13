"""
Created on 05/01/2026

This script Loads all tide gauge data, removes vertical land motion and reconciles
regional SLR.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# ===============================================================================
# Import Modules
# ===============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt  # zero-phase, stable
import matplotlib.dates as mdates
from typing import Union, Optional
import re
from pathlib import Path
from typing import Optional, Sequence, Union, IO

# ===============================================================================
# Functions
# ===============================================================================


def lowpass_hourly_with_gap_handling(
    df: pd.DataFrame,
    col: str = "v_obs",
    cutoff_days: float = 30.0,
    order: int = 6,
    fs_hours: float = 1.0,  # samples per hour
    small_gap_limit_hours: int = 48,  # interpolate interior gaps up to this length
    edge_fill_limit_hours: int = 48,  # temporarily fill leading/trailing NaNs up to this
    renan_buffer_hours: int = 24,  # re-NaN buffer around large gaps after filtering
    return_monthly: bool = True,
    missing_days_allowed: int = 15,  # PSMSL-style: if >15 days missing, mask monthly mean
):
    """
    Create an hourly, low-pass filtered series with robust gap handling:
      - Fill small interior gaps to run the filter.
      - Temporarily fill edges and large gaps, then re-NaN those regions after filtering.
    Optionally compute monthly means from the filtered output, masking months with insufficient raw coverage.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and a column `col`.
    col : Name of the water level column.
    cutoff_days : Low-pass cutoff in days (e.g., 30).
    order : Butterworth filter order (before forward-backward).
    fs_hours : Sampling rate (samples per hour).
    small_gap_limit_hours : Max interior gap length to interpolate.
    edge_fill_limit_hours : Max edge gap length to temporarily fill without classifying as "large".
    renan_buffer_hours : Hours to buffer on each side of large gaps when re-masking as NaN post-filter.
    return_monthly : If True, also return monthly means masked by raw-data availability.
    missing_days_allowed : How many missing days are allowed per month before masking.

    Returns
    -------
    result : dict with keys:
        'df_hourly' : DataFrame with columns [col, col+'_lp'] on hourly grid.
        'monthly_lp' : (optional) DataFrame of monthly means of the filtered series (masked).
        'large_gaps' : list of (start, end) timestamps for large gaps in the hourly grid.
    """

    # --- 0) Preconditions ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    df = df.sort_index()
    if df.index.freq is None:
        # Build a contiguous hourly index over full range
        full_idx = pd.date_range(
            df.index.min().floor("h"), df.index.max().ceil("h"), freq="h"
        )
        df_hourly = df.reindex(full_idx)
    else:
        # If already regular, enforce exact hourly frequency
        df_hourly = df.asfreq("h")

    # Raw hourly series
    s_raw = pd.to_numeric(df_hourly[col], errors="coerce")

    # --- 1) Identify gap runs (NaNs) and classify "large" vs "small" ---
    def find_nan_runs(s: pd.Series):
        idx = s.index
        mask = s.isna().to_numpy()
        runs = []
        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                    runs.append(
                        (idx[i], idx[j], j - i + 1)
                    )  # start_ts, end_ts, length_hours
                i = j + 1
            else:
                i += 1
        return runs

    nan_runs = find_nan_runs(s_raw)

    # Edge runs: those touching the first or last index
    first_idx = df_hourly.index[0]
    last_idx = df_hourly.index[-1]

    large_gaps = []
    for start_ts, end_ts, L in nan_runs:
        is_edge = (start_ts == first_idx) or (end_ts == last_idx)
        if is_edge and L > edge_fill_limit_hours:
            large_gaps.append((start_ts, end_ts))
        elif (not is_edge) and L > small_gap_limit_hours:
            large_gaps.append((start_ts, end_ts))
        # smaller gaps will be interpolated

    # --- 2) Fill small interior gaps by time interpolation ---
    s_work = s_raw.interpolate(
        method="time",
        limit=small_gap_limit_hours,
        limit_area="inside",
        limit_direction="both",
    )

    # --- 3) Temporarily fill edges up to edge_fill_limit_hours ---
    # This keeps leading/trailing short gaps from breaking the filter.
    s_work = s_work.ffill(limit=edge_fill_limit_hours).bfill(
        limit=edge_fill_limit_hours
    )

    # After step 2 & 3, we may still have NaNs (very long edge gaps or interior gaps > limit).
    # To ensure the filter can run, fill any remaining NaNs *temporarily* using nearest neighbor.
    if s_work.isna().any():
        s_work = (
            s_work.ffill().bfill()
        )  # remove all NaNs, but we will re-NaN large gaps later

    # --- 4) Design & apply low-pass (zero-phase) ---
    cutoff_cph = 1.0 / (cutoff_days * 24.0)  # cycles per hour
    sos = butter(N=order, Wn=cutoff_cph, btype="low", output="sos", fs=fs_hours)
    # Forward–backward zero-phase filtering (effective order doubled; magnitude squared). [2](https://jp.mathworks.com/help//signal/ref/filtfilt.html?lang=en)
    v_lp = sosfiltfilt(
        sos, s_work.to_numpy()
    )  # Requires finite values across the full array. [1](https://docs.scipy.org/doc//scipy-1.17.0/reference/generated/scipy.signal.sosfiltfilt.html)
    s_lp = pd.Series(v_lp, index=df_hourly.index)

    # --- 5) Re-NaN large gaps + buffer to avoid artifacts near gap edges ---
    if large_gaps:
        # Build a boolean mask to NaN the filtered output in those windows
        to_nan = pd.Series(False, index=df_hourly.index)
        # Precompute buffers as Timedelta for convenience
        buf = pd.Timedelta(hours=renan_buffer_hours)

        for g_start, g_end in large_gaps:
            # Extended window
            win_start = max(df_hourly.index[0], g_start - buf)
            win_end = min(df_hourly.index[-1], g_end + buf)
            to_nan.loc[win_start:win_end] = True

        s_lp = s_lp.mask(to_nan)

    # --- 6) Assemble hourly DataFrame ---
    out = df_hourly.copy()
    out[col] = s_raw
    out[f"{col}_lp"] = s_lp

    result = {"df_hourly": out, "large_gaps": large_gaps}

    # --- 7) Optional: monthly mean of filtered series, with a RAW-data coverage mask ---
    if return_monthly:
        # Count raw observed hours per month (before any interpolation)
        monthly_raw_hours = s_raw.notna().resample("MS").sum()

        # Threshold per month based on missing_days_allowed
        dim = monthly_raw_hours.index.days_in_month
        min_hours_required = np.maximum(dim - missing_days_allowed, 0) * 24

        monthly_mean_lp = s_lp.resample("MS").mean()

        # Mask months with insufficient raw coverage
        monthly_mean_lp_masked = monthly_mean_lp.where(
            monthly_raw_hours >= min_hours_required
        )

        result["monthly_lp"] = monthly_mean_lp_masked.rename(
            f"{col}_MonthlyMean_LP"
        ).reset_index()

    return result

def parse_meiv2(
    filepath_or_buffer: Union[str, Path, IO[str]],
    *,
    cols: Optional[Sequence[str]] = None,
    missing_sentinel: float = -999.00,
    set_index: bool = True,
    long: bool = False
) -> pd.DataFrame:
    """
    Parse a Multivariate ENSO Index (MEI.v2) text file into a pandas DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str | Path | IO[str]
        Path to the MEI.v2 text file or an already-open text file/stream
        (e.g., io.StringIO). Lines that don't start with a 4-digit year
        are ignored automatically.
    cols : Sequence[str], optional
        Column names to use. By default:
        ["YEAR","DJ","JF","FM","MA","AM","MJ","JJ","JA","AS","SO","ON","ND"]
        If provided, must contain 13 names (including "YEAR").
    missing_sentinel : float, default -999.00
        Numeric value used in the file to denote missing data. Will be
        replaced with NaN.
    set_index : bool, default True
        If True, set the "YEAR" column as the DataFrame index.
    long : bool, default False
        If True, return a long/tidy DataFrame with columns:
        ["YEAR", "SEASON", "MEI_v2"].

    Returns
    -------
    pandas.DataFrame
        A DataFrame in wide (default) or long format, with missing values
        replaced by NaN.

    Notes
    -----
    - Only lines beginning with a 4-digit year and followed by exactly 12
      values are kept.
    - Footer/header text and stray lines are ignored.
    """
    default_cols = ["YEAR","DJ","JF","FM","MA","AM","MJ","JJ","JA","AS","SO","ON","ND"]
    COLS = list(cols) if cols is not None else default_cols

    if len(COLS) != 13 or COLS[0] != "YEAR":
        raise ValueError(
            "cols must be 13 names including 'YEAR' followed by 12 seasonal names."
        )

    # Prepare line matcher: YEAR + the rest of the line
    year_line = re.compile(r"^\s*(\d{4})\s+(.+?)\s*$")

    # Handle path vs. open file/stream
    def _iter_lines() -> IO[str]:
        if hasattr(filepath_or_buffer, "read"):
            # Already a file-like object
            return filepath_or_buffer
        else:
            return Path(filepath_or_buffer).open("r", encoding="utf-8", errors="replace")

    rows = []
    needs_close = not hasattr(filepath_or_buffer, "read")
    f = _iter_lines()
    try:
        for raw in f:
            m = year_line.match(raw)
            if not m:
                continue

            year = int(m.group(1))
            tokens = m.group(2).split()
            if len(tokens) != 12:
                # Skip non-conforming data rows
                continue

            try:
                values = [float(x) for x in tokens]
            except ValueError:
                # Skip lines with non-numeric tokens
                continue

            rows.append([year] + values)
    finally:
        if needs_close:
            f.close()

    df = pd.DataFrame(rows, columns=COLS)

    # Replace sentinel missing values with NaN
    df = df.replace(missing_sentinel, np.nan)

    if set_index:
        df = df.set_index("YEAR")

    if long:
        # Produce tidy long format
        df_long = (
            df.reset_index()
              .melt(id_vars=["YEAR"], var_name="SEASON", value_name="MEI_v2")
              .sort_values(["YEAR", "SEASON"], ignore_index=True)
        )
        return df_long

    return df


# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\Hourly"
dir_enso = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\ENSO"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\RSLR\figures"
os.makedirs(dir_out, exist_ok=True)

# Monthly aggregation method
agg_method = "lowpass"  # Options: 'mean', 'lowpass'


# Record Length Trim (cut records to being within certain dates)
dt_start = None  # pd.Timestamp("1992-01-01")  or none



# ===============================================================================
# Tide Gauge Info
# ===============================================================================


TG = [
    # {"station": "9419750", "name": "CrescentCity"},
    # {"station": "9431647", "name": "PortOrford"},
    # {"station": "9432780", "name": "Charleston"},
    # {"station": "9435380", "name": "SouthBeach"},
    # {"station": "9437540", "name": "Garibaldi"},
    # {"station": "9439099", "name": "Wauna"},
    # {"station": "9440569", "name": "Skamokawa"},
    # {"station": "9440581", "name": "CapeDisappointment"},
    {"station": "9439040", "name": "Astoria"},
    {"station": "9440910", "name": "TokePoint"},
    # {"station": "9441102", "name": "Westport"},
    # {"station": "9442396", "name": "LaPush"},
    {"station": "9443090", "name": "NeahBay"},
    {"station": "9444090", "name": "PortAngeles"},
    {"station": "9444900", "name": "PortTownsend"},
    {"station": "9449880", "name": "FridayHarbor"},
    {"station": "9449424", "name": "CherryPoint"},
    {"station": "9447130", "name": "Seattle"},
    #{"station": "9446484", "name": "Tacoma"},
]

TG_table = pd.DataFrame(TG).set_index("station")


# ===============================================================================
# Load & Process Tide Gauge data 
# ===============================================================================
data_mnth = []
data_pred = []

TG_table["seasonal_amp"] = np.full(
    len(TG_table), np.nan
)  # To store seasonal amplitude for each station
for cnt, row in enumerate(TG_table.iterrows()):
    stat = row[0]

    print(f"Processing station {stat} ({cnt + 1}/{len(TG_table)})...")

    file_in = os.path.join(dir_in, f"NOAA_TG_{stat}", "combined_resolved.parquet")
    df = pd.read_parquet(file_in)

    # Make sure t is datetime
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").set_index("t")

    if dt_start is not None:
        print(f"Trimming data from station {stat} to specified {dt_start}")
        df = df.loc[pd.Timestamp(dt_start) :]

    # ---- Aggregation ----
    if agg_method == "mean":
        s = df["v_obs"]

        # Monthly mean and count (non-NaN)
        monthly_mean = s.resample("MS").mean()
        monthly_count = s.resample("MS").count()

        # Rule: require at least this many samples per month
        MIN_SAMPLES = 400  # A little over half a month for hourly sampled data

        # Mask insufficiently sampled months to NaN
        monthly_mean_masked = monthly_mean.mask(monthly_count < MIN_SAMPLES)

        monthly = monthly_mean_masked.rename("v_MonthlyMean").reset_index()

    elif agg_method == "lowpass":
        res = lowpass_hourly_with_gap_handling(
            df,
            col="v_obs",
            cutoff_days=30,
            order=6,
            fs_hours=1.0,
            small_gap_limit_hours=48,
            edge_fill_limit_hours=48,
            renan_buffer_hours=24,
            return_monthly=True,
            missing_days_allowed=15,
        )

        df_hourly = res[
            "df_hourly"
        ]  # columns: v_obs (raw), v_obs_lp (low-pass) with re-NaNs over big gaps
        monthly = res["monthly_lp"]  # masked monthly means of the low-pass series
        gaps = res["large_gaps"]  # list of (start, end) timestamps for large gaps

        monthly = monthly.rename(
            columns={"index": "t", "v_obs_MonthlyMean_LP": "MonthlyMean"}
        )

    else:
        raise ValueError("agg_method must be 'mean' or 'lowpass'.")

    # ---- Common time fields ----
    monthly["t"] = pd.to_datetime(
        monthly["t"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    monthly["t_mpl"] = mdates.date2num(monthly["t"])  # float days since 0001-01-01

    monthly["station"] = stat  # Add station column

    # --- Seasonal cycle (monthly climatology) ---

    # First Temporarily detrend to calculate the monthly signal
    temp = monthly.dropna(subset=["t_mpl", "MonthlyMean"])
    coef = np.polyfit(temp["t_mpl"], temp["MonthlyMean"], 1)  # slope in m/day
    poly = np.poly1d(coef)
    monthly["trend"] = poly(monthly["t_mpl"])

    data_detrend = monthly.copy()
    data_detrend["v_detrend"] = monthly["MonthlyMean"] - monthly["trend"]

    # Calendar month number (1-12) and name
    data_detrend["month"] = data_detrend["t"].dt.month
    data_detrend["month_name"] = data_detrend["t"].dt.month_name()

    # Compute climatology: mean over all years for each calendar month
    climo_mean = data_detrend.groupby("month")["v_detrend"].mean()
    climo_std = data_detrend.groupby("month")["v_detrend"].std()
    climo_n = data_detrend.groupby("month")["v_detrend"].count()
    climo_se = climo_std / np.sqrt(climo_n)

    # Ensure we have all 12 months; if any missing, interpolate across months
    full_months = pd.Index(range(1, 13), name="month")
    climo_mean = climo_mean.reindex(full_months)
    climo_std = climo_std.reindex(full_months)
    climo_n = climo_n.reindex(full_months).fillna(0)
    climo_se = climo_se.reindex(full_months)

    # If some months have no data, fill mean with neighbor interpolation
    if climo_mean.isna().any():
        climo_mean = climo_mean.interpolate(limit_direction="both")
    # For uncertainty, keep NaN when count==0 (no basis), else interpolate SE gently
    if climo_se.isna().any():
        climo_se = climo_se.interpolate(limit_direction="both")

    # Repeat the seasonal cycle across the full monthly index
    monthly["seasonal_cycle"] = data_detrend["month"].map(climo_mean)
    monthly["seasonal_cycle_se"] = data_detrend["month"].map(climo_se)

    # QA: amplitude (peak-to-peak) of the seasonal cycle for this station
    TG_table.at[stat, "seasonal_amp"] = (
        float(climo_mean.max() - climo_mean.min())
        if climo_mean.notna().any()
        else np.nan
    )

    monthly["anomaly"] = (
        monthly["MonthlyMean"] - monthly["seasonal_cycle"] - monthly["trend"]
    )

    data_mnth.append(monthly)


# Combine all stations
data_mnth = pd.concat(data_mnth, ignore_index=True)


# ===============================================================================
# Load & Process ENSO data 
# ===============================================================================

ENSO = parse_meiv2(os.path.join(dir_enso, "meiv2.txt"), long=True)


# ===============================================================================
# El Nino Year Comparison (March -> March) vs. Seasonal Cycle
# ===============================================================================
plot_station = "9447130"  # Seattle
enso_start_years = [1982, 1997, 2015, 2026]  # each spans Mar(year) -> Mar(year+1)
detrend_for_comparison = True  # remove long-term SLR trend so eras are comparable

d = (
    data_mnth[data_mnth["station"] == plot_station]
    .dropna(subset=["t"])
    .set_index("t")
    .sort_index()
)

series = d["MonthlyMean"] - d["trend"] if detrend_for_comparison else d["MonthlyMean"]

# March -> March ordering (13 points, March repeated at both ends)
month_seq = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3,4]
month_labels = [pd.Timestamp(2000, m, 1).strftime("%b") for m in month_seq]
x = np.arange(len(month_seq))

fig, ax = plt.subplots(figsize=(10, 6))

for yr in enso_start_years:
    idx = pd.date_range(f"{yr}-03-01", f"{yr + 1}-03-01", freq="MS")
    vals = series.reindex(idx).to_numpy()
    if np.all(np.isnan(vals)):
        print(f"No data for {yr}-{yr + 1} at station {plot_station}; skipping.")
        continue
    ax.plot(x, vals, marker="o", ms=4, lw=1.75, label=f"{yr}\u2013{yr + 1}")

# Seasonal climatology (one value per calendar month), wrapped Mar -> Mar
climo_mean = d.groupby(d.index.month)["seasonal_cycle"].first()
climo_se = d.groupby(d.index.month)["seasonal_cycle_se"].first()
seas = np.array([climo_mean.get(m, np.nan) for m in month_seq], dtype=float)
seas_se = np.array([climo_se.get(m, np.nan) for m in month_seq], dtype=float)

ax.plot(x, seas, "k--", lw=2.5, label="Mean seasonal cycle")
ax.fill_between(
    x, seas - 1.96 * seas_se, seas + 1.96 * seas_se, color="k", alpha=0.12, lw=0
)

ax.axhline(0, color="0.6", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("Month (March \u2192 March)")
ax.set_ylabel( "Monthly mean water level (cm)")
ax.set_title(
    f"El Ni\u00f1o years vs. seasonal cycle \u2014 "
    f"{TG_table.loc[plot_station, 'name']} ({plot_station})"
)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    os.path.join(dir_out, f"ENSO_MonthlyMean_MarMar_{plot_station}.png"), dpi=200
)
plt.show()


# ===============================================================================
# El Nino Year Comparison - Panel per Tide Gauge
# ===============================================================================
n_stat = len(TG_table)
n_col = 2
n_row = int(np.ceil(n_stat / n_col))

fig, axes = plt.subplots(
    n_row, n_col, figsize=(4 * n_col, 2.8 * n_row), sharex=True, sharey=True
)
axes = np.atleast_1d(axes).ravel()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, (stat, info) in zip(axes, TG_table.iterrows()):
    d = (
        data_mnth[data_mnth["station"] == stat]
        .dropna(subset=["t"])
        .set_index("t")
        .sort_index()
    )

    series = (
        d["MonthlyMean"] - d["trend"] if detrend_for_comparison else d["MonthlyMean"]
    )

    for k, yr in enumerate(enso_start_years):
        idx = pd.date_range(f"{yr}-03-01", f"{yr + 1}-03-01", freq="MS")
        vals = series.reindex(idx).to_numpy()
        ax.plot(
            x,
            vals,
            marker="o",
            ms=3,
            lw=1.5,
            color=colors[k % len(colors)],
            label=f"{yr}\u2013{yr + 1}",
        )

    climo_mean = d.groupby(d.index.month)["seasonal_cycle"].first()
    climo_se = d.groupby(d.index.month)["seasonal_cycle_se"].first()
    seas = np.array([climo_mean.get(m, np.nan) for m in month_seq], dtype=float)
    seas_se = np.array([climo_se.get(m, np.nan) for m in month_seq], dtype=float)

    ax.plot(x, seas, "k--", lw=2, label="Mean seasonal cycle")
    ax.fill_between(
        x, seas - 1.96 * seas_se, seas + 1.96 * seas_se, color="k", alpha=0.12, lw=0
    )

    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(f"{info['name']} ({stat})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_xlim(x[0], x[-1])

# Hide unused panels
for ax_extra in axes[n_stat:]:
    ax_extra.axis("off")

fig.supxlabel("Month")
fig.supylabel("Monthly mean water level (cm)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
fig.suptitle("El Ni\u00f1o years vs. seasonal cycle \u2014 all tide gauges")
fig.tight_layout(rect=(0.02, 0.05, 1, 0.97))
fig.savefig(os.path.join(dir_out, "ENSO_MonthlyMean_MarMar_AllStations.png"), dpi=200)
plt.show()
