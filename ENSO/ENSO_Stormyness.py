"""
Extract POT extreme water levels from tide gauge observations and compare El Nino
ENSO years (March -> February) against all other years, per tide gauge.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D

sns.set_theme(style="darkgrid")
# Make gridlines more visible
sns.set_context("notebook", rc={"grid.linewidth": 1.2})
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['grid.linewidth'] = 1.2

module_dir = r"C:\Users\kaip461\GitHub\Kai_Python\General_Functions"
sys.path.insert(0, module_dir)
from POT_Extremes_XR import compute_return_intervals_forwardstop_map

# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\Hourly"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures"
os.makedirs(dir_out, exist_ok=True)

# Select tide gauges (same table as ENSO_WaterLevels_WAcombined.py)
TG = [
    {"station": "9440569", "name": "Skamokawa"},
    {"station": "9439040", "name": "Astoria"},
    {"station": "9440910", "name": "TokePoint"},
    {"station": "9441102", "name": "Westport"},
    {"station": "9442396", "name": "LaPush"},
    {"station": "9443090", "name": "NeahBay"},
    {"station": "9444090", "name": "PortAngeles"},
    {"station": "9444900", "name": "PortTownsend"},
    {"station": "9449880", "name": "FridayHarbor"},
    {"station": "9449424", "name": "CherryPoint"},
    {"station": "9447130", "name": "Seattle"},
    {"station": "9446484", "name": "Tacoma"},
]

# Record trim (None or e.g. pd.Timestamp("1992-01-01"))
dt_start = None
# Fraction of hours with valid data for an ENSO year to count as in the record
min_coverage = 0.5

# POT settings
RPs = np.array([1, 5, 10, 25, 50, 100])
decluster_window = "72h"
# 'peaks' keeps independent peaks >= decluster_window apart; 'runs' lets
# successive high tides chain into multi-month clusters at tidal-range thresholds.
decluster_method = "peaks"
# ForwardStop threshold search band (percentiles of hourly water level)
pot_pct_low = 99.6
pot_pct_high = 99.9

# El Nino peak years; each ENSO year runs Mar(year-1) -> Feb(year)
enso_peak_years = [1983, 1998, 2016]
enso_year_labels = {1983: "1982-1983", 1998: "1997-1998", 2016: "2015-2016"}
detrend_timeseries_before_pot = True

# ===============================================================================
# Functions
# ===============================================================================


def load_tide_gauges(tg_list, dir_in, dt_start=None):
    """Load hourly observed water levels (m) into a (station, time) DataArray."""
    series = {}
    for row in tg_list:
        stat = row["station"]
        print(f"Loading station {stat} ({row['name']})...")
        df = pd.read_parquet(
            os.path.join(dir_in, f"NOAA_TG_{stat}", "combined_resolved.parquet")
        )
        df["t"] = pd.to_datetime(df["t"])
        s = df.sort_values("t").drop_duplicates("t").set_index("t")["v_obs"]
        if dt_start is not None:
            s = s.loc[pd.Timestamp(dt_start) :]
        series[stat] = pd.to_numeric(s, errors="coerce")

    wide = pd.DataFrame(series).sort_index()
    return xr.DataArray(
        wide.to_numpy().T,
        dims=("station", "time"),
        coords={"station": list(wide.columns), "time": wide.index.to_numpy()},
        name="waterlevel",
    )


def detrend_timeseries(da):
    """Remove each station's linear trend while preserving its midpoint level."""
    out = da.copy()
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    time_years = (times - times[0]).total_seconds() / (365.2425 * 24 * 3600)
    center_year = np.nanmean(time_years)

    for i in range(da.sizes["station"]):
        values = da.values[i]
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        slope = np.polyfit(time_years[valid], values[valid], 1)[0]
        out.values[i] = values - slope * (time_years - center_year)
    return out


def extremes_to_frame(da, extreme_ds, names):
    """Flatten padded event indices to a tidy (station, t, value) frame."""
    idx = extreme_ds["extreme_time_index"].values
    values = da.values
    times = da["time"].values

    rows = []
    for i, stat in enumerate(da["station"].values):
        ev = idx[i][idx[i] >= 0]
        rows.append(
            pd.DataFrame(
                {
                    "station": stat,
                    "name": names[stat],
                    "t": times[ev],
                    "value": values[i, ev],
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    months = df["t"].dt.month
    # ENSO year runs Mar(y-1) -> Feb(y) and is labelled by its peak year.
    df["enso_year"] = (df["t"].dt.year + (months >= 3).astype(int)).astype(np.int16)
    return df


def station_year(df):
    """Mean and max of the event values per station / ENSO year."""
    return (
        df.groupby(["station", "name", "enso_year"], observed=True)["value"]
        .agg(year_mean="mean", year_max="max")
        .reset_index()
    )


def station_event_counts(df, in_record, names):
    """Count POT events for every coverage-qualified station / ENSO year."""
    rows = []
    for stat, years in in_record.items():
        counts = df.loc[df["station"] == stat].groupby("enso_year").size()
        for year in sorted(years):
            rows.append(
                {
                    "station": stat,
                    "name": names[stat],
                    "enso_year": year,
                    "event_count": counts.get(year, 0),
                }
            )
    return pd.DataFrame(rows)


def record_enso_years(da, min_coverage=0.5):
    """Which ENSO years have valid data coverage for each station?

    An ENSO year is in the record when at least `min_coverage` of its hours have
    finite observations at that station (same rule as ENSO_AnnualMax.py).
    """
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    enso_year = times.year + (times.month >= 3).astype(int)
    expected_hours = 365 * 24
    in_record = {}
    for i, stat in enumerate(da["station"].values):
        coverage = pd.Series(np.isfinite(da.values[i]), index=enso_year).groupby(level=0).sum()
        in_record[stat] = sorted(
            int(y) for y in coverage.index[coverage >= min_coverage * expected_hours]
        )
    return in_record


def _grid(n, ncol=3, w=6.4, h=4.4):
    """Return figure and axes for n subplots in a grid."""
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(w * ncol, h * nrow), sharey=True, squeeze=False
    )
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def yearly_stat_figure(
    yearly, in_record, col, label, fname, y_label=None, title_label=None
):
    """Per station: distribution of all `col`, El Nino years as colored dots.
    El Nino years inside the record with no POT event get an 'x' on the bottom axis.
    """
    stations = list(dict.fromkeys(yearly["station"]))
    fig, axes = _grid(len(stations))
    
    # Define colors for each El Niño year
    enso_colors = {1983: "#1f77b4", 1998: "#ff7f0e", 2016: "#2ca02c"}

    for ax, stat in zip(axes, stations):
        sub = yearly[yearly["station"] == stat]
        other = sub[~sub["enso_year"].isin(enso_peak_years)]
        enso = sub[sub["enso_year"].isin(enso_peak_years)]
        
        # Get station name for title
        station_name = next((row["name"] for row in TG if row["station"] == stat), stat)

        if len(other) > 1:
            sns.violinplot(
                y=other[col],
                x=np.zeros(len(other)),
                color="0.75",
                cut=0,
                inner=None,
                linewidth=1.0,
                ax=ax,
            )
            ax.axhline(other[col].median(), color="black", ls="--", lw=2.0)
        
        # Plot each El Niño year with a different color
        missing_years = []
        for enso_year in enso_peak_years:
            enso_subset = enso[enso["enso_year"] == enso_year]
            if len(enso_subset) > 0:
                ax.scatter(
                    np.zeros(len(enso_subset)),
                    enso_subset[col],
                    s=150,
                    color=enso_colors[enso_year],
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=5,
                )
            elif enso_year in in_record.get(stat, []):
                missing_years.append(enso_year)

        # No event in these El Niño years; mark with 'x' at bottom, spread
        # side to side. x in data coords, y in axes fraction (0 = bottom).
        spacing = 0.12
        offsets = (np.arange(len(missing_years)) - (len(missing_years) - 1) / 2) * spacing
        for x_off, enso_year in zip(offsets, missing_years):
            ax.scatter(
                x_off,
                0.04,
                marker="x",
                s=200,
                color=enso_colors[enso_year],
                linewidth=2,
                zorder=5,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_title(f"{station_name}, n={len(sub)} years", fontsize=26)
        ax.tick_params(axis="y", labelsize=23)
        ax.grid(True, axis="y", alpha=0.6, linewidth=1.0)

    # Only show y-axis labels on bottom right subplot
    bottom_left = (int(np.ceil(len(stations) / 3)) - 1) * 3
    for i, ax in enumerate(axes[: len(stations)]):
        ax.tick_params(axis="y", labelleft=(i == bottom_left))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=enso_year_labels[year],
               markerfacecolor=enso_colors[year], markersize=11, markeredgecolor="white")
        for year in enso_peak_years
    ]
    legend_elements.append(
        Line2D([0], [0], marker="x", color="w", label="No POT events",
               markeredgecolor="black", markersize=13, markeredgewidth=2)
    )
    
    y_label = y_label or f"{label} POT water level (m)"
    title_label = title_label or f"ENSO-year {label.lower()} of POT events"
    fig.supylabel(f"{y_label}", fontsize=28, x=0.04)
    fig.suptitle(
        f"El Niño years vs. other years — {title_label}",
        fontsize=34,
    )
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        fontsize=22,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(dir_out, fname), dpi=200)
    plt.show()


# ===============================================================================
# Load Data & Calculate Extremes
# ===============================================================================
names = {row["station"]: row["name"] for row in TG}
da = load_tide_gauges(TG, dir_in, dt_start)
if detrend_timeseries_before_pot:
    da = detrend_timeseries(da)

print("Calculating POT extremes...")
_, extreme_ds = compute_return_intervals_forwardstop_map(
    da,
    RPs=RPs,
    r=decluster_window,
    l=30,
    method="emp",
    fitter_method="mle",
    n_boot=200,
    pct_low=pot_pct_low,
    pct_high=pot_pct_high,
    decluster=decluster_method,
    return_extremes=True,
)

events = extremes_to_frame(da, extreme_ds, names)
yearly = station_year(events)
in_record = record_enso_years(da, min_coverage=min_coverage)
event_counts = station_event_counts(events, in_record, names)

# ===============================================================================
# Figures
# ===============================================================================
yearly_stat_figure(
    yearly, in_record, "year_mean", "Mean", "ENSO_Stormyness_WL_YearlyMean_Stations.png"
)
yearly_stat_figure(
    yearly, in_record, "year_max", "Max", "ENSO_Stormyness_WL_YearlyMax_Stations.png"
)
yearly_stat_figure(
    event_counts,
    in_record,
    "event_count",
    "Number",
    "ENSO_Stormyness_EventCount_Stations.png",
    y_label="Number of POT events",
    title_label="ENSO-year POT event counts",
)
