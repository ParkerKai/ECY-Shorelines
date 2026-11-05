"""
Calculate March-February annual maximum water levels and compare selected
El Nino years against the full distribution at each tide gauge.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D

# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\Hourly"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures"
os.makedirs(dir_out, exist_ok=True)

# Select tide gauges.
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

# Record trim (None or e.g. pd.Timestamp("1992-01-01")).
dt_start = None
min_coverage = 0.5
detrend_before_annual_max = True

# ENSO years are labelled by the February year: Mar 1982-Feb 1983, etc.
enso_peak_years = [1983, 1998, 2016]
enso_labels = {
    1983: "1982-1983",
    1998: "1997-1998",
    2016: "2015-2016",
}
enso_colors = {
    1983: "#d55e00",
    1998: "#0072b2",
    2016: "#009e73",
}


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


def annual_maxima(da, min_coverage=0.5):
    """Return annual maxima and differences from each station's mean annual max."""
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    enso_year = times.year + (times.month >= 3).astype(int)
    expected_hours = 365 * 24
    rows = []

    for i, stat in enumerate(da["station"].values):
        values = pd.Series(da.values[i], index=enso_year)
        valid = values.notna()
        coverage = valid.groupby(level=0).sum()
        valid_years = coverage.index[coverage >= min_coverage * expected_hours]
        maxima = values[valid].groupby(level=0).max()
        for year in valid_years:
            rows.append(
                {
                    "station": stat,
                    "name": next(row["name"] for row in TG if row["station"] == stat),
                    "enso_year": int(year),
                    "annual_max": maxima.loc[year],
                }
            )

    result = pd.DataFrame(rows)
    result["max_minus_mean"] = result["annual_max"] - result.groupby("station")[
        "annual_max"
    ].transform("mean")
    return result


def enso_legend_handles():
    """Create consistent legend entries for the selected ENSO years."""
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=enso_colors[year],
            markeredgecolor="white",
            markersize=11,
            label=enso_labels[year],
        )
        for year in enso_peak_years
    ]


def _grid(n, ncol=3, w=6.4, h=4.4):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(w * ncol, h * nrow), sharey=True, squeeze=False
    )
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def annual_max_figure(yearly, fname):
    """Plot all annual maxima as violins with selected ENSO years as dots."""
    stations = list(dict.fromkeys(yearly["station"]))
    fig, axes = _grid(len(stations))

    for ax, stat in zip(axes, stations):
        sub = yearly[yearly["station"] == stat]
        if len(sub) > 1:
            sns.violinplot(
                y=sub["annual_max"],
                x=np.zeros(len(sub)),
                color="0.75",
                cut=0,
                inner="quartile",
                linewidth=1.0,
                ax=ax,
            )

        enso = sub[sub["enso_year"].isin(enso_peak_years)]
        for year in enso_peak_years:
            selected = enso[enso["enso_year"] == year]
            ax.scatter(
                np.zeros(len(selected)),
                selected["annual_max"],
                color=enso_colors[year],
                edgecolor="white",
                zorder=3,
                s=260,
            )

        ax.set_title(f"{sub['name'].iloc[0]}, n={len(sub)} years", fontsize=26)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=23)
        ax.grid(axis="y", alpha=0.6, linewidth=1.0)

    bottom_left = (int(np.ceil(len(stations) / 3)) - 1) * 3
    for i, ax in enumerate(axes[: len(stations)]):
        ax.tick_params(axis="y", labelleft=(i == bottom_left))
    fig.supylabel(
        f"March-February annual maximum water level (m)",
        fontsize=28,
        x=0.04,
    )
    fig.suptitle(
        "El Nino years vs. all years - March-February annual maxima", fontsize=34
    )
    axes[0].legend(
        handles=enso_legend_handles(),
        loc="upper right",
        frameon=False,
        fontsize=19,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(dir_out, fname), dpi=200)
    plt.show()


def annual_max_difference_figure(yearly, fname):
    """Boxplot annual maximum minus mean annual maximum, with ENSO dots."""
    stations = list(dict.fromkeys(yearly["station"]))
    fig, axes = _grid(len(stations))

    for ax, stat in zip(axes, stations):
        sub = yearly[yearly["station"] == stat]
        ax.boxplot(
            sub["max_minus_mean"],
            positions=[0],
            widths=0.5,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "0.75", "edgecolor": "0.25"},
            medianprops={"color": "0.25", "linewidth": 1.5},
            whiskerprops={"color": "0.25"},
            capprops={"color": "0.25"},
        )

        enso = sub[sub["enso_year"].isin(enso_peak_years)]
        for year in enso_peak_years:
            selected = enso[enso["enso_year"] == year]
            ax.scatter(
                np.zeros(len(selected)),
                selected["max_minus_mean"],
                color=enso_colors[year],
                edgecolor="white",
                zorder=3,
                s=260,
            )

        ax.set_title(f"{sub['name'].iloc[0]}, n={len(sub)} years", fontsize=26)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=23)
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=0)
        ax.grid(axis="y", alpha=0.6, linewidth=1.0)

    bottom_left = (int(np.ceil(len(stations) / 3)) - 1) * 3
    for i, ax in enumerate(axes[: len(stations)]):
        ax.tick_params(axis="y", labelleft=(i == bottom_left))
    fig.supylabel(
        f"Annual maximum difference from mean (m)",
        fontsize=28,
        x=0.04,
    )
    fig.suptitle(
        "El Nino years vs. all years - annual maximum difference from mean",
        fontsize=34,
    )
    axes[0].legend(
        handles=enso_legend_handles(),
        loc="upper right",
        frameon=False,
        fontsize=19,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(dir_out, fname), dpi=200)
    plt.show()


# ===============================================================================
# Load Data, Calculate Maxima, and Plot
# ===============================================================================
da = load_tide_gauges(TG, dir_in, dt_start)
if detrend_before_annual_max:
    da = detrend_timeseries(da)
yearly_max = annual_maxima(da, min_coverage=min_coverage)
annual_max_figure(yearly_max, "ENSO_AnnualMax_WL_Stations.png")
annual_max_difference_figure(
    yearly_max, "ENSO_AnnualMax_DifferenceFromMean_Stations.png"
)
