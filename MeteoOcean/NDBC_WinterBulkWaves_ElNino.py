"""Compare winter (DJF) bulk wave conditions at an NDBC buoy with El Nino winters.

Reads the NetCDF written by DownloadNDBC_BulkWaves.py, plots side-by-side wave roses
(Hs vs peak direction) for all winters and El Nino winters, then computes the mean wave
power and power-weighted mean wave direction for each winter and shows the distribution
across winters as violins with the El Nino winters as dots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from xoa.plot import plot_rose

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46088"
DATA_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\BulkWaves")
OUT_DIR = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures\Waves"
)
WINTER_MONTHS = [12, 1, 2]
# winters are labelled by the year of their December, so 1982 means Dec 1982 - Feb 1983
EL_NINO_WINTERS = [1982, 1997, 2015]
EL_NINO_COLORS = {1982: "#d55e00", 1997: "#0072b2", 2015: "#009e73"}
MIN_COVERAGE = 0.5  # fraction of DJF hours with valid Hs, Tp and Dp required per winter
HS_EDGES = [0, 1, 2, 3, 4, 6, 8]
NSECTOR = 16
RHO, G = 1025.0, 9.81
TE_FACTOR = 0.9  # energy period estimated as Te = 0.9 * Tp (no spectrum available)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def load_station(data_dir, station_id):
    files = sorted(data_dir.glob(f"ndbc{station_id}_bulk_waves_*.nc"))
    if not files:
        raise SystemExit(f"No bulk wave file for station {station_id} in {data_dir}")
    ds = xr.open_dataset(files[-1]).load()
    # keep one record per hour so periods with 30/10-min sampling are not over-weighted
    _, idx = np.unique(ds.time.dt.floor("1h").values, return_index=True)
    return ds.isel(time=idx)


def wave_power(ds):
    """Deep-water wave energy flux per metre of crest [kW m-1]."""
    te = TE_FACTOR * ds.Tp
    return (RHO * G**2 / (64 * np.pi) * ds.Hs**2 * te / 1000).assign_attrs(
        long_name="deep-water wave power", units="kW m-1"
    )


def winter_stats(ds):
    """Mean wave power and power-weighted mean direction for each DJF winter."""
    winter_year = ds.time.dt.year.where(ds.time.dt.month == 12, ds.time.dt.year - 1)
    df = pd.DataFrame(
        {
            "winter": winter_year.values,
            "month": ds.time.dt.month.values,
            "P": wave_power(ds).values,
            "Dp": ds.Dp.values,
        },
        index=ds.time.values,
    )
    df = df[df["month"].isin(WINTER_MONTHS)].dropna(subset=["P", "Dp"])

    rows = []
    for yr, sub in df.groupby("winter"):
        start = pd.Timestamp(yr, 12, 1)
        n_hours = (pd.Timestamp(yr + 1, 3, 1) - start) / pd.Timedelta("1h")
        coverage = len(sub) / n_hours
        if coverage < MIN_COVERAGE:
            continue
        th = np.deg2rad(sub["Dp"])
        mwd = np.rad2deg(
            np.arctan2((sub["P"] * np.sin(th)).sum(), (sub["P"] * np.cos(th)).sum())
        )
        rows.append(
            {
                "winter": int(yr),
                "coverage": coverage,
                "mean_power": sub["P"].mean(),
                "mean_direction": mwd % 360,
            }
        )
    return pd.DataFrame(rows)


def plot_wave_roses(periods, station_id):
    fig = plt.figure(figsize=(10.5, 4.5))
    fig.subplots_adjust(right=0.75, wspace=0.4)  # room for the legend
    roses = []
    for i, (label, sub) in enumerate(periods.items()):
        valid = (sub.Hs.notnull() & sub.Dp.notnull()).values
        rose = plot_rose(
            sub.Hs[valid],
            sub.Dp[valid],
            edges=HS_EDGES,
            nsector=NSECTOR,
            fig=fig,
            subplot=121 + i,
            legend=i == 1,
        )
        rose.ax.set_title(label, wrap=True)
        roses.append(rose)
    rmax = max(r.ax.get_rmax() for r in roses)
    for r in roses:
        r.ax.set_rmax(rmax)
    fig.suptitle(
        f"NDBC {station_id} wave roses: Hs [m] vs peak direction (coming from)"
    )
    return fig


def plot_violins(stats, found, station_id):
    panels = [
        ("mean_power", "Mean DJF wave power [kW m⁻¹]", "Wave power"),
        (
            "mean_direction",
            "Power-weighted mean DJF direction [° true N, coming from]",
            "Wave direction",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    for ax, (col, ylabel, title) in zip(axes, panels):
        vals = stats[col].to_numpy()
        parts = ax.violinplot(vals, positions=[0], widths=0.8, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("0.75")
            body.set_edgecolor("0.25")
            body.set_alpha(1)
        for q, ls in zip(np.percentile(vals, [25, 50, 75]), [":", "--", ":"]):
            ax.hlines(q, -0.25, 0.25, color="0.25", linestyle=ls, linewidth=1)
        for yr in found:
            sel = stats[stats["winter"] == yr]
            ax.scatter(
                np.zeros(len(sel)),
                sel[col],
                color=EL_NINO_COLORS.get(yr, "tab:red"),
                edgecolor="white",
                zorder=3,
                s=120,
            )
        ax.set_title(f"{title}, n={len(stats)} winters")
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=EL_NINO_COLORS.get(yr, "tab:red"),
            markeredgecolor="white",
            markersize=9,
            label=f"{yr}-{yr + 1}",
        )
        for yr in found
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle(f"NDBC {station_id} El Nino winters vs. all winters (DJF)")
    fig.tight_layout()
    return fig


def main():
    ds = load_station(DATA_DIR, STATION_ID)
    winter_year = ds.time.dt.year.where(ds.time.dt.month == 12, ds.time.dt.year - 1)
    is_winter = ds.time.dt.month.isin(WINTER_MONTHS)
    is_nino = is_winter & winter_year.isin(EL_NINO_WINTERS)
    print(f"Station {STATION_ID}: {ds.time.values[0]} to {ds.time.values[-1]}")

    stats = winter_stats(ds)
    found = [yr for yr in EL_NINO_WINTERS if yr in set(stats["winter"])]
    missing = sorted(set(EL_NINO_WINTERS) - set(found))
    print(f"{len(stats)} winters with >= {MIN_COVERAGE:.0%} coverage")
    print(f"El Nino winters with data: {found}; missing/insufficient: {missing}")
    if not found:
        raise SystemExit("No El Nino winter data at this station.")
    print(stats.round(2).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.assign(el_nino=stats["winter"].isin(found)).to_csv(
        OUT_DIR / f"ndbc{STATION_ID}_winter_power_direction.csv", index=False
    )

    periods = {
        "All winters (DJF)": ds.isel(time=is_winter.values),
        "El Nino winters (DJF "
        + ", ".join(f"{y}-{y + 1}" for y in found)
        + ")": ds.isel(time=is_nino.values),
    }
    fig = plot_wave_roses(periods, STATION_ID)
    fig.savefig(
        OUT_DIR / f"ndbc{STATION_ID}_bulk_waverose_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig2 = plot_violins(stats, found, STATION_ID)
    fig2.savefig(
        OUT_DIR / f"ndbc{STATION_ID}_violin_power_direction_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
