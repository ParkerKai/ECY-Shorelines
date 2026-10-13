"""
Aggregate COSMOS extreme water level and wave event data by county and ENSO year
(March -> February), then compare El Nino years against the all-year distribution.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# ===============================================================================
# User Inputs
# ===============================================================================
extreme_wl_in = (
    r"N:\SEA\Kai\Data\SLR_Atlas\Cosmos_Extremes_WL\Cosmos_Extremes_WL_Events.nc"
)
extreme_wave_in = (
    r"N:\SEA\Kai\Data\SLR_Atlas\Cosmos_Extremes_Wave\Cosmos_Extremes_Wave_Events.nc"
)
county_in = r"N:\SEA\Kai\Data\BarriersToMigration\CountyBounds.shp"

dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures"
os.makedirs(dir_out, exist_ok=True)

county_field = "COUNTY_NM"

# El Nino peak years; each ENSO year runs Mar(year-1) -> Feb(year)
enso_peak_years = [1983, 1998, 2016]
detrend_for_comparison = True

# Stations sit just offshore of the polygons, so snap them to the nearest
# county within this distance (degrees) when they fall outside.
max_snap_deg = 0.05

# ===============================================================================
# Functions
# ===============================================================================


def assign_stations_to_counties(lon, lat, counties, field, max_snap_deg=0.05):
    """Return a Series of county names indexed by station position."""
    pts = gpd.GeoDataFrame(
        {"idx": np.arange(len(lon))},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    poly = counties[[field, "geometry"]].to_crs("EPSG:4326")

    joined = gpd.sjoin(pts, poly, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    names = joined[field]

    missing = names.isna()
    if missing.any() and max_snap_deg > 0:
        # Nearest joins need a projected CRS; use the shapefile's native one.
        proj_crs = counties.crs
        snapped = gpd.sjoin_nearest(
            pts.loc[missing.to_numpy()].to_crs(proj_crs),
            counties[[field, "geometry"]],
            how="left",
            max_distance=max_snap_deg * 364000.0,  # ~degrees -> US survey feet
        )
        snapped = snapped[~snapped.index.duplicated(keep="first")]
        names.loc[snapped.index] = snapped[field]

    return names.reindex(pts.index)


def events_to_frame(ds, var, counties, field, max_snap_deg=0.05):
    """Flatten a (station, event) extremes dataset to a tidy county-tagged frame."""
    values = ds[var].values
    times = ds["time"].values

    county_by_station = assign_stations_to_counties(
        ds["lon"].values, ds["lat"].values, counties, field, max_snap_deg
    )
    county_codes = pd.Categorical(county_by_station.to_numpy())

    # Most (station, event) slots are unused padding; drop them before flattening.
    valid = np.isfinite(values) & ~np.isnat(times)
    valid &= ~county_by_station.isna().to_numpy()[:, None]
    st_idx, _ = np.nonzero(valid)

    df = pd.DataFrame(
        {
            "station": st_idx.astype(np.int32),
            "county": pd.Categorical.from_codes(
                county_codes.codes[st_idx], categories=county_codes.categories
            ),
            "t": times[valid],
            "value": values[valid],
        }
    )
    months = df["t"].dt.month
    # ENSO year runs Mar(y-1) -> Feb(y) and is labelled by its peak year.
    df["enso_year"] = (df["t"].dt.year + (months >= 3).astype(int)).astype(np.int16)
    return df


def county_station_year(df):
    """Per county / ENSO year / station mean of the event values."""
    return (
        df.groupby(["county", "enso_year", "station"], observed=True)["value"]
        .mean()
        .rename("station_mean")
        .reset_index()
    )


def county_yearly(df, per_station):
    """County/ENSO-year aggregate: mean plus across-station and within-year spread."""
    across_station = per_station.groupby(["county", "enso_year"], observed=True).agg(
        mean=("station_mean", "mean"),
        std_stations=("station_mean", "std"),
        min_station=("station_mean", "min"),
        max_station=("station_mean", "max"),
        n_stations=("station_mean", "size"),
    )

    across_samples = df.groupby(["county", "enso_year"], observed=True).agg(
        std_samples=("value", "std"),
        n_samples=("value", "size"),
    )

    out = across_station.join(across_samples)
    out.loc[out["n_stations"] < 2, "std_stations"] = np.nan
    out.loc[out["n_samples"] < 2, "std_samples"] = np.nan
    return out.sort_index()


def detrend_by_year(per_station):
    """Remove each county's linear trend in ENSO year from the station means."""
    out = per_station.copy()

    def _fit(g):
        annual = g.groupby("enso_year")["station_mean"].mean()
        x = annual.index.to_numpy(dtype=float)
        y = annual.to_numpy(dtype=float)
        if len(x) < 3 or x.std() == 0:
            return g["station_mean"]
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        return g["station_mean"] - slope * (
            g["enso_year"].to_numpy(dtype=float) - x.mean()
        )

    out["station_mean"] = (
        out.groupby("county", observed=True, group_keys=False)
        .apply(_fit, include_groups=False)
        .reindex(out.index)
    )
    return out


def enso_violin_figure(per_station, label, units, fname):
    """Per county: all-year station distribution followed by each El Nino year."""
    if detrend_for_comparison:
        per_station = detrend_by_year(per_station)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_labels = ["All years"] + [f"{y - 1}\u2013{y}" for y in enso_peak_years]
    palette = dict(
        zip(
            group_labels,
            ["0.75"] + [colors[k % len(colors)] for k in range(len(enso_peak_years))],
        )
    )

    all_years = per_station.assign(group=group_labels[0])
    events = per_station[per_station["enso_year"].isin(enso_peak_years)].assign(
        group=lambda d: d["enso_year"].map(dict(zip(enso_peak_years, group_labels[1:])))
    )
    plot_df = pd.concat([all_years, events], ignore_index=True)
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=group_labels)

    counties = sorted(per_station["county"].unique())
    ncol = 3
    nrow = int(np.ceil(len(counties) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.6 * ncol, 3.6 * nrow), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, cname in zip(axes, counties):
        sub = plot_df[plot_df["county"] == cname]
        sns.violinplot(
            data=sub,
            x="group",
            y="station_mean",
            hue="group",
            order=group_labels,
            hue_order=group_labels,
            palette=palette,
            legend=False,
            cut=0,
            width=2,
            density_norm="width",
            inner="quartile",
            linewidth=1.0,
            ax=ax,
        )
        ax.set_xlim(-0.6, len(group_labels) - 0.4)

        baseline = sub.loc[sub["group"] == group_labels[0], "station_mean"]
        ax.axhline(float(baseline.median()), color="0.4", ls="--", lw=1.0, zorder=0)

        # Empty ENSO years mean no POT exceedance, not missing data.
        for pos, g in enumerate(group_labels):
            if sub.loc[sub["group"] == g, "station_mean"].notna().sum() == 0:
                ax.annotate(
                    "n=0",
                    (pos, 0.5),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.4",
                    rotation=90,
                )

        ax.set_title(
            f"{str(cname).title()} (n={sub['station'].nunique()} stations)",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8, labelrotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-1, len(group_labels) + 1)

    for ax in axes[len(counties) :]:
        ax.set_visible(False)

    detrend_note = " (detrended)" if detrend_for_comparison else ""
    fig.supylabel(f"ENSO-year mean {label} ({units}){detrend_note}")
    fig.suptitle(
        f"El Ni\u00f1o years vs. all years \u2014 {label} by county "
        "(ENSO year = Mar\u2192Feb; violins span stations)"
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.savefig(os.path.join(dir_out, fname), dpi=200)
    plt.show()
    return fig


# ===============================================================================
# Load Data
# ===============================================================================
county = gpd.read_file(county_in)

ext_wl = xr.open_dataset(extreme_wl_in)[["waterlevel", "lon", "lat"]]
ext_wave = xr.open_dataset(extreme_wave_in)[["Hs", "lon", "lat"]]

# ===============================================================================
# County / ENSO-Year Aggregation
# ===============================================================================
print("Aggregating water level events...")
wl_events = events_to_frame(ext_wl, "waterlevel", county, county_field, max_snap_deg)
wl_station_year = county_station_year(wl_events)
wl_yearly = county_yearly(wl_events, wl_station_year)

print("Aggregating wave events...")
wave_events = events_to_frame(ext_wave, "Hs", county, county_field, max_snap_deg)
wave_station_year = county_station_year(wave_events)
wave_yearly = county_yearly(wave_events, wave_station_year)

# ===============================================================================
# Figures
# ===============================================================================
enso_violin_figure(
    wl_station_year,
    "extreme water level",
    "m",
    "ENSO_Stormyness_WL_Violin_Counties.png",
)
enso_violin_figure(
    wave_station_year, "extreme Hs", "m", "ENSO_Stormyness_Hs_Violin_Counties.png"
)
