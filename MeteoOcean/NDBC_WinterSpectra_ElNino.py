"""Compare average winter (DJF) directional wave spectra at an NDBC buoy with El Nino winters.

Reads the NetCDF written by DownloadNDBC_DirectionalSpectra.py, computes the mean
frequency and directional spectra for all winters and for El Nino winters, saves them,
and plots side-by-side wave roses (Hs vs peak direction) with xoa plus the mean
directional spectra. Also compares with ERA5 modelled wave spectra.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from xoa.plot import plot_rose

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46041"
DATA_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\FullSpectral")
ERA5_DIR = Path(r"E:\ERA5_Waves\PNW_Era5Spectral")
OUT_DIR = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures\Waves"
)
WINTER_MONTHS = [12, 1, 2]
# winters are labelled by the year of their December, so 1982 means Dec 1982 - Feb 1983
EL_NINO_WINTERS = [1982, 1997, 2015]
EL_NINO_COLORS = {1982: "#d55e00", 1997: "#0072b2", 2015: "#009e73"}
EL_NINO_VIOLIN = [2015]  # Only plot 2015 in violin (insufficient data for 82, 97)
HS_EDGES = [0, 1, 2, 3, 4, 6, 8]
NSECTOR = 16
NDIR = 72
RHO, G = 1025.0, 9.81
TE_FACTOR = 0.9  # energy period estimated as Te = 0.9 * Tp

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def load_station(data_dir, station_id):
    files = sorted(data_dir.glob(f"ndbc{station_id}_directional_spectra_*.nc"))
    if not files:
        raise SystemExit(f"No spectra file for station {station_id} in {data_dir}")
    ds = xr.open_dataset(files[-1]).load()
    # keep one record per hour so periods with 30/10-min sampling are not over-weighted
    _, idx = np.unique(ds.time.dt.floor("1h").values, return_index=True)
    return ds.isel(time=idx)


def era5_file(era5_dir, year, month):
    return era5_dir / f"ERA5_SpectralWaves_WC_{year}_{month}.nc"


def load_era5_station(era5_dir, lat, lon, winters):
    """E(f, theta) [m2 Hz-1 rad-1] at the ERA5 grid point nearest (lat, lon) for DJF of `winters`.

    Each monthly file holds the whole regional grid (~4.5 GB in memory), so the point is
    selected lazily before loading and only that column is read from disk.
    """
    months = [(y, 12) for y in winters] + [
        (y + 1, m) for y in winters for m in (1, 2)
    ]
    points = []
    for year, month in sorted(months):
        f = era5_file(era5_dir, year, month)
        if not f.exists():
            print(f"  Warning: missing ERA5 file {f.name}")
            continue
        with xr.open_dataset(f) as ds:
            points.append(
                ds.d2fd.sel(latitude=lat, longitude=(lon + 180) % 360 - 180, method="nearest")
                .drop_vars(["number", "expver"], errors="ignore")
                .load()
            )
    if not points:
        raise SystemExit(f"No ERA5 files found for winters {winters} in {era5_dir}")
    d2fd = xr.concat(points, dim="valid_time").rename(
        valid_time="time", frequencyNumber="frequency", directionNumber="direction"
    )

    # d2fd is log10(E); bins with no energy are stored as missing.
    # Directions are bin numbers (7.5 deg + 15 deg steps, oceanographic "going to");
    # rotate by 180 deg so they match the NDBC "coming from" convention.
    # Frequencies are bin numbers on the WAM grid f_n = 0.03453 * 1.1**(n-1) Hz.
    efth = (10.0 ** d2fd).fillna(0.0)
    efth = efth.assign_coords(
        frequency=0.03453 * 1.1 ** (d2fd.frequency.values - 1),
        direction=(7.5 + 15.0 * (d2fd.direction.values - 1) + 180.0) % 360.0,
    ).sortby("direction")
    return efth.transpose("time", "frequency", "direction").assign_attrs(
        long_name="ERA5 directional wave spectrum", units="m2 Hz-1 rad-1"
    )


def era5_winter_means(era5_dir, lat, lon, cache_file):
    """DJF-mean ERA5 E(f, theta) for every complete winter in the archive, one per winter.

    Reading the station point takes a few seconds per monthly file, so per-winter means are
    cached to `cache_file`; later runs only read winters that are not in the cache yet.
    """
    available = sorted(
        y
        for y in range(1900, 2200)
        if all(
            era5_file(era5_dir, yy, m).exists()
            for yy, m in [(y, 12), (y + 1, 1), (y + 1, 2)]
        )
    )
    cached = None
    if cache_file.exists():
        cached = xr.load_dataset(cache_file)
        if not np.isclose([cached.attrs["lat"], cached.attrs["lon"]], [lat, lon]).all():
            print(f"  ERA5 cache {cache_file.name} is for another location; rebuilding")
            cached = None
    todo = [y for y in available if cached is None or y not in cached.winter.values]

    means = []
    for i, year in enumerate(todo, 1):
        print(f"  ERA5 winter {year}-{year + 1} ({i}/{len(todo)})", flush=True)
        efth = load_era5_station(era5_dir, lat, lon, [year])
        efth = efth.isel(time=efth.time.dt.month.isin(WINTER_MONTHS).values)
        means.append(
            xr.Dataset(
                {
                    "efth": efth.mean("time"),
                    "count": xr.DataArray(efth.time.size),
                }
            ).expand_dims(winter=[year])
        )
    if means:
        new = xr.concat(means, dim="winter")
        out = new if cached is None else xr.concat([cached, new], dim="winter").sortby("winter")
        out = out.assign_attrs(
            lat=lat,
            lon=lon,
            description="ERA5 DJF-mean directional wave spectra at the grid point "
            "nearest (lat, lon); winters labelled by the year of December",
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(cache_file)
        return out
    return cached


def fourier_coefficients(ds):
    """S(f) times the directional Fourier coefficients a1, b1, a2, b2 (linear in energy)."""
    a1, a2 = np.deg2rad(ds.alpha1), np.deg2rad(ds.alpha2)
    coefs = xr.Dataset(
        {
            "S": ds.spectral_density,
            "Sa1": ds.spectral_density * ds.r1 * np.cos(a1),
            "Sb1": ds.spectral_density * ds.r1 * np.sin(a1),
            "Sa2": ds.spectral_density * ds.r2 * np.cos(2 * a2),
            "Sb2": ds.spectral_density * ds.r2 * np.sin(2 * a2),
        }
    )
    # records use the 47- or 64-band NDBC grids; interpolate onto the union grid
    return coefs.interpolate_na("frequency")


def bulk_parameters(ds):
    S = ds.spectral_density.interpolate_na("frequency").fillna(0)
    hs = 4 * np.sqrt(S.integrate("frequency"))
    ipeak = ds.spectral_density.fillna(-1).argmax("frequency")
    dp = ds.alpha1.isel(frequency=ipeak)
    tp = 1 / ds.frequency.isel(frequency=ipeak)
    valid = hs > 0
    return xr.Dataset(
        {
            "hs": hs.where(valid).assign_attrs(
                long_name="significant wave height", units="m"
            ),
            "tp": tp.where(valid).assign_attrs(long_name="peak period", units="s"),
            "dp": dp.where(valid).assign_attrs(
                long_name="peak wave direction (alpha1 at peak frequency), coming from",
                units="degree",
            ),
        }
    ).drop_vars("frequency")


def bulk_parameters_era5(efth):
    """Hs, Tp and peak direction (mean direction at the peak frequency, as for NDBC alpha1)."""
    dth = np.deg2rad(360.0 / efth.direction.size)
    S = efth.sum("direction") * dth
    hs = 4 * np.sqrt(S.integrate("frequency"))
    ipeak = S.argmax("frequency")
    tp = 1 / efth.frequency.isel(frequency=ipeak)
    e_peak = efth.isel(frequency=ipeak)
    th = np.deg2rad(efth.direction)
    dp = np.rad2deg(
        np.arctan2((e_peak * np.sin(th)).sum("direction"), (e_peak * np.cos(th)).sum("direction"))
    ) % 360
    valid = hs > 0
    return xr.Dataset(
        {
            "hs": hs.where(valid).assign_attrs(
                long_name="significant wave height", units="m"
            ),
            "tp": tp.where(valid).assign_attrs(long_name="peak period", units="s"),
            "dp": dp.where(valid).assign_attrs(
                long_name="mean direction at peak frequency, coming from",
                units="degree",
            ),
        }
    ).drop_vars(["frequency", "latitude", "longitude"], errors="ignore")


def mean_spectra(ds, label):
    coefs = fourier_coefficients(ds)
    # only average records with a complete directional description
    complete = coefs.to_array().notnull().all("variable")
    mean = coefs.where(complete).mean("time")
    count = complete.sum("time")

    theta = xr.DataArray(
        np.arange(0, 360, 360 / NDIR),
        dims="direction",
        attrs={"long_name": "direction waves come from", "units": "degree"},
    )
    th = np.deg2rad(theta)
    efth = (
        mean.S * 0.5
        + mean.Sa1 * np.cos(th)
        + mean.Sb1 * np.sin(th)
        + mean.Sa2 * np.cos(2 * th)
        + mean.Sb2 * np.sin(2 * th)
    ) / np.pi
    efth = efth.clip(min=0).assign_coords(direction=theta)
    out = xr.Dataset(
        {
            "spectral_density": mean.S.assign_attrs(
                long_name=f"mean spectral wave energy density, {label}", units="m2 Hz-1"
            ),
            "directional_spectrum": efth.assign_attrs(
                long_name=f"mean directional wave spectrum, {label}",
                units="m2 Hz-1 rad-1",
            ),
            "count": count.assign_attrs(long_name="number of records averaged"),
        }
    )
    return out.transpose("frequency", "direction")


def mean_spectra_era5(efth, label):
    return era5_spectrum(efth.mean("time"), label, efth.time.size)


def era5_spectrum(mean, label, count):
    S = mean.sum("direction") * np.deg2rad(360.0 / mean.direction.size)
    out = xr.Dataset(
        {
            "spectral_density": S.assign_attrs(
                long_name=f"mean spectral wave energy density, {label}", units="m2 Hz-1"
            ),
            "directional_spectrum": mean.assign_attrs(
                long_name=f"mean directional wave spectrum, {label}",
                units="m2 Hz-1 rad-1",
            ),
            "count": xr.DataArray(count).assign_attrs(
                long_name="number of records averaged"
            ),
        }
    )
    return out.transpose("frequency", "direction")


def plot_directional_spectra(spectra, titles, station_id):
    vmax = max(float(s.directional_spectrum.max()) for s in spectra)
    fig, axes = plt.subplots(
        1, len(spectra), figsize=(5 * len(spectra), 4.8), subplot_kw={"projection": "polar"}
    )
    for ax, sp, title in zip(axes, spectra, titles):
        e = sp.directional_spectrum
        
        # Get frequency dimension name (could be 'frequency' or 'frequencyNumber')
        freq_dims = [d for d in e.dims if 'freq' in d.lower()]
        freq_dim = freq_dims[0] if freq_dims else 'frequency'
        
        th = np.deg2rad(np.append(e.direction.values, 360.0))
        data = np.concatenate([e.values, e.values[:, :1]], axis=1)
        pc = ax.pcolormesh(
            th,
            1 / e[freq_dim].values,
            data,
            shading="nearest",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 20)
        ax.set_rlabel_position(22.5)
        ax.set_title(title, pad=20, wrap=True)

    fig.colorbar(pc, ax=axes, shrink=0.8, label="E(f, θ) [m² Hz⁻¹ rad⁻¹]")
    fig.suptitle(
        f"Station {station_id} mean DJF directional spectra (radius = period [s])"
    )
    return fig


def wave_power(hs, tp):
    """Deep-water wave energy flux per metre of crest [kW m-1]."""
    te = TE_FACTOR * tp
    return (RHO * G**2 / (64 * np.pi) * hs**2 * te / 1000)


def winter_bulk_stats(ds, station_id):
    """Mean wave power and power-weighted mean direction for each DJF winter from spectral data."""
    winter_year = ds.time.dt.year.where(ds.time.dt.month == 12, ds.time.dt.year - 1)
    
    bulk = bulk_parameters(ds)
    df = pd.DataFrame(
        {
            "winter": winter_year.values,
            "month": ds.time.dt.month.values,
            "Hs": bulk.hs.values,
            "Tp": bulk.tp.values,
            "Dp": bulk.dp.values,
        },
        index=ds.time.values,
    )
    df["P"] = wave_power(df["Hs"], df["Tp"])
    df = df[df["month"].isin(WINTER_MONTHS)].dropna(subset=["P", "Dp"])

    rows = []
    for yr, sub in df.groupby("winter"):
        th = np.deg2rad(sub["Dp"])
        mwd = np.rad2deg(
            np.arctan2((sub["P"] * np.sin(th)).sum(), (sub["P"] * np.cos(th)).sum())
        )
        rows.append(
            {
                "winter": int(yr),
                "mean_power": sub["P"].mean(),
                "mean_direction": mwd % 360,
            }
        )
    return pd.DataFrame(rows)


def era5_winter_bulk_stats(efth_dataset):
    """Compute mean wave power and direction for each winter from ERA5 spectral data.
    
    Parameters
    ----------
    efth_dataset : xr.Dataset
        ERA5 winter means dataset with 'efth' (directional spectrum) and 'winter' dimension
    
    Returns
    -------
    pd.DataFrame with columns: winter, mean_power, mean_direction
    """
    rows = []
    for year in efth_dataset.winter.values:
        efth = efth_dataset.sel(winter=year).efth
        bulk = bulk_parameters_era5(efth.expand_dims("time"))
        
        # Calculate wave power from Hs and Tp - extract scalar values
        hs_val = float(bulk.hs.values.squeeze())
        tp_val = float(bulk.tp.values.squeeze())
        if hs_val > 0 and tp_val > 0:
            p = wave_power(hs_val, tp_val)
            dp_val = float(bulk.dp.values.squeeze())
            rows.append({
                "winter": int(year),
                "mean_power": p,
                "mean_direction": dp_val,
            })
    return pd.DataFrame(rows)


def plot_violins(stats_all_era5, stats_era5_years, stats_nino_obs, year_nino, station_id):
    """Plot violin distributions from ERA5 with all model years and observed 2015 as markers.
    
    Parameters
    ----------
    stats_all_era5 : pd.DataFrame
        ERA5 bulk stats for all winters (violin distribution)
    stats_era5_years : pd.DataFrame
        ERA5 bulk stats for all El Nino years (for markers)
    stats_nino_obs : pd.DataFrame
        Observed NDBC bulk stats for El Nino year (2015 only)
    year_nino : int
        Observed El Nino year (2015) for labeling triangle
    station_id : str
        Station identifier
    """
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
        # ERA5 all winters as violin
        vals = stats_all_era5[col].to_numpy()
        parts = ax.violinplot(vals, positions=[0], widths=0.8, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("0.75")
            body.set_edgecolor("0.25")
            body.set_alpha(1)
        for q, ls in zip(np.percentile(vals, [25, 50, 75]), [":", "--", ":"]):
            ax.hlines(q, -0.25, 0.25, color="0.25", linestyle=ls, linewidth=1)
        
        # Add ERA5 modeled El Nino data as circles (all years)
        for el_nino_year in stats_era5_years["winter"].unique():
            sel_era5 = stats_era5_years[stats_era5_years["winter"] == el_nino_year]
            if len(sel_era5) > 0:
                ax.scatter(
                    np.zeros(len(sel_era5)),
                    sel_era5[col],
                    marker="o",
                    color=EL_NINO_COLORS.get(el_nino_year, "tab:red"),
                    edgecolor="white",
                    zorder=3,
                    s=150,
                    linewidths=2,
                )
        
        # Add observed NDBC El Nino data as triangle (only 2015)
        sel_obs = stats_nino_obs[stats_nino_obs["winter"] == year_nino]
        if len(sel_obs) > 0:
            ax.scatter(
                np.zeros(len(sel_obs)),
                sel_obs[col],
                marker="^",
                color=EL_NINO_COLORS.get(year_nino, "tab:red"),
                edgecolor="white",
                zorder=3,
                s=200,
            )
        
        ax.set_title(f"{title}, n={len(stats_all_era5)} winters")
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
    
    # Add legend with markers for all ERA5 years and observed 2015
    handles = []
    for el_nino_year in sorted(EL_NINO_WINTERS):
        if el_nino_year in stats_era5_years["winter"].values:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=EL_NINO_COLORS.get(el_nino_year, "tab:red"),
                    markeredgecolor="white",
                    markersize=8,
                    linewidth=2,
                    label=f"ERA5 {el_nino_year}-{el_nino_year + 1}",
                )
            )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=EL_NINO_COLORS.get(year_nino, "tab:red"),
            markeredgecolor="white",
            markersize=10,
            label=f"Observed {year_nino}-{year_nino + 1}",
        )
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    fig.suptitle(f"NDBC {station_id} El Nino winters (ERA5 distribution)")
    fig.tight_layout()
    return fig


def plot_frequency_spectra(lines, station_id):
    """`lines` is a list of (spectrum, label, color, linestyle)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for sp, label, color, ls in lines:
        # Only plot model data (solid lines), skip buoy data (dotted lines)
        if ls == ":":
            continue
        kw = dict(color=color, linestyle=ls, linewidth=2, label=label)
        ax1.plot(1 / sp.frequency, sp.spectral_density, **kw)
        d = sp.directional_spectrum.fillna(0).integrate("frequency")
        ax2.plot(
            np.append(d.direction.values, d.direction.values[0] + 360),
            np.append(d.values, d.values[0]),
            **kw,
        )
    ax1.set_xlabel("Wave period [s]")
    ax1.set_ylabel("S(f) [m² Hz⁻¹]")
    ax1.set_xlim(0, 25)
    ax1.set_title("Frequency spectrum")
    ax2.set_xlabel("Direction waves come from [° true N]")
    ax2.set_ylabel("D(θ) = ∫E(f, θ) df [m² rad⁻¹]")
    ax2.set_xlim(0, 360)
    ax2.set_xticks(range(0, 361, 45))
    ax2.set_title("Directional spectrum")
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Station {station_id} mean DJF energy density spectra (ERA5 model)"
    )
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.legend(
        *ax1.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.8, 0.4)
    )
    return fig


######################### Main #####################
def main():
    # Load NDBC data
    ndbc_ds = load_station(DATA_DIR, STATION_ID)
    winter_year = ndbc_ds.time.dt.year.where(ndbc_ds.time.dt.month == 12, ndbc_ds.time.dt.year - 1)
    is_winter = ndbc_ds.time.dt.month.isin(WINTER_MONTHS)
    is_nino = is_winter & winter_year.isin(EL_NINO_WINTERS)

    found = sorted({int(y) for y in winter_year.where(is_nino, drop=True).values})
    missing = sorted(set(EL_NINO_WINTERS) - set(found))
    print(f"\nNDBC Station {STATION_ID}: {ndbc_ds.time.values[0]} to {ndbc_ds.time.values[-1]}")
    print(f"El Nino winters with data: {found}; missing: {missing}")
    if not found:
        raise SystemExit("No El Nino winter data at this station.")

    # Load ERA5 at the buoy location, only for the El Nino winters that the buoy also observed
    print(f"\nLoading ERA5 data for station {STATION_ID}...")
    era5_efth = load_era5_station(
        ERA5_DIR,
        float(ndbc_ds.attrs["latitude"]),
        float(ndbc_ds.attrs["longitude"]),
        found,
    )
    era5_efth = era5_efth.isel(time=era5_efth.time.dt.month.isin(WINTER_MONTHS).values)
    print(
        f"  ERA5 grid point: {float(era5_efth.latitude):.2f}N, "
        f"{float(era5_efth.longitude):.2f}E"
    )
    nino_label = "El Nino winters (DJF " + ", ".join(f"{y}-{y + 1}" for y in found) + ")"

    ndbc_periods = {
        "NDBC All winters (DJF)": ndbc_ds.isel(time=is_winter.values),
        f"NDBC {nino_label}": ndbc_ds.isel(time=is_nino.values),
    }

    # Compute mean spectra
    ndbc_spectra = []
    ndbc_labels = []
    for label, sub in ndbc_periods.items():
        try:
            sp = mean_spectra(sub, label)
            ndbc_spectra.append(sp)
            ndbc_labels.append(label)
            print(
                f"{label}: {sub.time.size} hourly records, Hs(mean spectrum) = "
                f"{4 * np.sqrt(float(sp.spectral_density.fillna(0).integrate('frequency'))):.2f} m"
            )
        except Exception as e:
            print(f"Warning: Could not compute NDBC spectra for {label}: {e}")

    era5_label = f"ERA5 model {nino_label}"
    era5_nino_spectrum = mean_spectra_era5(era5_efth, era5_label)
    print(
        f"{era5_label}: {era5_efth.time.size} hourly records, Hs(mean spectrum) = "
        f"{4 * np.sqrt(float(era5_nino_spectrum.spectral_density.integrate('frequency'))):.2f} m"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save NDBC spectra to netCDF
    out_nc = OUT_DIR / f"ndbc{STATION_ID}_mean_winter_spectra_elnino.nc"
    xr.concat(
        ndbc_spectra, dim=xr.DataArray(["all_winters", "el_nino_winters"], dims="period")
    ).assign_attrs(
        station_id=STATION_ID,
        winter_months="DJF",
        el_nino_winters=", ".join(f"{y}-{y + 1}" for y in found),
    ).to_netcdf(out_nc)
    print(f"Saved {out_nc}")

    # Plot Figure 1: Wave roses - NDBC only (with ERA5 added as third subplot)
    print("\nGenerating wave rose figure...")
    fig = plt.figure(figsize=(14, 5))
    fig.subplots_adjust(right=0.8, wspace=0.35)
    roses = []
    
    for i, (label, sub) in enumerate(ndbc_periods.items()):
        bulk = bulk_parameters(sub)
        rose = plot_rose(
            bulk.hs,
            bulk.dp,
            edges=HS_EDGES,
            nsector=NSECTOR,
            fig=fig,
            subplot=131 + i,
            legend=False,
        )
        rose.ax.set_title(label, wrap=True)
        roses.append(rose)
    
    # Add ERA5 El Nino rose as third subplot
    bulk_era5 = bulk_parameters_era5(era5_efth)
    rose_era5 = plot_rose(
        bulk_era5.hs,
        bulk_era5.dp,
        edges=HS_EDGES,
        nsector=NSECTOR,
        fig=fig,
        subplot=133,
        legend=True,
    )
    rose_era5.ax.set_title(era5_label, wrap=True)
    roses.append(rose_era5)
    
    rmax = max(r.ax.get_rmax() for r in roses)
    for r in roses:
        r.ax.set_rmax(rmax)
    fig.suptitle(
        f"Station {STATION_ID} wave roses: Hs [m] vs peak direction (coming from)"
    )
    fig.savefig(
        OUT_DIR / f"station{STATION_ID}_waverose_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved wave rose figure")

    # Figure 2: Directional spectra - NDBC + ERA5 El Nino
    print("Generating directional spectra figure...")
    fig2_spectra = [ndbc_spectra[0], ndbc_spectra[1], era5_nino_spectrum]
    fig2_titles = [ndbc_labels[0], ndbc_labels[1], era5_label]
    
    fig2 = plot_directional_spectra(fig2_spectra, fig2_titles, STATION_ID)
    fig2.savefig(
        OUT_DIR / f"station{STATION_ID}_dirspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved directional spectra figure")

    # Figure 3: line spectra - ERA5 (solid) and NDBC (dotted); all winters in black and
    # each El Nino winter separately
    print("\nComputing ERA5 DJF means for all winters in the archive...")
    winter_means = era5_winter_means(
        ERA5_DIR,
        float(ndbc_ds.attrs["latitude"]),
        float(ndbc_ds.attrs["longitude"]),
        OUT_DIR / f"era5_station{STATION_ID}_DJF_mean_spectra_by_winter.nc",
    )
    w = winter_means["count"]
    era5_all = era5_spectrum(
        (winter_means.efth * w).sum("winter") / w.sum(), "ERA5 all winters", int(w.sum())
    )
    y0, y1 = int(winter_means.winter.min()), int(winter_means.winter.max())

    def hs(sp):
        return 4 * np.sqrt(float(sp.spectral_density.fillna(0).integrate("frequency")))

    lines = [
        (era5_all, f"ERA5 all winters ({y0}-{y1 + 1})", "k", "-"),
        (ndbc_spectra[0], "NDBC all winters", "k", ":"),
    ]
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:brown"]
    for year, color in zip(EL_NINO_WINTERS, colors * len(EL_NINO_WINTERS)):
        tag = f"{year}-{year + 1}"
        if year in winter_means.winter.values:
            sel = winter_means.sel(winter=year)
            lines.append(
                (era5_spectrum(sel.efth, f"ERA5 {tag}", int(sel["count"])), f"ERA5 {tag}", color, "-")
            )
        if year in found:
            sub = ndbc_ds.isel(time=(is_winter & (winter_year == year)).values)
            lines.append((mean_spectra(sub, f"NDBC {tag}"), f"NDBC {tag}", color, ":"))
    for sp, label, _, _ in lines:
        print(f"  {label}: {int(sp['count'].max())} records, Hs(mean spectrum) = {hs(sp):.2f} m")

    fig3 = plot_frequency_spectra(lines, STATION_ID)
    fig3.savefig(
        OUT_DIR / f"station{STATION_ID}_freqspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved frequency spectra figure")
    
    # Figure 4: Violin plots with ERA5 distribution and both ERA5 & observed El Nino data (2015 only)
    if EL_NINO_VIOLIN:
        print("\nGenerating violin plot for bulk wave parameters...")
        year_nino = EL_NINO_VIOLIN[0]  # Only plot observed for this year (2015)
        if year_nino in found:
            # Compute ERA5 bulk stats for all winters (for violin distribution)
            print("  Computing ERA5 bulk statistics for all winters...")
            stats_all_era5 = era5_winter_bulk_stats(winter_means)
            
            # Get ERA5 stats for ALL El Nino winters (for circle markers)
            print("  Computing ERA5 bulk statistics for all El Nino years...")
            stats_era5_years_list = []
            for el_nino_year in EL_NINO_WINTERS:
                if el_nino_year in winter_means.winter.values:
                    era5_year = winter_means.sel(winter=el_nino_year)
                    era5_year_ds = xr.Dataset(
                        {var: era5_year[var].expand_dims("winter") for var in era5_year.data_vars},
                        coords={"winter": [el_nino_year]}
                    )
                    stats_year = era5_winter_bulk_stats(era5_year_ds)
                    stats_era5_years_list.append(stats_year)
            stats_era5_years = pd.concat(stats_era5_years_list, ignore_index=False) if stats_era5_years_list else pd.DataFrame()
            
            # Get observed NDBC stats for the El Nino year (2015 only)
            stats_nino_obs = winter_bulk_stats(
                ndbc_ds.isel(time=(is_winter & (winter_year == year_nino)).values), STATION_ID
            )
            
            fig4 = plot_violins(stats_all_era5, stats_era5_years, stats_nino_obs, year_nino, STATION_ID)
            fig4.savefig(
                OUT_DIR / f"station{STATION_ID}_violin_power_direction_winter_elnino.png",
                dpi=200,
                bbox_inches="tight",
            )
            print(f"Saved violin plot figure")
        else:
            print(f"Warning: El Nino {year_nino} data not available; skipping violin plot")
    
    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
