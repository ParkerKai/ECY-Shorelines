"""Compare average winter (DJF) directional wave spectra at an NDBC buoy with El Nino winters.

Reads the NetCDF written by DownloadNDBC_DirectionalSpectra.py, computes the mean
frequency and directional spectra for all winters and for El Nino winters, saves them,
and plots side-by-side wave roses (Hs vs peak direction) with xoa plus the mean
directional spectra.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xoa.plot import plot_rose

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46088"
DATA_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\FullSpectral")
OUT_DIR = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\ENSO\figures\Waves"
)
WINTER_MONTHS = [12, 1, 2]
# winters are labelled by the year of their December, so 1982 means Dec 1982 - Feb 1983
EL_NINO_WINTERS = [1982, 1997, 2015]
HS_EDGES = [0, 1, 2, 3, 4, 6, 8]
NSECTOR = 16
NDIR = 72

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def load_station(data_dir, station_id):
    files = sorted(data_dir.glob(f"ndbc{station_id}_directional_spectra_*.nc"))
    if not files:
        raise SystemExit(f"No spectra file for station {station_id} in {data_dir}")
    ds = xr.open_dataset(files[-1]).load()
    # keep one record per hour so periods with 30/10-min sampling are not over-weighted
    _, idx = np.unique(ds.time.dt.floor("1h").values, return_index=True)
    return ds.isel(time=idx)


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


def plot_directional_spectra(spectra, titles, station_id):
    vmax = max(float(s.directional_spectrum.max()) for s in spectra)
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.8), subplot_kw={"projection": "polar"}
    )
    for ax, sp, title in zip(axes, spectra, titles):
        e = sp.directional_spectrum
        th = np.deg2rad(np.append(e.direction.values, 360.0))
        data = np.concatenate([e.values, e.values[:, :1]], axis=1)
        pc = ax.pcolormesh(
            th,
            1 / e.frequency.values,
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
        ax.set_title(title, pad=20)
    fig.colorbar(pc, ax=axes, shrink=0.8, label="E(f, θ) [m² Hz⁻¹ rad⁻¹]")
    fig.suptitle(
        f"NDBC {station_id} mean DJF directional spectra (radius = period [s])"
    )
    return fig


def plot_frequency_spectra(spectra, titles, station_id):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for sp, title, color in zip(spectra, titles, ["k", "tab:red"]):
        ax1.plot(1 / sp.frequency, sp.spectral_density, color=color, label=title)
        d = sp.directional_spectrum.fillna(0).integrate("frequency")
        d_closed = np.append(d.values, d.values[0])
        ax2.plot(np.append(d.direction.values, 360), d_closed, color=color, label=title)
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
    ax1.legend()
    fig.suptitle(f"NDBC {station_id} mean DJF energy density spectra")
    fig.tight_layout()
    return fig


def main():
    ds = load_station(DATA_DIR, STATION_ID)
    winter_year = ds.time.dt.year.where(ds.time.dt.month == 12, ds.time.dt.year - 1)
    is_winter = ds.time.dt.month.isin(WINTER_MONTHS)
    is_nino = is_winter & winter_year.isin(EL_NINO_WINTERS)

    found = sorted({int(y) for y in winter_year.where(is_nino, drop=True).values})
    missing = sorted(set(EL_NINO_WINTERS) - set(found))
    print(f"Station {STATION_ID}: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"El Nino winters with data: {found}; missing: {missing}")
    if not found:
        raise SystemExit("No El Nino winter data at this station.")

    periods = {
        "All winters (DJF)": ds.isel(time=is_winter.values),
        "El Nino winters (DJF "
        + ", ".join(f"{y}-{y + 1}" for y in found)
        + ")": ds.isel(time=is_nino.values),
    }

    spectra = []
    for label, sub in periods.items():
        sp = mean_spectra(sub, label)
        spectra.append(sp)
        print(
            f"{label}: {sub.time.size} hourly records, Hs(mean spectrum) = "
            f"{4 * np.sqrt(float(sp.spectral_density.fillna(0).integrate('frequency'))):.2f} m"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nc = OUT_DIR / f"ndbc{STATION_ID}_mean_winter_spectra_elnino.nc"
    xr.concat(
        spectra, dim=xr.DataArray(["all_winters", "el_nino_winters"], dims="period")
    ).assign_attrs(
        station_id=STATION_ID,
        winter_months="DJF",
        el_nino_winters=", ".join(f"{y}-{y + 1}" for y in found),
    ).to_netcdf(out_nc)
    print(f"Saved {out_nc}")

    fig = plt.figure(figsize=(10.5, 4.5))
    fig.subplots_adjust(right=0.75, wspace=0.4)  # room for the legend
    roses = []
    for i, (label, sub) in enumerate(periods.items()):
        bulk = bulk_parameters(sub)
        rose = plot_rose(
            bulk.hs,
            bulk.dp,
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
        f"NDBC {STATION_ID} wave roses: Hs [m] vs peak direction (coming from)"
    )
    fig.savefig(
        OUT_DIR / f"ndbc{STATION_ID}_waverose_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig2 = plot_directional_spectra(spectra, list(periods), STATION_ID)
    fig2.savefig(
        OUT_DIR / f"ndbc{STATION_ID}_dirspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig3 = plot_frequency_spectra(spectra, list(periods), STATION_ID)
    fig3.savefig(
        OUT_DIR / f"ndbc{STATION_ID}_freqspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
