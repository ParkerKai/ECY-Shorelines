"""Compare average winter (DJF) directional wave spectra at an NDBC buoy with El Nino winters.

Reads the NetCDF written by DownloadNDBC_DirectionalSpectra.py, computes the mean
frequency and directional spectra for all winters and for El Nino winters, saves them,
and plots side-by-side wave roses (Hs vs peak direction) with xoa plus the mean
directional spectra. Also compares with ERA5 modelled wave spectra.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xoa.plot import plot_rose

# ----------------------------------------------------------------------------- user inputs
STATION_ID = "46088"
DATA_DIR = Path(rf"D:\Data\Waves\buoys\ndbc\ndbc{STATION_ID}\FullSpectral")
ERA5_DIR = Path(r"E:\ERA5_Waves\PNW_Era5Spectral")
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


def load_era5_station(era5_dir, ndbc_ds):
    """Load ERA5 spectral data for the NDBC station location."""
    # Get station coordinates from NDBC data
    station_lat = float(ndbc_ds.attrs.get('latitude', None))
    station_lon = float(ndbc_ds.attrs.get('longitude', None))
    
    if station_lat is None or station_lon is None:
        raise SystemExit("Could not find latitude/longitude in NDBC dataset attributes")
    
    print(f"Station location: ({station_lat:.2f}°N, {abs(station_lon):.2f}°W)")
    
    # Load all ERA5 files for the date range in NDBC data
    ndbc_years = set(ndbc_ds.time.dt.year.values)
    files_to_load = []
    
    for year in sorted(ndbc_years):
        year_files = sorted(era5_dir.glob(f"ERA5_SpectralWaves_WC_{year}_*.nc"))
        files_to_load.extend(year_files)
    
    if not files_to_load:
        raise SystemExit(f"No ERA5 files found for years {sorted(ndbc_years)} in {era5_dir}")
    
    print(f"Loading {len(files_to_load)} ERA5 files...")
    datasets = [xr.open_dataset(f) for f in files_to_load]
    era5_ds = xr.concat(datasets, dim='valid_time').load()
    
    # Rename valid_time to time for consistency
    era5_ds = era5_ds.rename({'valid_time': 'time'})
    
    # Find nearest grid point to station
    # Normalize longitude to -180 to 180 if needed
    era5_lons = era5_ds.longitude.values
    station_lon_normalized = ((station_lon + 180) % 360) - 180
    
    # Find nearest point
    dist = np.sqrt(
        (era5_ds.latitude.values - station_lat)**2 + 
        (era5_lons - station_lon_normalized)**2
    )
    # dist is 1D (latitude shape), we need to find the min across all points
    nearest_lat_idx = np.argmin(np.abs(era5_ds.latitude.values - station_lat))
    nearest_lon_idx = np.argmin(np.abs(era5_lons - station_lon_normalized))
    
    print(f"Nearest ERA5 grid point: {era5_ds.latitude.values[nearest_lat_idx]:.2f}°, "
          f"{era5_lons[nearest_lon_idx]:.2f}°")
    
    # Extract data at nearest point
    era5_station = era5_ds.isel(latitude=nearest_lat_idx, longitude=nearest_lon_idx)
    
    # Resample to hourly if needed (similar to NDBC)
    if 'time' in era5_station.dims:
        _, idx = np.unique(era5_station.time.dt.floor("1h").values, return_index=True)
        era5_station = era5_station.isel(time=idx)
    
    return era5_station


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


def fourier_coefficients_era5(ds):
    """Extract directional spectral components from ERA5 d2fd variable.
    
    Assumes d2fd has dimensions (time, directionNumber, frequencyNumber)
    after extracting a single grid point, and represents the 2D directional 
    wave spectrum E(f, θ).
    """
    if 'd2fd' not in ds.data_vars:
        raise ValueError(f"Variable 'd2fd' not found in ERA5 data. Available: {list(ds.data_vars)}")
    
    e = ds.d2fd  # (time, directionNumber, frequencyNumber)
    
    # Integrate over direction to get frequency spectrum
    s_f = e.integrate('directionNumber')
    
    # Compute mean direction and directional spreading
    # Create xarray DataArrays with proper dimension names for broadcasting
    theta_rad = np.deg2rad(ds.directionNumber.values)
    theta_da = xr.DataArray(theta_rad, dims=['directionNumber'], coords={'directionNumber': ds.directionNumber})
    
    cos_theta = np.cos(theta_rad)
    cos_theta_da = xr.DataArray(cos_theta, dims=['directionNumber'], coords={'directionNumber': ds.directionNumber})
    
    sin_theta = np.sin(theta_rad)
    sin_theta_da = xr.DataArray(sin_theta, dims=['directionNumber'], coords={'directionNumber': ds.directionNumber})
    
    cos_2theta = np.cos(2 * theta_rad)
    cos_2theta_da = xr.DataArray(cos_2theta, dims=['directionNumber'], coords={'directionNumber': ds.directionNumber})
    
    sin_2theta = np.sin(2 * theta_rad)
    sin_2theta_da = xr.DataArray(sin_2theta, dims=['directionNumber'], coords={'directionNumber': ds.directionNumber})
    
    # Integrate directional distribution weighted by frequency spectrum
    a1 = (e * cos_theta_da).integrate('directionNumber') / (s_f + 1e-10)
    b1 = (e * sin_theta_da).integrate('directionNumber') / (s_f + 1e-10)
    a2 = (e * cos_2theta_da).integrate('directionNumber') / (s_f + 1e-10)
    b2 = (e * sin_2theta_da).integrate('directionNumber') / (s_f + 1e-10)
    
    coefs = xr.Dataset(
        {
            "S": s_f,
            "Sa1": s_f * a1,
            "Sb1": s_f * b1,
            "Sa2": s_f * a2,
            "Sb2": s_f * b2,
        }
    )
    return coefs


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


def bulk_parameters_era5(ds):
    """Compute bulk wave parameters from ERA5 2D spectrum."""
    if 'd2fd' not in ds.data_vars:
        raise ValueError("Variable 'd2fd' not found in ERA5 data")
    
    # Frequency spectrum
    s_f = ds.d2fd.integrate('directionNumber').fillna(0)
    hs = 4 * np.sqrt(s_f.integrate('frequencyNumber'))
    
    # Peak frequency and period - use fillna to handle all-NaN slices
    s_f_filled = s_f.fillna(0)
    ipeak = s_f_filled.argmax('frequencyNumber')
    tp = 1 / ds.frequencyNumber.isel(frequencyNumber=ipeak)
    
    # Peak direction: find direction at peak frequency
    s_peak = ds.d2fd.isel(frequencyNumber=ipeak).fillna(0)
    ipeak_dir = s_peak.argmax('directionNumber')
    dp = ds.directionNumber.isel(directionNumber=ipeak_dir)
    
    valid = hs > 0
    return xr.Dataset(
        {
            "hs": hs.where(valid).assign_attrs(
                long_name="significant wave height", units="m"
            ),
            "tp": tp.where(valid).assign_attrs(long_name="peak period", units="s"),
            "dp": dp.where(valid).assign_attrs(
                long_name="peak wave direction at peak frequency, coming from",
                units="degree",
            ),
        }
    )


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


def mean_spectra_era5(ds, label):
    """Compute mean directional spectra from ERA5 2D spectrum."""
    coefs = fourier_coefficients_era5(ds)
    
    # Average over time records with valid data
    complete = coefs.to_array().notnull().all("variable")
    mean = coefs.where(complete).mean("time")
    count = complete.sum("time")
    
    # Compute directional spectrum for standard grid
    theta = xr.DataArray(
        np.arange(0, 360, 360 / NDIR),
        dims="direction",
        attrs={"long_name": "direction waves come from", "units": "degree"},
    )
    th = np.deg2rad(theta)
    
    # Reconstruct directional spectrum from Fourier coefficients
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
    return out.transpose("frequencyNumber", "direction")


def plot_directional_spectra(spectra, titles, station_id):
    vmax = max(float(s.directional_spectrum.max()) for s in spectra)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), subplot_kw={"projection": "polar"}
    )
    axes = axes.flatten()
    for ax, sp, title in zip(axes[:len(spectra)], spectra, titles):
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
        ax.set_title(title, pad=20)
    
    # Hide unused subplots if only 3 spectra
    for ax in axes[len(spectra):]:
        ax.set_visible(False)
    
    fig.colorbar(pc, ax=axes[:len(spectra)], shrink=0.8, label="E(f, θ) [m² Hz⁻¹ rad⁻¹]")
    fig.suptitle(
        f"Station {station_id} mean DJF directional spectra (radius = period [s])"
    )
    return fig


def plot_frequency_spectra(spectra, titles, station_id):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["k", "tab:red", "tab:blue", "tab:green"]
    for sp, title, color in zip(spectra, titles, colors[:len(spectra)]):
        freq_dims = [d for d in sp.spectral_density.dims if 'freq' in d.lower()]
        freq_dim = freq_dims[0] if freq_dims else 'frequency'
        ax1.plot(1 / sp[freq_dim], sp.spectral_density, color=color, label=title, linewidth=2)
        d = sp.directional_spectrum.fillna(0).integrate(freq_dim)
        d_closed = np.append(d.values, d.values[0])
        ax2.plot(np.append(d.direction.values, 360), d_closed, color=color, label=title, linewidth=2)
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
    fig.suptitle(f"Station {station_id} mean DJF energy density spectra")
    fig.tight_layout()
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

    # Load ERA5 data
    print(f"\nLoading ERA5 data for station {STATION_ID}...")
    era5_ds = load_era5_station(ERA5_DIR, ndbc_ds)
    
    # Filter ERA5 to winters
    era5_winter_year = era5_ds.time.dt.year.where(
        era5_ds.time.dt.month == 12, era5_ds.time.dt.year - 1
    )
    era5_is_winter = era5_ds.time.dt.month.isin(WINTER_MONTHS)
    era5_is_nino = era5_is_winter & era5_winter_year.isin(found)

    ndbc_periods = {
        "NDBC All winters (DJF)": ndbc_ds.isel(time=is_winter.values),
        "NDBC El Nino winters (DJF "
        + ", ".join(f"{y}-{y + 1}" for y in found)
        + ")": ndbc_ds.isel(time=is_nino.values),
    }
    
    era5_periods = {
        "ERA5 model All winters (DJF)": era5_ds.isel(time=era5_is_winter.values),
        "ERA5 model El Nino winters (DJF "
        + ", ".join(f"{y}-{y + 1}" for y in found)
        + ")": era5_ds.isel(time=era5_is_nino.values),
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

    era5_spectra = []
    era5_labels = []
    for label, sub in era5_periods.items():
        try:
            sp = mean_spectra_era5(sub, label)
            era5_spectra.append(sp)
            era5_labels.append(label)
            freq_dim = [d for d in sp.spectral_density.dims if 'freq' in d.lower()][0]
            print(
                f"{label}: {sub.time.size} hourly records, Hs(mean spectrum) = "
                f"{4 * np.sqrt(float(sp.spectral_density.fillna(0).integrate(freq_dim))):.2f} m"
            )
        except Exception as e:
            print(f"Warning: Could not compute ERA5 spectra for {label}: {e}")

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
    if era5_spectra:
        try:
            era5_nino_key = "ERA5 model El Nino winters (DJF " + ", ".join(f"{y}-{y + 1}" for y in found) + ")"
            bulk_era5 = bulk_parameters_era5(era5_periods[era5_nino_key])
            rose_era5 = plot_rose(
                bulk_era5.hs,
                bulk_era5.dp,
                edges=HS_EDGES,
                nsector=NSECTOR,
                fig=fig,
                subplot=133,
                legend=False,
            )
            rose_era5.ax.set_title(era5_labels[1] if len(era5_labels) > 1 else "ERA5 model El Nino", wrap=True)
            roses.append(rose_era5)
        except Exception as e:
            print(f"Warning: Could not plot ERA5 wave rose: {e}")
    
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
    fig2_spectra = [ndbc_spectra[0], ndbc_spectra[1]]
    fig2_titles = [ndbc_labels[0], ndbc_labels[1]]
    if era5_spectra:
        fig2_spectra.append(era5_spectra[1])
        fig2_titles.append(era5_labels[1])
    
    fig2 = plot_directional_spectra(fig2_spectra, fig2_titles, STATION_ID)
    fig2.savefig(
        OUT_DIR / f"station{STATION_ID}_dirspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved directional spectra figure")

    # Figure 3: Frequency spectra - NDBC + ERA5 El Nino
    print("Generating frequency spectra figure...")
    fig3_spectra = [ndbc_spectra[0], ndbc_spectra[1]]
    fig3_titles = [ndbc_labels[0], ndbc_labels[1]]
    if era5_spectra:
        fig3_spectra.append(era5_spectra[1])
        fig3_titles.append(era5_labels[1])
    
    fig3 = plot_frequency_spectra(fig3_spectra, fig3_titles, STATION_ID)
    fig3.savefig(
        OUT_DIR / f"station{STATION_ID}_freqspec_winter_elnino.png",
        dpi=200,
        bbox_inches="tight",
    )
    print(f"Saved frequency spectra figure")
    
    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
