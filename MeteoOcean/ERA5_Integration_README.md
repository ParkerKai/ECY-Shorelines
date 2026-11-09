# Enhanced NDBC Spectral Analysis with ERA5 Model Data

## Overview
The script has been updated to include ERA5 modeled wave spectra data alongside the NDBC buoy observations. This enables comparison between observed and modeled winter wave spectra during El Niño events.

## Changes Made

### 1. **Data Loading**
- Added `load_era5_station()` function to load ERA5 spectral netCDF files from `E:\ERA5_Waves\PNW_Era5Spectral`
- Automatically extracts data at the nearest grid point to the NDBC station using coordinates from NDBC metadata
- Handles time coordinate renaming (`valid_time` → `time` for consistency)
- Resamples to hourly data to match NDBC frequency

### 2. **Spectral Analysis Functions**
Added ERA5-specific functions:
- `fourier_coefficients_era5()`: Computes directional Fourier coefficients (a1, b1, a2, b2) from ERA5 2D directional spectrum (d2fd)
- `mean_spectra_era5()`: Computes mean spectra for specified periods (all winters, El Niño winters)
- `bulk_parameters_era5()`: Extracts bulk parameters (Hs, Tp, Dp) from ERA5 2D spectrum

### 3. **Plotting Updates**
- `plot_directional_spectra()`: Now handles up to 4 subplots (2x2 grid) for NDBC all/El Niño + ERA5 El Niño
- `plot_frequency_spectra()`: Updated to handle 3+ spectral datasets with different colors
- Both functions now handle flexible frequency dimension names (`frequency` or `frequencyNumber`)

### 3. **Output Figures**

**Figure 1 - Wave Roses** (3 subplots):
- Left: NDBC all winters (DJF)
- Middle: NDBC El Niño winters  
- Right: ERA5 model El Niño winters

**Figure 2 - Directional Spectra** (up to 4 polar subplots):
- NDBC all winters
- NDBC El Niño winters
- ERA5 model El Niño winters
- 4th subplot (unused for now, hidden)

**Figure 3 - Frequency & Directional Spectra** (2 line plots):
- Frequency spectrum: S(f) vs period
- Directional spectrum: D(θ) vs direction
- Both show all available datasets (NDBC obs and ERA5 model)

## Data Structure

### NDBC Files
- Location: `D:\Data\Waves\buoys\ndbc\ndbc46088\FullSpectral`
- Variables: `spectral_density`, `alpha1`, `alpha2`, `r1`, `r2` (directional Fourier coefficients)
- Frequency dimension: 47 bands
- Coordinates: time, frequency, latitude (metadata), longitude (metadata)

### ERA5 Files
- Location: `E:\ERA5_Waves\PNW_Era5Spectral`
- Filename pattern: `ERA5_SpectralWaves_WC_{YEAR}_{MONTH}.nc`
- Main variable: `d2fd` (2D directional spectrum, shape: time × direction × frequency × lat × lon)
- Dimensions: 24 directions, 30 frequencies, 56 latitudes, 42 longitudes
- Time coordinate: `valid_time` (hourly)

## Configuration

The script uses these settings (customizable in the "user inputs" section):

```python
STATION_ID = "46088"                    # NDBC station ID
DATA_DIR = Path(r"D:\Data\Waves\buoys\ndbc\ndbc46088\FullSpectral")
ERA5_DIR = Path(r"E:\ERA5_Waves\PNW_Era5Spectral")
OUT_DIR = Path(r"...\figures\Waves")    # Output location
WINTER_MONTHS = [12, 1, 2]              # December, January, February
EL_NINO_WINTERS = [1982, 1997, 2015]    # Years of El Niño winters (by December year)
HS_EDGES = [0, 1, 2, 3, 4, 6, 8]        # Wave height bin edges (m)
NSECTOR = 16                             # Number of direction sectors for wave rose
NDIR = 72                                # Number of direction bins in spectra
```

## Running the Script

```bash
cd C:\Users\kaip461\GitHub\ECY-Shorelines\MeteoOcean
& "C:\Users\kaip461\AppData\Local\miniforge3\envs\geoprocessing\python.exe" NDBC_WinterSpectra_ElNino.py
```

**Dependencies**: xarray, numpy, matplotlib, xoa

## Output Files

All files saved to `OUT_DIR`:

1. **ndbc46088_mean_winter_spectra_elnino.nc** - NetCDF with mean NDBC spectra
2. **station46088_waverose_winter_elnino.png** - 3-panel wave rose comparison
3. **station46088_dirspec_winter_elnino.png** - Polar directional spectrum plots
4. **station46088_freqspec_winter_elnino.png** - Frequency and directional spectra

## Notes

- The script loads full monthly ERA5 files; for large date ranges this may take several minutes
- NaN handling is built in for missing data periods
- Station location is automatically extracted from NDBC metadata attributes
- All times are kept in UTC (xarray time handling)

