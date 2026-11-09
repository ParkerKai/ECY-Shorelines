# Enhanced NDBC Spectral Analysis with ERA5 Model Data

## Overview
The script has been updated to include ERA5 modeled wave spectra data alongside the NDBC buoy observations. This enables comparison between observed and modeled winter wave spectra during El Niño events.

## Changes Made

### 1. **Data Loading**
- `load_era5_station()` reads only the DJF monthly files for the El Ni?o winters that the buoy also observed
- For each file, the grid point nearest to the buoy (lat/lon from the NDBC file attributes) is selected **before** loading, so only that column is read from disk (a full monthly file is ~4.5 GB in memory)
- Decodes `d2fd`: values are log10(E), so E = 10**d2fd with missing (zero-energy) bins set to 0
- `frequencyNumber`/`directionNumber` are bin indices: f = 0.03453?1.1^(n-1) Hz, ? = 7.5? + 15??(n-1)
- ERA5 directions are "going to"; they are rotated 180? to "coming from" to match NDBC

### 2. **Spectral Analysis Functions**
- `mean_spectra_era5()`: time-mean E(f, ?) and S(f) = ?E d?
- `bulk_parameters_era5()`: Hs, Tp and peak direction (mean direction at the peak frequency, as for NDBC alpha1)

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
- Solid lines = ERA5 model, dotted lines = NDBC buoy
- Black = all winters (ERA5 line uses every complete DJF winter in the archive, 1940–2025)
- One coloured line per El Niño winter (not averaged); ERA5 lines appear for every
  El Niño winter, buoy lines only for winters inside the buoy record

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
5. **era5_station46088_DJF_mean_spectra_by_winter.nc** - Cache of per-winter ERA5 mean
   spectra at the station. The first run builds it (~17 min, ~255 monthly files); later
   runs only compute winters that are missing. Delete it to force a rebuild.

## Notes

- ERA5 loading takes ~4 s per monthly file (only the station point is read)
- NaN handling is built in for missing data periods
- Station location is automatically extracted from NDBC metadata attributes
- All times are kept in UTC (xarray time handling)

