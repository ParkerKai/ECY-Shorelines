# -*- coding: utf-8 -*-
"""
Created on Oct 30, 2026

This script compares CORA outputs vs Tide Gauge observational data

@author: kaparker
    USGS: PCMSC
    kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import sys
import os
import json
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from glob import glob
import utide

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the CORA data resides
dir_in_cora = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WLs\Cora\TideGauges"
dir_in_tg = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\6_min"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WLs\Cora\Validation"

# Create output directory if it doesn't exist
os.makedirs(dir_out, exist_ok=True)

# Gauges to process
Gauges = [
    "9443090",  # Neah Bay
    "9444090",  # Port Angeles
    "9444900",  # Port Townsend
    "9447130",  # Seattle
    "9446484",  # Tacoma
    "9449880",  # Friday Harbor
    "9449424",  # Cherry Point
    "9442396",  # La Push
    "9440910",  # Toke Point
    "9441102",  # Westport
    "9440581"   # Cape Dissappointment
]

Gauge_name = [
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Seattle",
    "Tacoma",
    "Friday Harbor",
    "Cherry Point",
    "La Push",
    "Toke Point",
    "Westport",
    "CapeDissappointment"
    
]

# Variable to process (calculate stats and make plots)
Var_process = "ntr"  # 'wl', 'tide', 'ntr'

# Candidate column names in the tide gauge parquet for predicted tide / NTR
TG_tide_cols = ["v_pred", "pred", "tide", "v_tide"]
TG_ntr_cols = ["v_ntr", "ntr", "residual"]

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")

try:
    from Kai_ModelValidation import (
        mean_absolute_difference_corrected,
        ModelStat_Var,
        intersect_Var,
        bias,
    )
except ImportError:
    print("Warning: Could not import Kai_ModelValidation functions")
    print("Using fallback implementations...")

    def mean_absolute_difference_corrected(obs, pred):
        """Mean absolute difference corrected"""
        return np.nanmean(np.abs(obs - pred))

    def bias(obs, pred):
        """Bias (mean residual)"""
        return np.nanmean(pred - obs)


def Read_TG_Parquet(dir_in_tg, gauge_ID, start_date=None, end_date=None):
    """
    Read tide gauge data from parquet files

    Parameters:
    -----------
    dir_in_tg : str
        Directory containing tide gauge data
    gauge_ID : str
        Gauge ID (NOAA station ID)
    start_date : str or None
        Start date for filtering (format: 'YYYY-MM-DD')
    end_date : str or None
        End date for filtering (format: 'YYYY-MM-DD')

    Returns:
    --------
    ds : xarray.Dataset
        Dataset with water level data
    """
    # Path to the gauge directory
    gauge_dir = os.path.join(dir_in_tg, f"NOAA_TG_{gauge_ID}")

    # Read the combined_resolved parquet file
    parquet_file = os.path.join(gauge_dir, "combined_resolved.parquet")

    if not os.path.exists(parquet_file):
        print(f"Warning: {parquet_file} not found")
        return None

    # Read with filters if specified
    df = pd.read_parquet(parquet_file)

    # Time column is 't', not 'time'
    if "t" in df.columns:
        df["time"] = pd.to_datetime(df["t"])
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        raise ValueError(f"No 'time' or 't' column in {parquet_file}")

    if "v_obs" not in df.columns:
        raise ValueError(f"No 'v_obs' column in {parquet_file}")

    # Filter by date if specified
    if start_date is not None:
        df = df[df["time"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["time"] <= pd.to_datetime(end_date)]

    # Sort by time
    df = df.sort_values("time")

    tide_col = next((c for c in TG_tide_cols if c in df.columns), None)
    if tide_col is None:
        raise ValueError(
            f"No predicted tide column ({TG_tide_cols}) in {parquet_file}. "
            f"Available: {list(df.columns)}"
        )
    wl = df["v_obs"].values.astype(float)
    tide = df[tide_col].values.astype(float)
    ntr_col = next((c for c in TG_ntr_cols if c in df.columns), None)
    ntr = df[ntr_col].values.astype(float) if ntr_col else wl - tide

    # Create xarray Dataset
    data_vars = {
        "wl": (["time"], wl, {"units": "m", "long_name": "Water Level Observed"}),
        "tide": (["time"], tide, {"units": "m", "long_name": "Tide (Predicted)"}),
        "ntr": (["time"], ntr, {"units": "m", "long_name": "Non-Tidal Residual"}),
    }

    coords = {"time": (["time"], df["time"].values, {"standard_name": "time observed"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"station_id": gauge_ID})

    return ds


def Read_TG_Latitude(dir_in_tg, gauge_ID):
    """Read station latitude from the gauge metadata JSON."""
    meta_files = glob(os.path.join(dir_in_tg, f"NOAA_TG_{gauge_ID}", "metadata_*.json"))
    if len(meta_files) == 0:
        raise FileNotFoundError(f"No metadata JSON found for gauge {gauge_ID}")
    with open(meta_files[0], "r") as f:
        return float(json.load(f)["Latitude"])


def Read_CORA_Parquet(file_path):
    """
    Read CORA data from parquet file

    Parameters:
    -----------
    file_path : str
        Path to CORA parquet file

    Returns:
    --------
    ds : xarray.Dataset
        Dataset with water level data
    """
    df = pd.read_parquet(file_path)

    # Ensure time column is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError(f"No time column found in {file_path}")

    if "water_level_m" not in df.columns:
        raise ValueError(f"No 'water_level_m' column in {file_path}")

    # Sort by time
    df = df.sort_values("time")

    # Create xarray Dataset
    data_vars = {
        "wl": (
            ["time"],
            df["water_level_m"].values,
            {"units": "m", "long_name": "Water Level (CORA)"},
        ),
    }

    coords = {"time": (["time"], df["time"].values, {"standard_name": "time"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds


def add_utide_tide(ds, lat):
    """
    Harmonic analysis of CORA water levels with UTide; adds 'tide' and 'ntr'.
    """
    t = pd.to_datetime(ds["time"].values)
    wl = ds["wl"].values.astype(float)

    # trend=False so sea-level trends remain in the NTR, as with NOAA predictions
    coef = utide.solve(
        t, wl, lat=lat, method="ols", conf_int="none", trend=False, verbose=False
    )
    tide = utide.reconstruct(t, coef, verbose=False).h

    ds["tide"] = (["time"], tide, {"units": "m", "long_name": "Tide (UTide)"})
    ds["ntr"] = (["time"], wl - tide, {"units": "m", "long_name": "Non-Tidal Residual"})
    return ds


def resample_to_hourly(ds):
    """
    Resample dataset to hourly resolution using linear interpolation

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset

    Returns:
    --------
    ds_hourly : xarray.Dataset
        Hourly resampled dataset
    """
    # Create hourly time grid
    t_start = pd.to_datetime(ds["time"].values[0]).floor("h")
    t_end = pd.to_datetime(ds["time"].values[-1]).ceil("h")
    t_new = pd.date_range(t_start, t_end, freq="h")

    # Interpolate to hourly grid
    data_vars = {}
    for var in ds.data_vars:
        vals = np.interp(
            t_new.astype(np.int64),
            ds["time"].values.astype(np.int64),
            ds[var].values,
            left=np.nan,
            right=np.nan,
        )
        data_vars[var] = (["time"], vals, ds[var].attrs)

    coords = {"time": (["time"], t_new.values, {"standard_name": "time"})}

    ds_hourly = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds_hourly


def intersect_datasets(ds_model, ds_obs):
    """
    Find temporal intersection of two datasets

    Parameters:
    -----------
    ds_model : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observation dataset

    Returns:
    --------
    t_intersect : np.ndarray
        Times present in both datasets
    """
    # Find overlapping time range
    t_model = pd.to_datetime(ds_model["time"].values)
    t_obs = pd.to_datetime(ds_obs["time"].values)

    t_start = max(t_model.min(), t_obs.min())
    t_end = min(t_model.max(), t_obs.max())

    # Select data within overlapping range
    t_intersect = pd.date_range(t_start, t_end, freq="h")

    return t_intersect.values


# ===============================================================================
# %% Read in the CORA Model Data FIRST (to determine time range)
# ===============================================================================
print("Reading CORA model data to determine time range...")
cora_datasets = []
cora_time_ranges = []

for cnt, gauge_ID in enumerate(Gauges):
    print(f"  Reading CORA data for {gauge_ID} ({Gauge_name[cnt]})...")

    # Find CORA file for this gauge
    cora_files = glob(os.path.join(dir_in_cora, f"{gauge_ID}*.parquet"))

    if len(cora_files) == 0:
        print(f"  WARNING: No CORA file found for gauge {gauge_ID}, skipping...")
        cora_datasets.append(None)
        cora_time_ranges.append(None)
        continue

    cora_file = cora_files[0]
    print(f"    Using file: {os.path.basename(cora_file)}")

    try:
        ds = Read_CORA_Parquet(cora_file)
        print("    Running UTide harmonic analysis...")
        ds = add_utide_tide(ds, Read_TG_Latitude(dir_in_tg, gauge_ID))
        cora_datasets.append(ds)
        cora_time_ranges.append((ds["time"].values.min(), ds["time"].values.max()))
        print(
            f"    Time range: {pd.to_datetime(ds['time'].values.min())} to {pd.to_datetime(ds['time'].values.max())}"
        )
    except Exception as e:
        print(f"  ERROR reading {cora_file}: {e}")
        cora_datasets.append(None)
        cora_time_ranges.append(None)

print(
    f"Successfully loaded {sum(1 for x in cora_datasets if x is not None)} CORA datasets"
)

# ===============================================================================
# %% Read in the Tide Gauge observation data (filtered to CORA time range)
# ===============================================================================
print("\nReading Tide Gauge observation data (filtered to CORA time range)...")
ds_gauge = []
for cnt, gauge_ID in enumerate(Gauges):
    if cora_time_ranges[cnt] is None:
        print(f"  Skipping gauge {gauge_ID} (no CORA data)")
        ds_gauge.append(None)
        continue

    print(f"  Reading Gauge {gauge_ID} ({Gauge_name[cnt]})...")

    # Get time range from CORA data
    t_start = pd.to_datetime(cora_time_ranges[cnt][0])
    t_end = pd.to_datetime(cora_time_ranges[cnt][1])

    print(f"    CORA time range: {t_start} to {t_end}")

    ds = Read_TG_Parquet(
        dir_in_tg,
        gauge_ID,
        start_date=t_start.strftime("%Y-%m-%d"),
        end_date=t_end.strftime("%Y-%m-%d"),
    )

    if ds is None:
        print(f"  WARNING: Could not read gauge {gauge_ID}, skipping...")
        ds_gauge.append(None)
        continue

    # Resample to hourly
    ds = resample_to_hourly(ds)
    ds_gauge.append(ds)
    print(f"    Resampled to hourly. Shape: {ds[Var_process].shape}")

print(
    f"Successfully loaded {len([d for d in ds_gauge if d is not None])} gauge datasets"
)

# ===============================================================================
# %% Prepare model data
# ===============================================================================
print("\nPreparing model data...")
ds_model = []

for cnt, gauge_ID in enumerate(Gauges):
    if cora_datasets[cnt] is None:
        print(f"  Skipping gauge {gauge_ID} (no model data)")
        ds_model.append(None)
        continue

    if ds_gauge[cnt] is None:
        print(f"  Skipping gauge {gauge_ID} (no observation data)")
        ds_model.append(None)
        continue

    print(f"  Preparing model data for {gauge_ID}...")
    ds_model.append(cora_datasets[cnt])

print(
    f"Successfully prepared {sum(1 for x in ds_model if x is not None)} model datasets"
)

# ===============================================================================
# %% Model validation Statistics
# ===============================================================================
print("\nCalculating validation statistics...")

metrics = [root_mean_squared_error, r2_score, mean_absolute_difference_corrected, bias]

t_intersect_save = []
Y_pred = []
slope = np.full((len(Gauges)), np.nan)
model_perf = np.full((len(Gauges), len(metrics)), np.nan)
std_ratio = np.full((len(Gauges)), np.nan)
corr_coef = np.full((len(Gauges)), np.nan)

valid_indices = []

for cnt1, gauge_ID in enumerate(Gauges):
    if cnt1 >= len(ds_gauge) or ds_gauge[cnt1] is None:
        print(f"Skipping {gauge_ID} (no observation data)")
        continue

    if cnt1 >= len(ds_model) or ds_model[cnt1] is None:
        print(f"Skipping {gauge_ID} (no model data)")
        continue

    print(f"Processing Gauge {gauge_ID} ({Gauge_name[cnt1]})...")
    valid_indices.append(cnt1)

    pull_gauge = ds_gauge[cnt1]
    pull_model = ds_model[cnt1]

    # Temporal intersection
    t_intersect = intersect_datasets(pull_model, pull_gauge)

    if len(t_intersect) == 0:
        print(f"  WARNING: No temporal overlap for {gauge_ID}")
        valid_indices.pop()
        continue

    t_intersect_save.append(t_intersect)

    # Subset to overlapping times
    pull_gauge_sel = pull_gauge.sel(time=t_intersect)
    pull_model_sel = pull_model.sel(time=t_intersect)

    # Remove NaN values
    valid_mask = ~(
        np.isnan(pull_gauge_sel[Var_process].values)
        | np.isnan(pull_model_sel[Var_process].values)
    )

    if np.sum(valid_mask) == 0:
        print(f"  WARNING: All values are NaN for {gauge_ID}")
        valid_indices.pop()
        t_intersect_save.pop()
        continue

    obs_data = pull_gauge_sel[Var_process].values[valid_mask]
    model_data = pull_model_sel[Var_process].values[valid_mask]

    print(f"  Valid data points: {len(obs_data)}")
    print(
        f"  Obs  - Min: {np.nanmin(obs_data):.3f}, Max: {np.nanmax(obs_data):.3f}, Mean: {np.nanmean(obs_data):.3f}"
    )
    print(
        f"  Model - Min: {np.nanmin(model_data):.3f}, Max: {np.nanmax(model_data):.3f}, Mean: {np.nanmean(model_data):.3f}"
    )

    # Calculate performance metrics
    try:
        model_perf[cnt1, 0] = root_mean_squared_error(obs_data, model_data)
        model_perf[cnt1, 1] = r2_score(obs_data, model_data)
        model_perf[cnt1, 2] = mean_absolute_difference_corrected(obs_data, model_data)
        model_perf[cnt1, 3] = bias(obs_data, model_data)
        std_ratio[cnt1] = np.std(model_data, ddof=1) / np.std(obs_data, ddof=1)
        corr_coef[cnt1] = np.corrcoef(obs_data, model_data)[0, 1]
        print(
            f"  RMSE: {model_perf[cnt1, 0]:.4f}, R²: {model_perf[cnt1, 1]:.4f}, Bias: {model_perf[cnt1, 3]:.4f}"
        )
    except Exception as e:
        print(f"  ERROR calculating metrics for {gauge_ID}: {e}")
        valid_indices.pop()
        continue

    # Fit regression line
    try:
        model_reg = LinearRegression()
        model_reg.fit(obs_data.reshape(-1, 1), model_data)
        slope[cnt1] = model_reg.coef_.squeeze()
        Y_pred.append(model_reg.predict(obs_data.reshape(-1, 1)))
    except Exception as e:
        print(f"  ERROR fitting regression for {gauge_ID}: {e}")
        Y_pred.append(None)

# Create metrics dataframe for valid gauges
if len(valid_indices) > 0:
    ModelMetrics = pd.DataFrame(
        {
            "Gauge_ID": [Gauges[i] for i in valid_indices],
            "Gauge_Name": [Gauge_name[i] for i in valid_indices],
            "RMSE": [model_perf[i, 0] for i in valid_indices],
            "R2": [model_perf[i, 1] for i in valid_indices],
            "MADc": [model_perf[i, 2] for i in valid_indices],
            "Bias": [model_perf[i, 3] for i in valid_indices],
            "Slope": [slope[i] for i in valid_indices],
            "StdRatio": [std_ratio[i] for i in valid_indices],
            "Corr": [corr_coef[i] for i in valid_indices],
        }
    )
else:
    print("ERROR: No valid gauges to process!")
    ModelMetrics = pd.DataFrame()

print("\n" + "=" * 80)
print(f"CORA Validation Metrics ({Var_process})")
print("=" * 80)
print(ModelMetrics.to_string())
print("=" * 80)

# Save metrics to CSV
metrics_file = os.path.join(dir_out, f"CORA_Validation_Metrics_{Var_process}.csv")
ModelMetrics.to_csv(metrics_file, index=False)
print(f"\nMetrics saved to: {metrics_file}")

# ===============================================================================
# %% Create Scatter Plots
# ===============================================================================
print("\nGenerating scatter plots...")

fig, ax = plt.subplots(2, 4, figsize=[14, 8])
ax = ax.flatten()

plot_cnt = 0
for cnt, site_idx in enumerate(valid_indices):
    if plot_cnt >= 8:
        break

    site = Gauges[site_idx]
    pull_gauge = ds_gauge[site_idx]
    pull_model = ds_model[site_idx]

    # Get intersection
    t_intersect = t_intersect_save[cnt]
    pull_gauge_sel = pull_gauge.sel(time=t_intersect)
    pull_model_sel = pull_model.sel(time=t_intersect)

    # Remove NaN values
    valid_mask = ~(
        np.isnan(pull_gauge_sel[Var_process].values)
        | np.isnan(pull_model_sel[Var_process].values)
    )

    obs_vals = pull_gauge_sel[Var_process].values[valid_mask]
    model_vals = pull_model_sel[Var_process].values[valid_mask]

    ax[plot_cnt].plot(obs_vals, model_vals, "k.", label="CORA vs Obs", alpha=0.5)

    # Get data range for diagonal
    min_val = min(obs_vals.min(), model_vals.min())
    max_val = max(obs_vals.max(), model_vals.max())

    ax[plot_cnt].plot([min_val, max_val], [min_val, max_val], "k:", linewidth=1)
    ax[plot_cnt].set_xlim([min_val - 0.5, max_val + 0.5])
    ax[plot_cnt].set_ylim([min_val - 0.5, max_val + 0.5])

    if Y_pred[cnt] is not None:
        ax[plot_cnt].plot(obs_vals, Y_pred[cnt], "r", label="Fit", linewidth=2)

    ax[plot_cnt].grid(True, alpha=0.3)
    ax[plot_cnt].set_title(Gauge_name[site_idx], fontweight="bold")

    # Add metrics text
    rmse_val = ModelMetrics.loc[ModelMetrics["Gauge_ID"] == site, "RMSE"].values
    if len(rmse_val) > 0:
        ax[plot_cnt].text(
            0.05,
            0.95,
            f"RMSE = {rmse_val[0]:.3f} m\nR² = {ModelMetrics.loc[ModelMetrics['Gauge_ID'] == site, 'R2'].values[0]:.3f}",
            fontsize=8,
            bbox=dict(facecolor="yellow", alpha=0.5),
            verticalalignment="top",
            transform=ax[plot_cnt].transAxes,
        )

    if plot_cnt == 4:
        ax[plot_cnt].set_ylabel(f"CORA {Var_process.upper()} (m)", fontsize=10)
        ax[plot_cnt].set_xlabel(f"Obs. {Var_process.upper()} (m)", fontsize=10)

    plot_cnt += 1

# Remove extra subplots
for i in range(plot_cnt, 8):
    fig.delaxes(ax[i])

plt.tight_layout()
fig.savefig(
    os.path.join(dir_out, f"CORA_Validation_Scatter_{Var_process}.png"),
    dpi=150,
    bbox_inches="tight",
)
print(
    f"Scatter plot saved to: {os.path.join(dir_out, f'CORA_Validation_Scatter_{Var_process}.png')}"
)
plt.close()

# ===============================================================================
# %% Create Time Series Plots
# ===============================================================================
print("Generating time series plots...")

xlims = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15")]

fig, ax = plt.subplots(
    len(valid_indices), 1, sharex="col", figsize=[12, 4 * len(valid_indices)]
)

if len(valid_indices) == 1:
    ax = [ax]

for cnt, site_idx in enumerate(valid_indices):
    site = Gauges[site_idx]
    pull_gauge = ds_gauge[site_idx]
    pull_model = ds_model[site_idx]

    # Get intersection
    t_intersect = t_intersect_save[cnt]
    pull_gauge_sel = pull_gauge.sel(time=t_intersect)
    pull_model_sel = pull_model.sel(time=t_intersect)

    ax[cnt].plot(
        pull_gauge_sel["time"].values,
        pull_gauge_sel[Var_process].values,
        "b-",
        label="Observations",
        linewidth=1.5,
    )
    ax[cnt].plot(
        pull_model_sel["time"].values,
        pull_model_sel[Var_process].values,
        "r--",
        label="CORA",
        linewidth=1.5,
    )

    ax[cnt].grid(True, alpha=0.3)
    ax[cnt].set_title(f"{Gauge_name[site_idx]} (ID: {site})", fontweight="bold")
    ax[cnt].set_xlim(xlims)

    if cnt == len(valid_indices) - 1:
        ax[cnt].set_xlabel("Time", fontsize=10)

    if cnt == len(valid_indices) // 2:
        ax[cnt].set_ylabel(f"{Var_process.upper()} (m)", fontsize=10)

    if cnt == 0:
        ax[cnt].legend(loc="upper right")

plt.tight_layout()
fig.savefig(
    os.path.join(dir_out, f"CORA_Validation_TimeSeries_{Var_process}.png"),
    dpi=150,
    bbox_inches="tight",
)
print(
    f"Time series plot saved to: {os.path.join(dir_out, f'CORA_Validation_TimeSeries_{Var_process}.png')}"
)
plt.close()

# ===============================================================================
# %% Create Taylor Diagram
# ===============================================================================
# Normalized by observed std so all gauges share one diagram (reference at r=1)
print("Generating Taylor diagram...")

r_max = max(1.5, np.nanmax(std_ratio[valid_indices]) * 1.1)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111, projection="polar")
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_rlim(0, r_max)

corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
ax.set_thetagrids(
    np.degrees(np.arccos(corr_ticks)), labels=[f"{c:g}" for c in corr_ticks]
)
ax.text(
    np.radians(45), r_max * 1.12, "Correlation", rotation=-45, ha="center", va="center"
)
ax.set_xlabel("Normalized Standard Deviation")

# Centered RMS difference contours (normalized)
theta_grid, r_grid = np.meshgrid(
    np.linspace(0, np.pi / 2, 200), np.linspace(0, r_max, 200)
)
rms_grid = np.sqrt(1 + r_grid**2 - 2 * r_grid * np.cos(theta_grid))
contours = ax.contour(
    theta_grid,
    r_grid,
    rms_grid,
    levels=np.arange(0.25, 2, 0.25),
    colors="0.6",
    linestyles="--",
    linewidths=0.8,
)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

ax.plot(np.linspace(0, np.pi / 2, 100), np.ones(100), "k:", linewidth=1)
ax.plot(0, 1, "k*", markersize=14, label="Observations")

colors = plt.cm.tab10(np.arange(len(valid_indices)))
for cnt, site_idx in enumerate(valid_indices):
    ax.plot(
        np.arccos(corr_coef[site_idx]),
        std_ratio[site_idx],
        "o",
        markersize=10,
        color=colors[cnt],
        label=Gauge_name[site_idx],
    )

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=9)
ax.set_title(f"CORA vs Tide Gauge: {Var_process.upper()}", fontweight="bold")

fig.savefig(
    os.path.join(dir_out, f"CORA_Validation_Taylor_{Var_process}.png"),
    dpi=150,
    bbox_inches="tight",
)
print(
    f"Taylor diagram saved to: {os.path.join(dir_out, f'CORA_Validation_Taylor_{Var_process}.png')}"
)
plt.close()

print("\n" + "=" * 80)
print("CORA Validation Complete!")
print("=" * 80)
print(f"Results saved to: {dir_out}")
print(f"Metrics file: {metrics_file}")
