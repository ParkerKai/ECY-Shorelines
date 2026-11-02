# -*- coding: utf-8 -*-
"""
CORA vs Tide Gauge Validation Script (Simplified - Metrics Only)

This script compares CORA model outputs vs Tide Gauge observational data
for water levels only.

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
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.linear_model import LinearRegression
from glob import glob

# Disable matplotlib GUI backend
import matplotlib
matplotlib.use('Agg')

print("Matplotlib backend set to:", matplotlib.get_backend())

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
]

Gauge_name = [
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Seattle",
    "Tacoma",
    "Friday Harbor",
    "Cherry Point",
]

# Variable to process (only water levels)
Var_process = "wl"

# ===============================================================================
# %% Define helper functions
# ===============================================================================

def mean_absolute_difference_corrected(obs, pred):
    """Mean absolute difference corrected"""
    return np.nanmean(np.abs(obs - pred))


def bias(obs, pred):
    """Bias (mean residual)"""
    return np.nanmean(pred - obs)


def Read_CORA_Parquet(file_path):
    """Read CORA data from parquet file"""
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


def Read_TG_Parquet(dir_in_tg, gauge_ID, start_date=None, end_date=None):
    """Read tide gauge data from parquet files"""
    gauge_dir = os.path.join(dir_in_tg, f"NOAA_TG_{gauge_ID}")
    parquet_file = os.path.join(gauge_dir, "combined_resolved.parquet")

    if not os.path.exists(parquet_file):
        print(f"Warning: {parquet_file} not found")
        return None

    df = pd.read_parquet(parquet_file)

    # Time column is 't'
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

    # Create xarray Dataset
    data_vars = {
        "wl": (
            ["time"],
            df["v_obs"].values,
            {"units": "m", "long_name": "Water Level Observed"},
        ),
    }

    coords = {"time": (["time"], df["time"].values, {"standard_name": "time observed"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"station_id": gauge_ID})

    return ds


def resample_to_hourly_xr(ds):
    """Resample to hourly using xarray's built-in resampling"""
    # Convert to DataFrame for easier resampling
    df = ds["wl"].to_dataframe().reset_index()
    
    # Set time as index
    df.set_index("time", inplace=True)
    
    # Resample to hourly: interpolate to fill gaps, then forward fill for leading gaps
    df_hourly = df.resample("h").interpolate(method="linear")
    # Forward fill to handle gaps at the start and within the data
    df_hourly = df_hourly.ffill()
    
    # Convert back to xarray
    ds_hourly = xr.Dataset(
        {"wl": (["time"], df_hourly["wl"].values)},
        coords={"time": df_hourly.index.values}
    )
    
    return ds_hourly


# ===============================================================================
# %% Read in CORA data to determine time ranges
# ===============================================================================
print("=" * 80)
print("CORA Water Level Validation")
print("=" * 80)
print("\nStep 1: Reading CORA model data...")
cora_datasets = []
cora_time_ranges = []

for cnt, gauge_ID in enumerate(Gauges):
    print(f"  {gauge_ID} ({Gauge_name[cnt]})...", end=" ", flush=True)

    # Find CORA file for this gauge
    cora_files = glob(os.path.join(dir_in_cora, f"{gauge_ID}*.parquet"))

    if len(cora_files) == 0:
        print("SKIP (no file)")
        cora_datasets.append(None)
        cora_time_ranges.append(None)
        continue

    cora_file = cora_files[0]

    try:
        ds = Read_CORA_Parquet(cora_file)
        cora_datasets.append(ds)
        t_min = pd.to_datetime(ds["time"].values.min())
        t_max = pd.to_datetime(ds["time"].values.max())
        cora_time_ranges.append((ds["time"].values.min(), ds["time"].values.max()))
        print(f"OK ({t_min.year}-{t_max.year})")
    except Exception as e:
        print(f"ERROR ({e})")
        cora_datasets.append(None)
        cora_time_ranges.append(None)

print(f"\nLoaded {sum(1 for x in cora_datasets if x is not None)} CORA datasets")

# ===============================================================================
# %% Read Tide Gauge data
# ===============================================================================
print("\nStep 2: Reading Tide Gauge observation data...")
ds_gauge = []

for cnt, gauge_ID in enumerate(Gauges):
    if cora_time_ranges[cnt] is None:
        print(f"  {gauge_ID}: SKIP (no CORA data)")
        ds_gauge.append(None)
        continue

    print(f"  {gauge_ID} ({Gauge_name[cnt]})...", end=" ", flush=True)

    # Get time range from CORA data
    t_start = pd.to_datetime(cora_time_ranges[cnt][0])
    t_end = pd.to_datetime(cora_time_ranges[cnt][1])

    try:
        ds = Read_TG_Parquet(
            dir_in_tg,
            gauge_ID,
            start_date=t_start.strftime("%Y-%m-%d"),
            end_date=t_end.strftime("%Y-%m-%d"),
        )

        if ds is None:
            print("SKIP (not found)")
            ds_gauge.append(None)
            continue

        print(f"OK ({len(ds['time'])} points)", end=" ", flush=True)

        # Resample to hourly
        print("resampling...", end=" ", flush=True)
        ds = resample_to_hourly_xr(ds)
        ds_gauge.append(ds)
        print(f"OK ({len(ds['time'])} hourly points)")

    except Exception as e:
        print(f"ERROR ({e})")
        ds_gauge.append(None)

print(f"Loaded {len([d for d in ds_gauge if d is not None])} gauge datasets")

# ===============================================================================
# %% Calculate validation metrics
# ===============================================================================
print("\nStep 3: Calculating validation metrics...")

metrics = [root_mean_squared_error, r2_score, mean_absolute_difference_corrected, bias]

Y_pred_list = []
slope = np.full((len(Gauges)), np.nan)
model_perf = np.full((len(Gauges), len(metrics)), np.nan)
valid_indices = []

for cnt1, gauge_ID in enumerate(Gauges):
    if cora_datasets[cnt1] is None or ds_gauge[cnt1] is None:
        continue

    print(f"  {gauge_ID} ({Gauge_name[cnt1]})...", end=" ", flush=True)

    try:
        pull_gauge = ds_gauge[cnt1]
        pull_model = cora_datasets[cnt1]

        # Find temporal overlap using index-based approach to avoid memory issues
        t_model_min = pd.to_datetime(pull_model["time"].values[0])
        t_model_max = pd.to_datetime(pull_model["time"].values[-1])
        t_gauge_min = pd.to_datetime(pull_gauge["time"].values[0])
        t_gauge_max = pd.to_datetime(pull_gauge["time"].values[-1])

        print(f"({t_gauge_min.year}-{t_gauge_max.year})", end=" ", flush=True)

        t_start = max(t_model_min, t_gauge_min)
        t_end = min(t_model_max, t_gauge_max)

        if t_start > t_end:
            print("SKIP (no overlap)")
            continue

        # Select data within the overlapping time range
        model_sel = pull_model.sel(time=slice(t_start, t_end))
        gauge_sel = pull_gauge.sel(time=slice(t_start, t_end))

        if len(model_sel["time"]) == 0 or len(gauge_sel["time"]) == 0:
            print("SKIP (empty selection)")
            continue

        print(f"(m:{len(model_sel['time'])},g:{len(gauge_sel['time'])})", end=" ", flush=True)

        # Check for NaNs
        gauge_valid = np.isfinite(gauge_sel["wl"].values).sum()
        model_valid = np.isfinite(model_sel["wl"].values).sum()
        
        print(f"(g_valid:{gauge_valid},m_valid:{model_valid})", end=" ", flush=True)

        # Use a simpler approach: just align the data using xarray's built-in capabilities
        # Both datasets are already hourly, just merge them
        print(f"(merge...)", end=" ", flush=True)
        
        # Simply select both to their common time range
        t_start = max(model_sel["time"].values.min(), gauge_sel["time"].values.min())
        t_end = min(model_sel["time"].values.max(), gauge_sel["time"].values.max())
        
        # Reindex both to common times using a simpler method
        model_time_common = model_sel.sel(time=slice(t_start, t_end))["wl"].values
        gauge_time_common = gauge_sel.sel(time=slice(t_start, t_end))["wl"].values
        
        # Both should now have same length (both hourly)
        if len(model_time_common) != len(gauge_time_common):
            print(f"SKIP (length mismatch: {len(model_time_common)} vs {len(gauge_time_common)})")
            continue
        
        # Remove NaNs
        valid_mask = ~(np.isnan(gauge_time_common) | np.isnan(model_time_common))

        if np.sum(valid_mask) < 100:
            print(f"SKIP (only {np.sum(valid_mask)} valid points)")
            continue

        obs_data = gauge_time_common[valid_mask]
        model_data_vals = model_time_common[valid_mask]

        # Calculate metrics
        valid_indices.append(cnt1)

        model_perf[cnt1, 0] = root_mean_squared_error(obs_data, model_data_vals)
        model_perf[cnt1, 1] = r2_score(obs_data, model_data_vals)
        model_perf[cnt1, 2] = mean_absolute_difference_corrected(obs_data, model_data_vals)
        model_perf[cnt1, 3] = bias(obs_data, model_data_vals)

        # Fit regression
        model_reg = LinearRegression()
        model_reg.fit(obs_data.reshape(-1, 1), model_data_vals)
        slope[cnt1] = model_reg.coef_.squeeze()
        Y_pred_list.append(model_reg.predict(obs_data.reshape(-1, 1)))

        print(f"OK (n={len(obs_data)}, RMSE={model_perf[cnt1, 0]:.3f}, R²={model_perf[cnt1, 1]:.3f})")

    except Exception as e:
        print(f"ERROR ({e})")
        import traceback
        traceback.print_exc()

# ===============================================================================
# %% Create results dataframe
# ===============================================================================
print("\nStep 4: Creating results...")

if len(valid_indices) > 0:
    ModelMetrics = pd.DataFrame(
        {
            "Gauge_ID": [Gauges[i] for i in valid_indices],
            "Gauge_Name": [Gauge_name[i] for i in valid_indices],
            "RMSE (m)": [model_perf[i, 0] for i in valid_indices],
            "R2": [model_perf[i, 1] for i in valid_indices],
            "MADc (m)": [model_perf[i, 2] for i in valid_indices],
            "Bias (m)": [model_perf[i, 3] for i in valid_indices],
            "Slope": [slope[i] for i in valid_indices],
        }
    )

    print("\n" + "=" * 80)
    print("CORA Water Level Validation Metrics")
    print("=" * 80)
    print(ModelMetrics.to_string(index=False))
    print("=" * 80)

    # Save metrics to CSV
    metrics_file = os.path.join(dir_out, f"CORA_Validation_Metrics_{Var_process}.csv")
    ModelMetrics.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to: {metrics_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Number of gauges: {len(valid_indices)}")
    print(f"  Mean RMSE: {ModelMetrics['RMSE (m)'].mean():.4f} m")
    print(f"  Mean R2: {ModelMetrics['R2'].mean():.4f}")
    print(f"  Mean Bias: {ModelMetrics['Bias (m)'].mean():.4f} m")

else:
    print("\nERROR: No valid gauges to process!")

print("\n" + "=" * 80)
print("Validation Complete!")
print("=" * 80)
