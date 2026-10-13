"""
Created on 07/06/2026

This script loads all Wave/WL datasets
  Cosmos               : CM

It then calculates extremes.
It then outputs the data a shapefile


@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# ===============================================================================
# Import Modules
# ===============================================================================

import os
import logging
import numpy as np
import xarray as xr
from glob import glob
import sys
from dask import compute as dask_compute
from dask.distributed import Client, LocalCluster
import geopandas as gpd

# ===============================================================================
# Functions
# ===============================================================================

# Absolute path to the directory containing the target file
module_dir = r"C:\Users\kaip461\GitHub\Kai_Python\General_Functions"
sys.path.insert(0, module_dir)

# Now you can import your module
try:
    from POT_Extremes_XR import (
        compute_return_intervals_forwardstop_map,
    )
except ImportError as e:
    raise ImportError(f"Could not import module from {module_dir}: {e}")

# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"N:\SEA\Kai\Data\Cosmos\WaterLevels\WaterLevels_Combined_000"
dir_out = r"N:\SEA\Kai\Data\SLR_Atlas"

N_WORKERS = 8

RPs = np.array([1, 5, 10, 25, 50, 100])
EXPORT_EXTREME_DATASET = True


def _slice_at_extreme_indices(ds, extreme_indices_ds):
    indices = extreme_indices_ds["extreme_time_index"]
    valid = indices >= 0
    selected = ds.isel(time=indices.where(valid, 0))

    for name, variable in selected.variables.items():
        if "station" in variable.dims and "event" in variable.dims:
            selected[name] = variable.where(valid)

    selected["pot_threshold"] = extreme_indices_ds["pot_threshold"]
    selected["n_events"] = extreme_indices_ds["n_events"]
    selected.attrs.update(extreme_indices_ds.attrs)
    selected.attrs["event_selection"] = (
        "Source dataset sampled at each station's declustered extreme time points"
    )

    if "time" in selected.coords:
        for attr in ("units", "calendar", "long_name"):
            selected["time"].attrs.pop(attr, None)

    return selected


# ===============================================================================
# Spin up a Dask cluster for parallel processing
# ===============================================================================


def main():
    cluster = LocalCluster(
        processes=True,
        n_workers=N_WORKERS,
        threads_per_worker=1,
        silence_logs=logging.INFO,
    )
    client = Client(cluster)
    print("Dask dashboard:", cluster.dashboard_link)  # http://127.0.0.1:8787/status
    # (Processes commonly outperform threads for Python-heavy workloads.) [3](https://distributed.dask.org/en/stable/efficiency.html?highlight=workers%20threads)[12](https://dask-local.readthedocs.io/en/latest/setup/single-distributed.html)

    # ===============================================================================
    # Load the Cosmos Regional data
    # ===============================================================================

    files = glob(os.path.join(dir_in, "*.nc"))

    save = []
    save_extreme_datasets = []
    for i, file in enumerate(files):
        print(f"Processing file: {file}")

        with xr.open_dataset(file, engine="h5netcdf") as ds:
            da = ds["waterlevel"].chunk({"time": -1, "station": 2})
            lat = ds["lat"].load()
            lon = ds["lon"].load()

            computed = compute_return_intervals_forwardstop_map(
                da,
                RPs=RPs,
                r="72h",
                l=30,
                method="both",
                fitter_method="mle",
                n_boot=200,
                return_extremes=EXPORT_EXTREME_DATASET,
            )
            if EXPORT_EXTREME_DATASET:
                results, extreme_indices = computed
            else:
                results = computed

            results["lat"] = lat
            results["lon"] = lon
            if EXPORT_EXTREME_DATASET:
                results, extreme_indices = dask_compute(results, extreme_indices)
                extreme_ds = _slice_at_extreme_indices(ds, extreme_indices).load()
            else:
                results = results.compute()

        save.append(results)
        if EXPORT_EXTREME_DATASET:
            save_extreme_datasets.append(extreme_ds)

        # st = 19  # pick a station index
        # series_i = ds["waterlevel"].isel(station=st).chunk({"time": -1})
        # series_i = _clean_1d_series(series_i, drop_zeros=False)

        # lo, hi = 80.0, 98.0
        # t_down, t_up = _percentile_band_xr(series_i, lo, hi)

        # u_i, dbg = pot_threshold_forward_stop_xr_debug(
        #     series_i, r="72h", thresh_down=t_down, thresh_up=t_up,
        #     l=30, alpha=0.05, strategy="gt", fitter_method="mle", n_boot=100
        # )

        # print("u_i:", u_i)
        # print("error:", dbg["error"])
        # print("first p_vals:", dbg["p_vals"][:10])
        # print("first xi:", dbg["xi"][:10], "first sigma:", dbg["sigma"][:10])

    final_ds = xr.concat(save, dim="station")
    if EXPORT_EXTREME_DATASET:
        final_extreme_ds = xr.concat(save_extreme_datasets, dim="station")

    final_ds = final_ds.compute()

    if EXPORT_EXTREME_DATASET:
        final_extreme_ds.to_netcdf(
            os.path.join(dir_out, "Cosmos_Extremes_WL_Events.nc"),
            engine="h5netcdf",
        )

    # ===============================================================================
    # Turn a subset into a geodataframe
    # ===============================================================================

    # Build GeoDataFrame from final_ds (point geometry from lon/lat)
    final_df = final_ds.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
        final_df,
        geometry=gpd.points_from_xy(final_df["lon"], final_df["lat"]),
        crs="EPSG:4326",
    )

    # Write once at the end (stage to local SSD path)
    gdf.to_file(os.path.join(dir_out, "Cosmos_Extremes_WL.shp"), engine="pyogrio")

    client.close()
    cluster.close()


if __name__ == "__main__":  # required for reliable spawn-based multiprocessing
    main()
