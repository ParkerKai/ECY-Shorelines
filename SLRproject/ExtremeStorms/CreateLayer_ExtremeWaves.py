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
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import xarray as xr
from glob import glob
import sys
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
        _clean_1d_series,
        _percentile_band_xr,
        pot_threshold_forward_stop_xr_debug,
    )
    from Kai_PerfTracker import PerfTracker
except ImportError as e:
    raise ImportError(f"Could not import module from {module_dir}: {e}")

# ===============================================================================
# User Inputs
# ===============================================================================
# dir_in = r"\\tsclient\N\SEA\Kai\Data\Cosmos\Waves\000"
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Waves\Waves\000"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Waves\Cosmos_Extremes_Wave"

# label describing this run configuration (VM vs local, network vs local drive, ...)
PERF_DIR = os.path.join(dir_out, "perf_logs")
N_WORKERS = 8
RUN_TAG = f"vm_TempDrive_{N_WORKERS}workers"
LOG_FILE = os.path.join(
    PERF_DIR,
    f"dask_{RUN_TAG}_{datetime.now():%Y%m%d_%H%M%S}.log",
)

RPs = np.array([1, 5, 10, 25, 50, 100])


# ===============================================================================
# Spin up a Dask cluster for parallel processing
# ===============================================================================


def main(perf):
    os.makedirs(PERF_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Dask run; log file: %s", LOG_FILE)

    with perf.stage("start_dask_cluster"):
        cluster = LocalCluster(
            processes=True,
            n_workers=N_WORKERS,
            threads_per_worker=1,
            silence_logs=logging.INFO,
        )  # tune n_workers to your CPU
        client = Client(cluster)
    print("Dask dashboard:", cluster.dashboard_link)  # http://127.0.0.1:8787/status
    logger.info("Dask dashboard: %s", cluster.dashboard_link)
    # (Processes commonly outperform threads for Python-heavy workloads.) [3](https://distributed.dask.org/en/stable/efficiency.html?highlight=workers%20threads)[12](https://dask-local.readthedocs.io/en/latest/setup/single-distributed.html)

    # ===============================================================================
    # Load the Cosmos Regional data
    # ===============================================================================

    with perf.stage("glob_input_files"):
        files = glob(os.path.join(dir_in, "*.nc"))

    save = []
    for i, file in enumerate(files):
        print(f"Processing file: {file}")

        # These files are multi-GB; keep Hs lazy and compute one file at a time.
        with xr.open_dataset(file, engine="h5netcdf") as ds:
            with perf.stage(f"read_file_{i:03d}"):
                da = ds["Hs"].chunk({"time": -1, "station": 2})
                lat = ds["lat"].load()
                lon = ds["lon"].load()

            with perf.stage(f"compute_file_{i:03d}"):
                results = compute_return_intervals_forwardstop_map(
                    da,
                    RPs=RPs,
                    r="72h",
                    l=10,
                    method="both",
                    fitter_method="mle",
                    n_boot=25,
                )
                results["lat"] = lat
                results["lon"] = lon
                results = results.compute()

        save.append(results)

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

    with perf.stage("concat"):
        final_ds = xr.concat(save, dim="station")

    with perf.stage("compute_extremes"):
        try:
            final_ds = final_ds.compute()  # single compute across all files  [4]
        except Exception:
            logger.exception("Dask compute_extremes failed")
            try:
                for worker, lines in client.get_worker_logs().items():
                    logger.error(
                        "Recent logs from worker %s:\n%s",
                        worker,
                        "\n".join(f"{level}: {message}" for level, message in lines),
                    )
            except Exception:
                logger.exception("Could not collect Dask worker logs after failure")
            raise

    # ===============================================================================
    # Turn a subset into a geodataframe
    # ===============================================================================

    # Build GeoDataFrame from final_ds (point geometry from lon/lat)
    with perf.stage("build_geodataframe"):
        gdf = gpd.GeoDataFrame(
            final_ds.to_dataframe().reset_index(),  # or select a subset of columns
            geometry=gpd.points_from_xy(final_ds["lon"].values, final_ds["lat"].values),
            crs="EPSG:4326",
        )

    # Write once at the end (stage to local SSD path)
    with perf.stage("write_shapefile"):
        gdf.to_file(
            os.path.join(dir_out, "Cosmos_Extremes_Wave.shp"), engine="pyogrio"
        )  # fast bulk I/O

    with perf.stage("shutdown_dask"):
        client.close()
        cluster.close()


if __name__ == "__main__":  # required for reliable spawn-based multiprocessing
    with PerfTracker(
        "CreateLayer_ExtremeWavesWLs", tag=RUN_TAG, out_dir=PERF_DIR, live_save=True
    ) as perf:
        main(perf)
