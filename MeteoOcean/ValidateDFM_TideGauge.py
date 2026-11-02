# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:56:45 2025

This script compares DFM outputs vs Tidegauge data

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
import scipy
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib
from glob import glob

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_model = r"Y:\PS_Cosmos\02_models\DFM_Regional"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\validation"

Gauges = [
    "9443090",  # Neah Bay
    "9444090",  # Port Anageles
    "9444900",  # Port Townsend
    "9447130",  # SEattle
    "9446484",  # Tacoma
    "9449880",  # Friday Harbor
    "9449424",
]  # Cherry POint

# sites=['cdip162']

Gauge_name = [
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Seattle",
    "Tacoma",
    "Friday Harbor",
    "Cherry Point",
]


# Variable to process (calculate stats and make plots)
Var_process = "tide"  # 'wl', 'tide', 'ntr'

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime

from Kai_ModelValidation import TaylorDiagram, mean_absolute_difference_corrected
from Kai_ModelValidation import ModelStat_Var, intersect_Var
from Kai_ModelValidation import bias, SpiderPlot
from Kai_Timeseries import gappy_interp


def Read_TG_Mat(dir_in_TG, gauge_ID):
    file_in = os.path.join(dir_in_TG, f"NOAA_TG_{gauge_ID}.mat")
    dat = scipy.io.loadmat(file_in)

    t_obs = matlab2datetime(dat["date_obs"].flatten(), "min")
    t_pre = matlab2datetime(dat["date_pred"].flatten(), "min")

    # interpolate tides onto the observed timeseries. Do this rather than intersection
    # To preserve the higher temporal resolution of the observations
    tide = gappy_interp(
        t_obs.to_numpy(),
        t_pre.to_numpy(),
        dat["wl_pre"].flatten(),
        maxgap=np.timedelta64(1, "D"),  # 1 day
        extrapolate=False,
    )

    data_vars = {
        "wl": (
            ["time"],
            dat["wl_obs"].flatten(),
            {"units": "m", "long_name": "Water Level Observed"},
        ),
        "tide": (
            ["time"],
            tide,
            {"units": "m", "long_name": "Tide (Water Level Predicted)"},
        ),
        "ntr": (
            ["time"],
            dat["wl_obs"].flatten() - tide,
            {
                "units": "m",
                "long_name": "Non-Tidal Residual (Observed vs. predicted Water Levels)",
            },
        ),
        # 'quality':(['t_obs'],  dat['quality'].flatten(),
        #              {'units': 'm',
        #               'standard_name':'Sigma'}),
        # 'sigma':(['t_obs'],  dat['sigma'].flatten(),
        #              {'units': 'None',
        #               'standard_name':'Quality Code'}),
    }

    # define coordinates
    coords = {"time": (["time"], t_obs, {"standard_name": "time observed"})}
    # create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "station": dat["station_info"][0][0][0][0][0][0],
            "Datum": dat["station_info"][0][0][0][0][0][2],
            "Units": dat["station_info"][0][0][0][0][0][3],
            "TimeZone": dat["station_info"][0][0][0][0][0][4],
        },
    )

    return ds


# ===============================================================================
# %% Read in the Tide Gauge data
# ===============================================================================
ds_gauge = []
for cnt, gauge_ID in enumerate(Gauges):
    ds = Read_TG_Mat(dir_in_TG, gauge_ID)

    # Conversion to NAVD88 (from VDATUM). Download of NAVD unavailable for gauge.
    # Port Townsend
    if gauge_ID == "9444900":
        ds["wl"] = ds["wl"] + 1.190
        ds["tide"] = ds["tide"] + 1.190
        ds.attrs["Datum"] = "NAVD88"

    # Friday Harbor
    if gauge_ID == "9449880":
        ds["wl"] = ds["wl"] + 1.271
        ds["tide"] = ds["tide"] + 1.271
        ds.attrs["Datum"] = "NAVD88"

    # Cherry POint
    if gauge_ID == "9449424":
        ds["wl"] = ds["wl"] + 1.317
        ds["tide"] = ds["tide"] + 1.317
        ds.attrs["Datum"] = "NAVD88"

    # Can't use resample as it doesn't preserve gaps
    # ds_sel = ds.resample(time="10min").interpolate("linear")

    # Resample to 10min
    rounded_dt = pd.to_datetime(ds["time"][0].values).round("D")
    t_start = np.datetime64(rounded_dt)

    rounded_dt = pd.to_datetime(ds["time"][-1].values).round("D")
    t_end = np.datetime64(rounded_dt)

    t_new = np.arange(t_start, t_end, np.timedelta64(10, "m"), dtype="datetime64[ns]")

    wl = gappy_interp(
        t_new,
        ds["time"].values,
        ds["wl"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    tide = gappy_interp(
        t_new,
        ds["time"].values,
        ds["tide"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    ntr = gappy_interp(
        t_new,
        ds["time"].values,
        ds["ntr"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "wl": (["time"], wl, {"units": "m", "long_name": "waterlevel"}),
        "tide": (["time"], tide, {"units": "m", "long_name": "tide"}),
        "ntr": (["time"], ntr, {"units": "m", "long_name": "non-tidal residual"}),
    }

    coords = {"time": (["time"], t_new, {"standard_name": "time"})}

    ds_sel = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_gauge.append(ds_sel)


# ===============================================================================
# %% Read in the Model Data
# ===============================================================================

# Load the data

files = glob(
    os.path.join(
        dir_in_model, "ERA5", "ERA5", "ERA5_000", "Results_Combined", "DFM_wl*"
    )
)
ds_full = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

files = glob(
    os.path.join(
        dir_in_model, "ERA5", "ERA5_tidal_Results", "Results_Combined", "000", "DFM_wl*"
    )
)
ds_tidal = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

station_id = ds_full["station"].values

ds_model = []
for cnt, gauge_ID in enumerate(Gauges):
    # FInd index for station_id
    stat_id = next(
        (cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None
    )

    # Subset to station and from 1995 on (since thats when gauge records start)
    ds_sel = ds_full.isel(station=stat_id)
    ds_sel = ds_sel.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel["waterlevel"] = ds_sel["waterlevel"] / 10000

    # Add tide from tide only runs
    ds_sel_tide = ds_tidal.isel(station=stat_id)
    ds_sel_tide = ds_sel_tide.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel_tide["waterlevel"] = ds_sel_tide["waterlevel"] / 10000
    ds_sel = ds_sel.assign(tide=ds_sel_tide["waterlevel"])

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "ntr": (
            ["time"],
            ds_sel["waterlevel"].data - ds_sel["tide"].data,
            {"units": "m", "long_name": "Non-Tidal Residual"},
        )
    }

    coords = {"time": (["time"], ds_sel["time"].data, {"standard_name": "time"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # create dataset
    ds_sel = ds_sel.assign(ntr=ds["ntr"])

    # Change variable names
    ds_sel = ds_sel.rename({"waterlevel": "wl"})

    ds_model.append(ds_sel)


# Export location of gauge
# lon = np.full([len(Gauges)],np.nan)
# lat = np.full([len(Gauges)],np.nan)
# stat_id = np.full([len(Gauges)],'                            ')
# for cnt,gauge_ID in enumerate(Gauges):
#     pull_model = ds_model[cnt]
#     lon[cnt] = pull_model['lon'][0].values
#     lat[cnt] = pull_model['lat'][0].values
#     stat_id[cnt] = pull_model['station'].values

# geom = gpd.points_from_xy(lon,lat, crs="EPSG:4326")
# df = pd.DataFrame({'Stat_ID':stat_id})
# Stations = gpd.GeoDataFrame(df,geometry=geom)
# Stations.to_file('Y:\PS_Cosmos\GIS\general\GaugeLocations.shp')


# ===============================================================================
# %% Model validation Statistics
# ===============================================================================


metrics = [root_mean_squared_error, r2_score, mean_absolute_difference_corrected, bias]

t_intersect_save = []
Y_pred = []
slope = np.full((len(Gauges)), np.nan)
model_perf = np.full((len(Gauges), len(metrics)), np.nan)

for cnt1, gauge_ID in enumerate(Gauges):
    print(f"Processing Gauge {gauge_ID}")

    pull_gauge = ds_gauge[cnt1]
    pull_model = ds_model[cnt1]

    # Temporal intersection (and save to save time for the future)
    t_intersect = intersect_Var(pull_model, pull_gauge, Var_process)

    t_intersect_save.append(t_intersect)

    pull_gauge = pull_gauge.sel(time=t_intersect)
    pull_model = pull_model.sel(time=t_intersect)

    # Calculate performance metric for models
    for cnt2, metric in enumerate(metrics):
        model_perf[cnt1, cnt2] = ModelStat_Var(
            pull_model, pull_gauge, Var_process, None, metric, intersect=False
        ).squeeze()

    # Fit regression line
    model = LinearRegression()
    X = pull_gauge[Var_process].values
    Y = pull_model[Var_process].values
    model.fit(X.reshape(-1, 1), Y)
    slope[cnt1] = model.coef_.squeeze()
    Y_pred.append(model.predict(X.reshape(-1, 1)))


ModelMetrics = pd.DataFrame(
    {
        "Gauge": Gauges,
        "RMSE": model_perf[:, 0],
        "R2": model_perf[:, 1],
        "MADc": model_perf[:, 2],
        "AbsBias": model_perf[:, 3],  # np.absolute(
        "Slope": slope,
    }
)


# ===============================================================================
# %% Plots
# ===============================================================================

fig, ax = matplotlib.pyplot.subplots(2, 4, figsize=[8, 6])
ax = ax.flatten()
for cnt, site in enumerate(Gauges):
    pull_gauge = ds_gauge[cnt]
    pull_model = ds_model[cnt]

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    pull_gauge = pull_gauge[Var_process].sel(time=t_intersect)
    pull_model = pull_model[Var_process].sel(time=t_intersect)

    ax[cnt].plot(pull_gauge, pull_model, "k.", label="Model V1", alpha=0.5)

    ax[cnt].plot([-2.5, 4.5], [-2.5, 4.5], "k:")
    ax[cnt].set_xlim([-3, 4.5])
    ax[cnt].set_ylim([-3, 4.5])

    ax[cnt].plot(pull_gauge, Y_pred[cnt], "r", label="Model Fit")

    ax[cnt].grid()
    ax[cnt].set_title(site)

    ax[cnt].text(
        -2,
        4,
        f"RMSE = {ModelMetrics['RMSE'][cnt].round(2)}",
        fontsize=8,
        bbox=dict(facecolor="grey", alpha=0.5),
    )

    if cnt == 4:
        ax[cnt].set_ylabel(f"Model {Var_process} (m)")
        ax[cnt].set_xlabel(f"Obs. {Var_process} (m)")

    if (cnt != 0) or (cnt != 4):
        ax[cnt].set_yticklabels([])

    if (cnt != 4) or (cnt != 5) or (cnt != 6) or (cnt != 7):
        ax[cnt].set_xticklabels([])

    if cnt == 6:
        ax[cnt].legend(loc="lower right")

fig.savefig(
    os.path.join(dir_out, f"DFM_Validation_Scatter_{Var_process}.tif"),
    dpi=800,
    bbox_inches="tight",
)

##############################################################################

xlims = [pd.Timestamp("2015-01-01-"), pd.Timestamp("2015-01-14")]


fig, ax = matplotlib.pyplot.subplots(len(Gauges), 1, sharex="col", figsize=[8, 6])

for cnt, site in enumerate(Gauges):
    pull_gauge = ds_gauge[cnt]
    pull_model = ds_model[cnt]

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    ax[cnt].plot(
        pull_gauge["time"].sel(time=t_intersect),
        pull_gauge[Var_process].sel(time=t_intersect),
        "k",
        label="Gauge",
    )
    ax[cnt].plot(
        pull_model["time"].sel(time=t_intersect),
        pull_model[Var_process].sel(time=t_intersect),
        "r",
        label="Model",
    )

    ax[cnt].grid()

    ax[cnt].set_title(site)
    ax[cnt].set_xlim(xlims)

    if cnt != 6:
        ax[cnt].set_xticklabels([])


ax[4].set_ylabel("WaterLevel (m)")
ax[4].legend()

fig.savefig(
    os.path.join(dir_out, f"DFM_Validation_TS_{Var_process}.tif"),
    dpi=800,
    bbox_inches="tight",
)

############################### Taylor Plot#####################################

# Var_list = ['Hsig','Tp','Dir']

# for Var in Var_list:

#     subplot_ind = [141,142,143,144]
#     fig, ax = matplotlib.pyplot.subplots(1, 4)
#     ax = ax.flatten()

#     for cnt,site in enumerate(sites):
#         # Reference dataset
#         t_intersect = intersect_Var([modelv1.sel(station = site),
#                                      modelv2.sel(station = site)],
#                                      buoy.sel(station=site),'Hsig')

#         data = buoy[Var].sel(time=t_intersect,station=site)
#         refstd = np.nanstd(data,ddof=1)           # Reference standard deviation


#         # Generate models
#         m1 = modelv1[Var].sel(time=t_intersect,station=site)    # Model 1
#         m2 =  modelv2[Var].sel(time=t_intersect,station=site) # Model 2

#         # Compute stddev and correlation coefficient of models
#         samples = np.array([ [np.nanstd(m,ddof=1), np.corrcoef(data, m)[0, 1]]
#                              for m in (m1,m2)])


#         # Taylor diagram
#         dia = TaylorDiagram(refstd, fig=fig,rect=subplot_ind[cnt], label="Reference",
#                             srange=(0.5, 1.5))

#         colors =  matplotlib.pyplot.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))


#         # Add the models to Taylor diagram
#         for i, (stddev, corrcoef) in enumerate(samples):
#             dia.add_sample(stddev, corrcoef,
#                            marker='o', ms=10, ls='',
#                            mfc=colors[i], mec=colors[i],
#                            label=f"Model {i+1}")

#         # Add grid
#         dia.add_grid()

#         # Add RMS contours, and label them
#         contours = dia.add_contours(colors='0.5')
#         matplotlib.pyplot.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

#         if cnt == 1:
#             # Add a figure legend
#             fig.legend(dia.samplePoints,
#                        [ p.get_label() for p in dia.samplePoints ],
#                        numpoints=1, prop=dict(size='small'), loc='upper right')


#         ax[cnt].set_title(site)
#         ax[cnt].set_xticklabels([])
#         ax[cnt].set_yticklabels([])

#     fig.set_size_inches(12, 4)
#     fig.savefig(os.path.join(dir_out,f'Taylor_{Var}.tif'),  dpi=800,
#             bbox_inches='tight')


############################## Radar Plot #######################################


fig, ax = matplotlib.pyplot.subplots(
    1,
    subplot_kw=dict(polar=True),
    figsize=(8, 8),
)

SpiderPlot(
    ax, ModelMetrics, id_column="Gauge", title="Model Validation Metrics", padding=1.01
)


fig.savefig(
    os.path.join(dir_out, f"DFM_Validation_Radar_{Var_process}.tif"),
    dpi=800,
    bbox_inches="tight",
)
