# -*- coding: utf-8 -*-
# Script to calculate ASL estimates for Washington State's coastal waters
# via coupling NOAA tide gauge data and VLM estimates
# (Python translation of the original MATLAB script)

import math
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.io import savemat
import statsmodels.api as sm

# ----------------------------
# set parameters
# ----------------------------

stnums = [
    9440910,
    9442396,
    9443090,
    9444090,
    9444900,
    9449424,
    9449880,
    9447130,
    9446484,
    9441102,
]
stnams = [
    "Toke Point",
    "La Push",
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Cherry Point",
    "Friday Harbor",
    "Seattle",
    "Tacoma",
    "Westport",
]

# vlm rates in mm/yr for each tide gauge derived from Newton et al., Supplemental Table S5 and S6
# (rates extracted from Newton et al. interpolated points near tide gauges)
vlms = [
    0.035921,
    -0.881595,
    2.80886,
    1.075373,
    -0.934305,
    1.254333,
    0.400442,
    -0.648102,
    -1.278814,
    1.632204,
]
vlmuc = [
    0.622934,
    0.829389,
    0.990746,
    0.53471,
    0.848552,
    0.589758,
    0.698111,
    0.470093,
    0.73966,
    0.841024,
]

stdt = datetime(1895, 1, 1)
yr = 1895  # beginning of the time-series

datum = "msl"  # use 'mhhw','mhw','mlw','mllw' or 'navd'
tm_sys = "gmt"  # can be switched to 'lst' or 'lst_ldt'
units = "metric"
enddt = datetime.now()

# build output matrix (mslout)
nomos = int(math.ceil((((enddt - stdt).days) / 365.0) * 12))
mslout = np.full((len(stnums) + 5, nomos), np.nan)

# fill in date and time rows
mo = 1
for i in range(nomos):
    mslout[0, i] = yr  # Year row
    mslout[1, i] = mo  # Month row
    # MATLAB datenum -> matplotlib date number
    mslout[2, i] = mdates.date2num(datetime(yr, mo, 15))
    mo += 1
    if mo == 13:
        mo = 1
        yr += 1

# ----------------------------
# Figure: Raw Monthly Data
# ----------------------------
plt.figure(figsize=(12, 4), facecolor="w")
ax = plt.subplot(1, 3, 1)

for i in range(len(stnums)):
    url = (
        "https://tidesandcurrents.noaa.gov/api/datagetter"
        "?product=monthly_mean&application=NOS.COOPS.TAC.WL&begin_date={}&end_date={}&datum={}&station={}&time_zone={}&units={}&format=csv"
    ).format(
        stdt.strftime("%Y%m%d"),
        enddt.strftime("%Y%m%d"),
        datum,
        str(stnums[i]),
        tm_sys,
        units,
    )

    # read CSV
    df = pd.read_csv(url)
    # coerce numeric columns, in case of blanks
    for col in ["Year", "Month", "Highest", "MHHW", "MSL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    yrs = df["Year"].to_numpy(dtype=float)
    mos = df["Month"].to_numpy(dtype=int)
    msl = df["MSL"].to_numpy(dtype=float)

    # remove VLM trend from msl: mm/yr -> m/yr (divide by 1000)
    msl_corr = msl + ((yrs - yrs[0]) * (vlms[i] / 1000.0))

    # calculate and remove the seasonal trend using 1999–2008 standard
    seasmsl = np.full(12, np.nan)
    for j in range(12):
        ind = np.where((yrs >= 1999) & (yrs <= 2008) & (mos == (j + 1)))[0]
        seasmsl[j] = np.nanmean(msl_corr[ind]) if len(ind) > 0 else np.nan

    # add data to mslout
    for j in range(len(yrs)):
        ind = np.where((mslout[0, :] == yrs[j]) & (mslout[1, :] == mos[j]))[0]
        if len(ind) > 0:
            mslout[i + 3, ind[0]] = msl_corr[j] - seasmsl[mos[j] - 1]

    # plot raw monthly data (meters)
    dnum = [mdates.date2num(datetime(int(y), int(m), 15)) for y, m in zip(yrs, mos)]
    ax.plot(dnum, msl, "-", linewidth=1)

ax.legend(stnams, loc="upper left")
ax.set_ylabel("meters (relative to MSL 1983–2001 epoch)")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_title("Raw Monthly Data")

# calculate regional mean ASL and standard error
mslout[-2, :] = np.nanmean(mslout[4:-2, :], axis=0)
counts = np.sum(~np.isnan(mslout[4:-2, :]), axis=0)
mslout[-1, :] = np.nanstd(mslout[4:-2, :], axis=0) / np.sqrt(np.maximum(counts, 1))

# ----------------------------
# Plot VLM + Seasonal removed (per station)
# ----------------------------
ax2 = plt.subplot(1, 3, 2)
for r in range(4, mslout.shape[0] - 2):
    ax2.plot(mslout[2, :], mslout[r, :], linewidth=1)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator())
)
ax2.legend(stnams, loc="upper left")
ax2.set_title("VLM and Seasonal Pattern Removed")

# ----------------------------
# Epoch (20-year mean, centered ±9 years)
# ----------------------------
ax3 = plt.subplot(1, 3, 3)
yrs_unique = np.unique(mslout[0, :].astype(int))
current_year = enddt.year

epoch = []
for idx in range(9, np.where(yrs_unique == current_year)[0][0] - 9 + 1):
    ymid = int(yrs_unique[idx])
    ind = np.where((mslout[0, :] >= ymid - 9) & (mslout[0, :] <= ymid + 9))[0]
    val = np.nanmean(np.nanmean(mslout[4:-2, :][:, ind]))
    epoch.append([mdates.date2num(datetime(ymid, 6, 1)), val, ymid])
epoch = np.array(epoch) if len(epoch) > 0 else np.empty((0, 3))

yrsave = []
for y in yrs_unique:
    ind = np.where(mslout[0, :] == y)[0]
    if len(ind) > 10:
        mean_cal = np.nanmean(np.nanmean(mslout[4:-2, :][:, ind]))
        # Water year average: shift index by -3 (Oct–Mar window logic, following original)
        ind_wy = ind - 3
        ind_wy = ind_wy[(ind_wy >= 0) & (ind_wy < mslout.shape[1])]
        mean_wy = (
            np.nanmean(np.nanmean(mslout[4:-2, :][:, ind_wy]))
            if len(ind_wy) > 10
            else np.nan
        )
    else:
        mean_cal = np.nan
        mean_wy = np.nan

    # epoch average for given year if exists
    epoch_match = epoch[epoch[:, 2] == y, 1] if epoch.size else np.array([])
    epoch_val = epoch_match[0] if len(epoch_match) else np.nan

    yrsave.append(
        [mdates.date2num(datetime(int(y), 6, 1)), mean_cal, int(y), epoch_val, mean_wy]
    )
yrsave = np.array(yrsave)

# save yrsave
savemat("AnnualAbsoluteSLData4WA.mat", {"yrsave": yrsave})

# plot regional mean monthly ASL + epoch line
ax3.plot(mslout[2, :], mslout[-2, :], color=[0.5, 0.5, 0.5])
if epoch.size:
    ax3.plot(epoch[:, 0], epoch[:, 1], "ks-", linewidth=2)
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
ax3.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator())
)
ax3.legend(["Regional ASL", "Tidal Epoch Average"], loc="upper left")
ax3.set_title("Mean Monthly ASL + 20 yr average")
plt.tight_layout()

# ----------------------------
# Projections (Kopp 2014 RCP8.5)
# ----------------------------
projs = np.array(
    [
        [-0.048, -0.010, 0.008, 0.029, 0.075, 0.128, 0.142, 0.166, 0.203, 0.241],
        [0.015, 0.067, 0.094, 0.116, 0.184, 0.251, 0.276, 0.303, 0.350, 0.408],
        [0.096, 0.157, 0.191, 0.221, 0.312, 0.402, 0.434, 0.472, 0.550, 0.664],
        [0.134, 0.233, 0.286, 0.334, 0.474, 0.619, 0.671, 0.734, 0.870, 1.234],
        [0.191, 0.333, 0.407, 0.476, 0.677, 0.889, 0.967, 1.063, 1.306, 2.033],
        [0.303, 0.476, 0.565, 0.648, 0.894, 1.164, 1.268, 1.402, 1.791, 2.857],
        [0.398, 0.616, 0.729, 0.835, 1.150, 1.504, 1.648, 1.833, 2.401, 4.032],
        [0.494, 0.757, 0.895, 1.024, 1.426, 1.883, 2.070, 2.315, 3.091, 5.176],
        [0.565, 0.885, 1.054, 1.214, 1.708, 2.279, 2.514, 2.838, 3.880, 6.642],
        [0.611, 1.014, 1.227, 1.424, 2.045, 2.763, 3.069, 3.475, 4.803, 8.259],
        [0.855, 1.185, 1.374, 1.557, 2.173, 2.934, 3.278, 3.787, 5.465, 9.612],
        [0.959, 1.332, 1.560, 1.774, 2.496, 3.412, 3.827, 4.450, 6.494, 11.543],
        [1.030, 1.460, 1.711, 1.960, 2.803, 3.886, 4.375, 5.115, 7.562, 13.567],
        [1.084, 1.572, 1.858, 2.134, 3.108, 4.368, 4.946, 5.818, 8.707, 15.908],
        [1.090, 1.660, 1.990, 2.319, 3.448, 4.920, 5.580, 6.594, 9.933, 18.361],
        [1.031, 1.705, 2.092, 2.473, 3.775, 5.458, 6.228, 7.407, 11.247, 20.982],
        [1.013, 1.778, 2.210, 2.632, 4.114, 6.025, 6.902, 8.243, 12.644, 23.612],
        [1.040, 1.869, 2.352, 2.831, 4.485, 6.631, 7.619, 9.138, 14.090, 26.190],
        [1.066, 1.945, 2.470, 2.985, 4.816, 7.215, 8.322, 10.030, 15.581, 29.181],
        [1.015, 2.001, 2.576, 3.156, 5.185, 7.854, 9.070, 10.957, 17.187, 32.143],
    ]
)

yrs_decadal = np.arange(2010, 2201, 10)
probs = [99, 95, 90, 83, 50, 17, 10, 5, 1, 0.1]  # (not used directly below)

# ----------------------------
# Figure 2: Observed
# ----------------------------
plt.figure(facecolor="w")
plt.plot(
    mslout[0, :] + (mslout[1, :] / 12.0), mslout[-2, :] * 3.28084, color=[0.7, 0.7, 0.7]
)
plt.plot(yrsave[:, 2], yrsave[:, 1] * 3.28084, "ko-", linewidth=1)
plt.ylabel("MSL; feet relative to 1990–2010 average")
plt.legend(["Regional ASL (monthly)", "Calendar Year Average ASL"])
plt.ylim([-1, 1])
plt.title("Observed")
plt.gca().tick_params(labelsize=12, width=1.5)

# ----------------------------
# Figure 3: Observed vs. projected (Kopp 2014) + AR6 panel
# ----------------------------
plt.figure(facecolor="w", figsize=(12, 4))

# Left panel: Kopp 2014 RCP8.5
axL = plt.subplot(1, 2, 1)
axL.plot(
    mslout[0, :] + (mslout[1, :] / 12.0), mslout[-2, :] * 3.28084, color=[0.7, 0.7, 0.7]
)
axL.plot(
    np.r_[2000, yrs_decadal], np.r_[0, projs[:, 4]], "r-", linewidth=2
)  # median (as in MATLAB, no unit conversion here)
axL.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 3]], "r-", linewidth=1)  # 17th
axL.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 5]], "r-", linewidth=1)  # 83rd
# epoch relative to 2000 in feet
if epoch.size:
    axL.plot(
        epoch[:, 2],
        (epoch[:, 1] - epoch[epoch[:, 2] == 2000, 1][0]) * 3.28084,
        "k-",
        linewidth=3,
    )
axL.plot(yrsave[:, 2], yrsave[:, 1] * 3.28084, "ko-", linewidth=1)
axL.set_xlim([1900, 2050])
axL.set_ylabel("MSL; feet relative to 1990–2010 average")
axL.legend(
    [
        "Regional ASL (monthly)",
        "median",
        "17th",
        "83rd",
        "Regional ASL 20 yr average",
        "Calendar Year Average ASL",
    ],
    loc="upper left",
)
axL.set_title("Observed vs. Kopp 2014 RCP8.5")

# Right panel: AR6 SSP5-8.5 (Friday Harbor example ranges, medium confidence)
axR = plt.subplot(1, 2, 2)
axR.plot(
    mslout[0, :] + (mslout[1, :] / 12.0), mslout[-2, :] * 3.28084, color=[0.7, 0.7, 0.7]
)

projsar6 = np.array(
    [
        [-0.009, 0.016, 0.029, 0.063],
        [0.007, 0.03, 0.051, 0.092],
        [0.03, 0.053, 0.089, 0.14],
        [0.056, 0.086, 0.142, 0.215],
        [0.076, 0.115, 0.192, 0.286],
    ]
)
yrs_ar6 = np.arange(2020, 2051, 10)

axR.plot(yrs_ar6, projsar6[2, :] * 3.281, "r-", linewidth=2)  # median
axR.plot(yrs_ar6, projsar6[1, :] * 3.281, "r-", linewidth=1)  # 17th
axR.plot(yrs_ar6, projsar6[3, :] * 3.281, "r-", linewidth=1)  # 83rd
if epoch.size:
    axR.plot(
        epoch[:, 2],
        (epoch[:, 1] - epoch[epoch[:, 2] == 2000, 1][0]) * 3.28084,
        "k-",
        linewidth=3,
    )
axR.plot(yrsave[:, 2], yrsave[:, 1] * 3.28084, "ko-", linewidth=1)
axR.set_xlim([1900, 2050])
axR.set_ylabel("MSL; feet relative to 1990–2010 average")
axR.set_title("Observed vs. AR6 SSP5-8.5")
plt.tight_layout()

# ----------------------------
# Alternate Figure 3 (Kopp 2014, various percentiles)
# ----------------------------
plt.figure(facecolor="w")
if epoch.size:
    plt.plot(
        epoch[:, 2],
        (epoch[:, 1] - epoch[epoch[:, 2] == 2000, 1][0]) * 3.28084,
        "k-",
        linewidth=3,
    )
plt.plot(yrsave[:, 2], yrsave[:, 1] * 3.28084, "k-", linewidth=1)

plt.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 4]], "r-", linewidth=2)
plt.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 3]], "r-", linewidth=1)
plt.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 5]], "r-", linewidth=1)
plt.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 0]], "r:", linewidth=1)
plt.plot(np.r_[2000, yrs_decadal], np.r_[0, projs[:, 8]], "r:", linewidth=1)

plt.xlim([1900, 2050])
plt.ylabel("MSL; feet relative to 1990–2010 average")
plt.legend(
    [
        "Regional ASL (monthly)",
        "median",
        "17th",
        "83rd",
        "Regional ASL 20 yr average",
        "Calendar Year Average ASL",
    ],
    loc="upper left",
)
plt.title("Observed vs. Kopp 2014 RCP8.5")

# ----------------------------
# Figure: Absolute Water Level Anomaly (Seasonal Pattern Removed)
# ----------------------------
plt.figure(facecolor="w", figsize=(10, 6))
ax_top = plt.subplot(2, 1, 1)
ax_top.plot(mslout[2, :], mslout[-2, :], color=[0.5, 0.5, 0.5])
ax_top.plot(yrsave[:, 0] - 1, yrsave[:, 3], "k.", markersize=20)
ax_top.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_top.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax_top.xaxis.get_major_locator())
)
ax_top.set_ylabel("meters")
ax_top.set_title("Absolute Water Level Anomaly (Seasonal Pattern Removed)")
ax_top.legend(["Monthly Average", "Water Year Average"])

ax_bot = plt.subplot(2, 1, 2)
for i in range(12):
    ind = np.where(mslout[1, :] == (i + 1))[0]
    vals = mslout[-2, ind]
    if len(ind) > 0 and np.any(~np.isnan(vals)):
        val = np.nanmax(vals)
        ind1 = ind[np.nanargmax(vals)]
        ax_bot.plot(i + 1, val, "r.", markersize=30)
        ax_bot.text(
            i + 1 + 0.1,
            val,
            datetime.fromordinal(int(mslout[2, ind1])).strftime("%Y"),
            color="r",
        )
ax_bot.set_xlabel("Month of the Year")
ax_bot.set_xlim([0, 13])
ax_bot.set_ylabel("meters")
ax_bot.set_title("Record Monthly Average Anomaly from Long-Term Average")
plt.tight_layout()

# export epoch array for use in ASL_Projection_Tool.m
savemat("ASLRecord4WA.mat", {"epoch": epoch})

# ----------------------------
# Coastal specific (feet plots)
# ----------------------------
plt.figure(facecolor="w", figsize=(10, 8))
ax_c1 = plt.subplot(2, 1, 1)

for i in range(len(stnums)):
    url = (
        "https://tidesandcurrents.noaa.gov/api/datagetter"
        "?product=monthly_mean&application=NOS.COOPS.TAC.WL&begin_date={}&end_date={}&datum={}&station={}&time_zone={}&units={}&format=csv"
    ).format(
        stdt.strftime("%Y%m%d"),
        enddt.strftime("%Y%m%d"),
        datum,
        str(stnums[i]),
        tm_sys,
        units,
    )
    df = pd.read_csv(url)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["MSL"] = pd.to_numeric(df["MSL"], errors="coerce")

    yrs = df["Year"].to_numpy(dtype=float)
    mos = df["Month"].to_numpy(dtype=int)
    msl = df["MSL"].to_numpy(dtype=float)

    msl_corr = msl + ((yrs - yrs[0]) * (vlms[i] / 1000.0))

    seasmsl = np.full(12, np.nan)
    for j in range(12):
        ind = np.where((yrs >= 1999) & (yrs <= 2008) & (mos == (j + 1)))[0]
        seasmsl[j] = np.nanmean(msl_corr[ind]) if len(ind) > 0 else np.nan

    for j in range(len(yrs)):
        ind = np.where((mslout[0, :] == yrs[j]) & (mslout[1, :] == mos[j]))[0]
        if len(ind) > 0:
            mslout[i + 3, ind[0]] = msl_corr[j] - seasmsl[mos[j] - 1]

    dnum = [mdates.date2num(datetime(int(y), int(m), 15)) for y, m in zip(yrs, mos)]
    ax_c1.plot(dnum, msl * 3.28, "-", linewidth=1)

ax_c1.legend(stnams, loc="upper left")
ax_c1.set_ylabel("Feet relative to MSL (1983–2001 epoch)")
ax_c1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_c1.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax_c1.xaxis.get_major_locator())
)
ax_c1.set_title("Raw Monthly Data")

ax_c2 = plt.subplot(2, 1, 2)
for r in range(4, mslout.shape[0] - 2):
    ax_c2.plot(mslout[2, :], mslout[r, :] * 3.28, linewidth=1)
ax_c2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_c2.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax_c2.xaxis.get_major_locator())
)
ax_c2.legend(stnams, loc="upper left")
ax_c2.set_title("Vertical Land Movement and Seasonal Pattern Removed")
ax_c2.set_ylabel("Feet relative to MSL (1983–2001 epoch)")
plt.tight_layout()

# ----------------------------
# Station-specific panels + regression (ft/century)
# ----------------------------
plt.figure(facecolor="w", figsize=(12, 8))
for i in range(len(stnums)):
    url = (
        "https://tidesandcurrents.noaa.gov/api/datagetter"
        "?product=monthly_mean&application=NOS.COOPS.TAC.WL&begin_date={}&end_date={}&datum={}&station={}&time_zone={}&units={}&format=csv"
    ).format(
        stdt.strftime("%Y%m%d"),
        enddt.strftime("%Y%m%d"),
        datum,
        str(stnums[i]),
        tm_sys,
        units,
    )
    df = pd.read_csv(url)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["MSL"] = pd.to_numeric(df["MSL"], errors="coerce")

    yrs = df["Year"].to_numpy(dtype=float)
    mos = df["Month"].to_numpy(dtype=int)
    msl = df["MSL"].to_numpy(dtype=float)

    ax_sp = plt.subplot(2, 3, i + 1)
    dnum = [mdates.date2num(datetime(int(y), int(m), 15)) for y, m in zip(yrs, mos)]
    ax_sp.plot(dnum, msl * 3.28, "k-")
    if i == 0:
        ax_sp.set_ylabel("Feet relative to MSL (1983–2001 epoch)")
    ax_sp.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_sp.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_sp.xaxis.get_major_locator())
    )
    ax_sp.set_title(f"{stnams[i]}")

plt.subplot(2, 3, (4, 5, 6))
for r in range(4, mslout.shape[0] - 2):
    plt.plot(mslout[2, :], mslout[r, :] * 3.28)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator())
)
plt.title("Vertical Land Movement and Seasonal Pattern Removed")
plt.ylabel("Feet relative to MSL (1983–2001 epoch)")

legstr = []
for i in range(len(stnams)):
    y = mslout[i + 3, :] * 3.28
    x = mslout[2, :]
    mask = ~np.isnan(y) & ~np.isnan(x)
    if np.sum(mask) > 2:
        X = np.column_stack([np.ones(np.sum(mask)), x[mask]])
        model = sm.OLS(y[mask], X).fit()
        b = model.params
        bint = model.conf_int()
        legstr.append(
            f"{stnams[i]}: {b[1] * 365 * 100:.3f}   {(((bint[1, 1] - bint[1, 0]) / 2.0) * 365 * 100):.3f} ft/century"
        )
    else:
        legstr.append(f"{stnams[i]}: NaN   NaN ft/century")
plt.legend(legstr, loc="upper left")
plt.tight_layout()

# ----------------------------
# Calculate change (boxplot & errorbar) using regional mean (row 14)
# ----------------------------
ind1 = np.where((mslout[0, :] >= 1900) & (mslout[0, :] <= 1910))[0]
ind2 = np.where((mslout[0, :] >= 1966) & (mslout[0, :] <= 1976))[0]
ind3 = np.arange(
    mslout.shape[1] - len(ind2), mslout.shape[1]
)  # last decade window, mirroring MATLAB

plt.figure(facecolor="w", figsize=(10, 6))
ax_bx = plt.subplot(2, 1, 1)
data_box = [
    (mslout[-1, ind1] * 39.37),
    (mslout[-1, ind2] * 39.37),
    (mslout[-1, ind3] * 39.37),
]
ax_bx.boxplot(
    data_box,
    labels=[
        "Start of Record: 1900–1910",
        "SMA Established: 1961–1971",
        "Recent Decade",
    ],
    patch_artist=False,
    sym="ko",
)
ax_bx.set_title("Absolute Sea Level in Washington: Summary of Monthly Averages")
ax_bx.set_ylabel("Inches")

ax_er = plt.subplot(2, 1, 2)
means = [np.nanmean(data_box[0]), np.nanmean(data_box[1]), np.nanmean(data_box[2])]
ses = [
    np.nanstd(data_box[0]) / math.sqrt(len(ind1)),
    np.nanstd(data_box[1]) / math.sqrt(len(ind2)),
    np.nanstd(data_box[2]) / math.sqrt(len(ind3)),
]
xpos = [1905, 1971, np.nanmean(mslout[0, ind3])]
ax_er.errorbar(xpos, means, ses, fmt="k.", markersize=12)
ax_er.set_ylabel("Inches")
ax_er.set_title("Mean and Standard Error")
plt.tight_layout()

# Show all figures
plt.show()
