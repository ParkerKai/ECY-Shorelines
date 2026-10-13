# -*- coding: utf-8 -*-
"""
Created on 05/01/2026

This script loads all SLR datasets
  Miller 2018 SLR data : M1
  Federal Task force   : FTF
  Cosmos               : CM
  
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from glob import glob
import geopandas as gpd
from shapely.geometry import Point
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Sequence

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


# ===============================================================================
# Functions
# ===============================================================================


def load_RSLR_table(file, sheet, header_row, nrows=16):
    # Read the block
    df = pd.read_excel(
        file, sheet_name=sheet, engine="openpyxl", header=header_row, nrows=nrows
    )

    # Drop first row, drop unnamed column
    df = df.drop(0).reset_index(drop=True)
    df = df.drop(columns=["Unnamed: 1"], errors="ignore")

    # Rename, convert, set index
    df = df.rename(columns={"19 year period centered on:": "Date"})
    df["Date"] = df["Date"].astype("Int16")
    df = df.set_index("Date")

    return df


def load_RSLR_overview(file, sheet):

    overview = pd.read_excel(
        file, sheet_name="Site Overview", engine="openpyxl", header=None, nrows=1
    )

    line1 = overview.iloc[0, 0]

    match = re.search(r"([0-9.]+)N,\s*([0-9.]+)W", line1)
    if match:
        lat_lon = pd.DataFrame(
            [[float(match.group(1)), -float(match.group(2))]], columns=["Lat", "Lon"]
        )  # West → negative
    else:
        print("Coordinates not found in the expected format.")
        lat_lon = pd.DataFrame([[np.nan, np.nan]], columns=["Lat", "Lon"])

    twl = pd.read_excel(
        file, sheet_name="Site Overview", engine="openpyxl", header=12, nrows=1
    )

    return lat_lon, twl

def ceil_to_grid(
    vals: ArrayLike,
    grid: ArrayLike,
    *,
    clamp: bool = True,
    return_type: str = "auto"
) -> ArrayLike:
    """
    Round each value in `vals` *up* to the next value in `grid` (a fixed, ascending set).
    
    This is the vectorized equivalent of ceiling to a custom grid:
      - If val is exactly on a grid point -> returns that same grid value.
      - If val lies between two grid points -> returns the next-higher grid value (ceiling).
      - If val > max(grid):
          - If clamp=True -> returns max(grid).
          - If clamp=False -> returns NaN.
      - If val < min(grid) -> returns min(grid) (because the ceiling of anything below
        the smallest grid value is that smallest value). If you prefer NaN in this case,
        set clamp=False and post-filter values below min(grid).

    NaNs in `vals` are preserved as NaN in the output.

    Parameters
    ----------
    vals : array-like
        Values to be rounded up. Accepts NumPy arrays, pandas Series, or sequences.
    grid : array-like
        The allowed grid values. Will be converted to a float array and *sorted ascending*.
        Duplicates are allowed but unnecessary.
    clamp : bool, default True
        Whether to clamp values greater than max(grid) to max(grid).
        If False, values above max(grid) become NaN.
    return_type : {"auto", "numpy", "pandas"}, default "auto"
        Controls the return container type:
        - "auto": returns a pandas Series if `vals` is a Series; otherwise a NumPy array.
        - "numpy": always returns a NumPy array.
        - "pandas": returns a pandas Series (index preserved if `vals` is a Series;
          otherwise a default RangeIndex Series).

    Returns
    -------
    array-like
        Rounded-up values, same shape as `vals`, with NaNs preserved.

    Notes
    -----
    - Complexity: O(N log M), where N = len(vals), M = len(grid).
      Implemented via `np.searchsorted`, which performs a binary search over `grid`.
    - `grid` is sorted internally; ensure it represents your intended ordering.
    - This function performs *ceiling* relative to the provided grid. If you need
      floor or nearest behavior, consider a companion function or extend `mode`.

    Examples
    --------
    >>> allowed = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3], dtype=float)
    >>> vals = np.array([0.0, 0.42, 1.0, 1.49, 3.2, np.nan])
    >>> ceil_to_grid_vec(vals, allowed)
    array([0.  , 0.5 , 1.  , 1.5 , 3.  ,  nan])

    # Without clamping, values above the grid become NaN:
    >>> ceil_to_grid_vec(vals, allowed, clamp=False)
    array([0.  , 0.5 , 1.  , 1.5 ,  nan,  nan])

    # Works with pandas Series and preserves index:
    >>> s = pd.Series(vals, index=['a','b','c','d','e','f'])
    >>> ceil_to_grid_vec(s, allowed, return_type="pandas")
    a    0.0
    b    0.5
    c    1.0
    d    1.5
    e    3.0
    f    NaN
    dtype: float64
    """
    # Convert vals and grid to NumPy float arrays
    # (preserve Series index for "auto"/"pandas" return)
    is_series = isinstance(vals, pd.Series)
    if is_series:
        vals_arr = vals.to_numpy(dtype=float)
        out_index = vals.index
    else:
        vals_arr = np.asarray(vals, dtype=float)
        out_index = None

    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.size == 0:
        raise ValueError("`grid` must contain at least one value.")
    grid_arr = np.sort(grid_arr)

    # Compute insertion indices: first index i such that grid[i] >= vals
    idxs = np.searchsorted(grid_arr, vals_arr, side='left')

    # Handle values above max(grid)
    if clamp:
        # clip into valid range [0, len(grid)-1]
        idxs = np.clip(idxs, 0, grid_arr.size - 1)
        out = grid_arr[idxs]
    else:
        # create NaN for above-range; below-range (idx=0) is valid ceil -> grid[0]
        out = np.full_like(vals_arr, np.nan, dtype=float)
        # valid positions are those strictly within [0, len(grid)-1]
        valid_mask = idxs < grid_arr.size
        out[valid_mask] = grid_arr[idxs[valid_mask]]

    # Preserve NaNs from input
    nan_mask = np.isnan(vals_arr)
    if nan_mask.size > 1:  
        out[nan_mask] = np.nan

    # Return container type
    if return_type == "numpy" or (return_type == "auto" and not is_series):
        return out
    else:
        # return pandas Series
        return pd.Series(out, index=out_index)



# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\SLR\LocalTables"
dir_out = r"N:\SEA\Kai\Figures\RSLR_Miller2018"

ProbExceed = 50


Cosmos_levels = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3], dtype=float)
FTF_levels = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=float)



# ===============================================================================
# Load the data
# ===============================================================================

files = glob(os.path.join(dir_in, "*.xlsx"))

lat_lon_list = []
twl_list = []
RCP45_mag = []
RCP45_prob = []
RCP85_mag = []
RCP85_prob = []
for file in files:
    lat_lon, twl = load_RSLR_overview(file, sheet="Site Overview")

    lat_lon_list.append(lat_lon)
    twl_list.append(twl)

    RCP45_mag.append(load_RSLR_table(file, "RCP 4.5", header_row=7))
    RCP45_prob.append(load_RSLR_table(file, "RCP 4.5", header_row=29))

    RCP85_mag.append(load_RSLR_table(file, "RCP 8.5", header_row=7))
    RCP85_prob.append(load_RSLR_table(file, "RCP 8.5", header_row=29))


lat_lon = pd.concat(lat_lon_list, ignore_index=True)
twl = pd.concat(twl_list, ignore_index=True)


#### Read the Shoreline data for plotting ######

cnty = gpd.read_file(r"N:\SEA\Kai\General\WA_County_Boundaries.shp")
# cnty.set_crs(epsg=32610, inplace=True, allow_override=True)
cnty.to_crs(epsg=4236, inplace=True)


shoreline = gpd.read_file(r"N:\SEA\Kai\General\SMA_MarineShoreline.gpkg")
shoreline.to_crs(epsg=4236, inplace=True)

# ===============================================================================
# Turn a subset into a geodataframe
# ===============================================================================


# First, find cosmos year of occurence

df = {'M1_2030' : np.full(len(lat_lon), np.nan),
      'M1_2070' : np.full(len(lat_lon), np.nan),
      'M1_2100' : np.full(len(lat_lon), np.nan),
      'CM_2030' : np.full(len(lat_lon), np.nan),
      'CM_2070' : np.full(len(lat_lon), np.nan),
      'CM_2100' : np.full(len(lat_lon), np.nan),
      'FTF_2030' : np.full(len(lat_lon), np.nan),
      'FTF_2070' : np.full(len(lat_lon), np.nan),
      'FTF_2100' : np.full(len(lat_lon), np.nan),
}

for index, row in lat_lon.iterrows():
    pull = RCP45_mag[index][ProbExceed]

    x = pull.index.to_numpy()
    y = pull.to_numpy()

    # ------- Year of Cosmos Level Occurrence  ------
    # df['year_25'][index] = np.interp(0.25, x, y).round(0)
    # df['year_50'][index] = np.interp(0.50, x, y).round(0)
    # df['year_100'][index] = np.interp(1.00, x, y).round(0)
    # df['year_150'][index] = np.interp(1.50, x, y).round(0)

    # ------- Cosmos Sea Level nearest to Miller, 2018 (rounded up)  ------
    df['CM_2030'][index] = ceil_to_grid(np.interp(2030, x, y), Cosmos_levels, clamp=True).round(2)
    df['CM_2070'][index] = ceil_to_grid(np.interp(2070, x, y), Cosmos_levels, clamp=True).round(2)
    df['CM_2100'][index] = ceil_to_grid(np.interp(2100, x, y), Cosmos_levels, clamp=True).round(2)

    # ------- Miller, 2018 RSLR  ------
    df['M1_2030'][index] = np.interp(2030, x, y).round(2)
    df['M1_2070'][index] = np.interp(2070, x, y).round(2)
    df['M1_2100'][index] = np.interp(2100, x, y).round(2)


    # ------- Federal Task Force nearest to Miller, 2018 (rounded up)  ------
    df['FTF_2030'][index] = ceil_to_grid(np.interp(2030, x, y), FTF_levels, clamp=True).round(2)
    df['FTF_2070'][index] = ceil_to_grid(np.interp(2070, x, y), FTF_levels, clamp=True).round(2)
    df['FTF_2100'][index] = ceil_to_grid(np.interp(2100, x, y), FTF_levels, clamp=True).round(2)





geometry = [Point(xy) for xy in zip(lat_lon["Lon"], lat_lon["Lat"])]

Out = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4236",
)


# ===============================================================================
# 2030 Plot
# ===============================================================================

# -----------------------------
# 1) Shared color scale (vmin/vmax)
# -----------------------------
metrics = ["M1_2030", "CM_2030", "FTF_2030"]
missing = [m for m in metrics if m not in Out.columns]
if missing:
    raise KeyError(f"Columns missing in 'Out': {missing}")

vmin = np.nanmin([Out[m].min() for m in metrics])
vmax = np.nanmax([Out[m].max() for m in metrics])
cmap = plt.get_cmap("viridis")
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# 2) Figure & axes (aligned sizes)
# -----------------------------
plt.style.use("seaborn-v0_8-white")  # clean, neutral base style
fig, axes = plt.subplots(
    1, 3, figsize=(12, 5),
    sharex=True, sharey=True,
    constrained_layout=True
)

titles = [
    "Miller (2018) – 2030, 50%",
    "Cosmos – 2030 best fit",
    "Federal Task Force – 2030 best fit"
]
panel_labels = ["(a)", "(b)", "(c)"]

# -----------------------------
# 3) Plot each panel (clean edges, equal aspect)
# -----------------------------
for ax, col, title, label in zip(axes, metrics, titles, panel_labels):
    # Counties as context (soft edges)
    cnty.plot(
        ax=ax, column="JURISDIC_2", cmap="Pastel2",
        edgecolor="#666", linewidth=0.4, alpha=0.30
    )

    # Overlay: RSLR polygons with shared scale and no edges
    Out.plot(
        ax=ax, column=col, cmap=cmap, norm=norm,
        edgecolor="none"
    )

    # Map extent in lon/lat (keep as per your original setup)
    ax.set_xlim(-125, -122)
    ax.set_ylim(46, 49)

    # Equal aspect to avoid distortion
    ax.set_aspect("equal", adjustable="box")

    # Title and panel label
    ax.set_title(title, fontsize=11, pad=6)
    ax.text(
        0.01, 0.98, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

# -----------------------------
# 4) Remove tick labels from ax2 & ax3
# -----------------------------
for ax in axes[1:]:
    ax.tick_params(
        labelbottom=False, labelleft=False,  # hide labels
        bottom=False, left=False            # hide tick marks
    )

# Optional: subtle grid only on the first panel
axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

# -----------------------------
# 5) Single shared colorbar (keeps panels same size)
# -----------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for older Matplotlib versions
cbar = fig.colorbar(
    sm, ax=axes, orientation="horizontal",
    fraction=0.05, pad=0.10
)
cbar.set_label("Relative sea-level rise (units) – 2030", fontsize=10)  # <-- set correct units
cbar.ax.tick_params(labelsize=9)

# -----------------------------
# 6) Save with lossless compression
# -----------------------------
out_path = os.path.join(dir_out, "2030_RSLRcompare.tiff")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved map to: {out_path}")


# ===============================================================================
# 2070 Plot
# ===============================================================================

# -----------------------------
# 1) Shared color scale (vmin/vmax)
# -----------------------------
metrics = ["M1_2070", "CM_2070", "FTF_2070"]
missing = [m for m in metrics if m not in Out.columns]
if missing:
    raise KeyError(f"Columns missing in 'Out': {missing}")

vmin = np.nanmin([Out[m].min() for m in metrics])
vmax = np.nanmax([Out[m].max() for m in metrics])
cmap = plt.get_cmap("viridis")
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# 2) Figure & axes (aligned sizes)
# -----------------------------
plt.style.use("seaborn-v0_8-white")  # clean, neutral base style
fig, axes = plt.subplots(
    1, 3, figsize=(12, 5),
    sharex=True, sharey=True,
    constrained_layout=True
)

titles = [
    "Miller (2018) – 2070, 50%",
    "Cosmos – 2070 best fit",
    "Federal Task Force – 2070 best fit"
]
panel_labels = ["(a)", "(b)", "(c)"]

# -----------------------------
# 3) Plot each panel (clean edges, equal aspect)
# -----------------------------
for ax, col, title, label in zip(axes, metrics, titles, panel_labels):
    # Counties as context (soft edges)
    cnty.plot(
        ax=ax, column="JURISDIC_2", cmap="Pastel2",
        edgecolor="#666", linewidth=0.4, alpha=0.30
    )

    # Overlay: RSLR polygons with shared scale and no edges
    Out.plot(
        ax=ax, column=col, cmap=cmap, norm=norm,
        edgecolor="none"
    )

    # Map extent in lon/lat (keep as per your original setup)
    ax.set_xlim(-125, -122)
    ax.set_ylim(46, 49)

    # Equal aspect to avoid distortion
    ax.set_aspect("equal", adjustable="box")

    # Title and panel label
    ax.set_title(title, fontsize=11, pad=6)
    ax.text(
        0.01, 0.98, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
    )

# -----------------------------
# 4) Remove tick labels from ax2 & ax3
# -----------------------------
for ax in axes[1:]:
    ax.tick_params(
        labelbottom=False, labelleft=False,  # hide labels
        bottom=False, left=False            # hide tick marks
    )

# Optional: subtle grid only on the first panel
axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

# -----------------------------
# 5) Single shared colorbar (keeps panels same size)
# -----------------------------
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for older Matplotlib versions
cbar = fig.colorbar(
    sm, ax=axes, orientation="horizontal",
    fraction=0.05, pad=0.10
)
cbar.set_label("Relative sea-level rise (units) – 2070", fontsize=10)  # <-- set correct units
cbar.ax.tick_params(labelsize=9)

# -----------------------------
# 6) Save with lossless compression
# -----------------------------
out_path = os.path.join(dir_out, "2070_RSLRcompare.tiff")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved map to: {out_path}")


# ===============================================================================
# Difference Plot
# ===============================================================================

# --- Prepare the positive difference metric ---
cm_vals = pd.to_numeric(Out["CM_2030"], errors="coerce")
m1_vals = pd.to_numeric(Out["M1_2030"], errors="coerce")
Out["CM_minus_M1_2030"] = (cm_vals - m1_vals)*100

# Sequential colormap (no divergence), starting at 0
vmin = 0.0
# use max; swap to percentile if you want robust scaling
vmax = np.nanmax(Out["CM_minus_M1_2030"])
cmap = plt.get_cmap("viridis")
norm = colors.Normalize(vmin=vmin, vmax=vmax)

plt.style.use("seaborn-v0_8-white")
fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

# Basemap (counties)
cnty.plot(
    ax=ax, column="JURISDIC_2", cmap="Pastel2",
    edgecolor="#666", linewidth=0.4, alpha=0.30
)

# Overlay: positive difference, clean edges
Out.plot(
    ax=ax, column="CM_minus_M1_2030", cmap=cmap, norm=norm,
    edgecolor="none"
)

# Extent and aspect
ax.set_xlim(-125, -122)
ax.set_ylim(46, 49)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Difference: Cosmos − Miller (2018), 2030", fontsize=11, pad=8)

# Subtle grid (optional)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

# Figure-level colorbar (sequential)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10)
cbar.set_label("CM_2030 − M1_2030 (cm)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

out_path = os.path.join(dir_out, "2030_CM_minus_M1.tiff")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved map to: {out_path}")






# ===============================================================================
# Aggregate by county
# ===============================================================================
# Spatial join: attach county name to each point
Out_with_county = gpd.sjoin(
    Out,
    cnty[["JURISDIC_2", "geometry"]],  # keep only name + geometry
    how="left",
    predicate="within",
)

# Rename column for clarity
Out_with_county.rename(columns={"JURISDIC_2": "County"}, inplace=True)

sns.set_theme(style="darkgrid", palette="deep")

fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=(10, 10))
sns.boxplot(ax=ax0, data=Out_with_county, x="County", y="M1_2030")
ax0.set_title("Distribution of 2030 Miller 2018 RSLR by County")
ax0.set_xticklabels([])
ax0.grid(True, which="both", axis="both", alpha=0.5)
ax0.set(xlabel=None)
ax0.plot([-1, 15], [0, 0], "k--", alpha = 0.5, label="SLR 0cm")
ax0.plot([-1, 15], [0.25, 0.25], "k--",alpha = 0.5,  label="SLR 25cm")
ax0.plot([-1, 15], [0.5, 0.5], "k--",alpha = 0.5,  label="SLR 50cm")
ax0.set_xlim([-0.5,14.5])

sns.boxplot(ax=ax1, data=Out_with_county, x="County", y="M1_2070")
ax1.set_title("Distribution of 2070 Miller 2018 RSLR by County")
ax1.set_xticklabels([])
ax1.grid(True, which="both", axis="both", alpha=0.5)
ax1.set(xlabel=None)
ax1.plot([-1, 15], [0.25, 0.25], "k--",alpha = 0.5,  label="SLR 25cm")
ax1.plot([-1, 15], [0.5, 0.5], "k--",alpha = 0.5,  label="SLR 50cm")
ax1.plot([-1, 15], [1, 1], "k--",alpha = 0.5,  label="SLR 100cm")
ax1.plot([-1, 15], [1.5, 1.5], "k--",alpha = 0.5,  label="SLR 150cm")
ax1.set_xlim([-0.5,14.5])


sns.boxplot(ax=ax2, data=Out_with_county, x="County", y="M1_2100")
ax2.set_title("Distribution of 2100 Miller 2018 RSLR by County")
ax2.grid(True, which="both", axis="both", alpha=0.5)
ax2.set(xlabel=None)
ax2.plot([-1, 15], [0.5, 0.5], "k--",alpha = 0.5,  label="SLR 50cm")
ax2.plot([-1, 15], [1, 1], "k--",alpha = 0.5,  label="SLR 100cm")
ax2.plot([-1, 15], [1.5, 1.5], "k--",alpha = 0.5,  label="SLR 150cm")
ax2.plot([-1, 15], [2.0, 2.0], "k--",alpha = 0.5,  label="SLR 200cm")
ax2.set_xlim([-0.5,14.5])

#ax2.legend()

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(dir_out, "RSLR_SLR_by_county.tiff"), dpi=200, bbox_inches="tight"
)

plt.show()
