# -*- coding: utf-8 -*-
"""
Created on 05/01/2026

This script process Miller 2018 SLR data and makes plots

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
import matplotlib
from glob import glob
import geopandas as gpd
from shapely.geometry import Point
import re
import matplotlib.pyplot as plt
import seaborn as sns



# ===============================================================================
# Functions
# ===============================================================================


def load_RSLR_table(file, sheet, header_row, nrows=16):
    # Read the block
    df = pd.read_excel(
        file,
        sheet_name=sheet,
        engine="openpyxl",
        header=header_row,
        nrows=nrows
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
            file,
            sheet_name='Site Overview',
            engine="openpyxl",
            header=None,
            nrows=1)


    line1 = overview.iloc[0,0]

    match = re.search(r"([0-9.]+)N,\s*([0-9.]+)W", line1)
    if match:
        lat_lon =pd.DataFrame([[float(match.group(1)),
                               -float(match.group(2))]]
                              , columns=["Lat","Lon"])   # West → negative
    else:
        print('Coordinates not found in the expected format.')
        lat_lon = pd.DataFrame([[np.nan,
                               np.nan]],
                               columns=["Lat","Lon"])
    

    twl = pd.read_excel(
            file,
            sheet_name='Site Overview',
            engine="openpyxl",
            header=12,
            nrows=1)

    
    return lat_lon, twl



# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\SLR\LocalTables"
dir_out= r'N:\SEA\Kai\Figures\RSLR_Miller2018'

ProbExceed = 50


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
        
    lat_lon, twl = load_RSLR_overview(file, sheet='Site Overview')

    lat_lon_list.append(lat_lon)
    twl_list.append(twl)

    RCP45_mag.append(load_RSLR_table(file, "RCP 4.5", header_row=7))
    RCP45_prob.append(load_RSLR_table(file, "RCP 4.5", header_row=29))

    RCP85_mag.append(load_RSLR_table(file, "RCP 8.5", header_row=7))
    RCP85_prob.append(load_RSLR_table(file, "RCP 8.5", header_row=29))


lat_lon = pd.concat(lat_lon_list, ignore_index=True)
twl = pd.concat(twl_list, ignore_index=True)


#### Read the Shoreline data for plotting ######

cnty = gpd.read_file(r'N:\SEA\Kai\General\WA_County_Boundaries.shp')
#cnty.set_crs(epsg=32610, inplace=True, allow_override=True)
cnty.to_crs(epsg=4236, inplace=True)


shoreline = gpd.read_file(r'N:\SEA\Kai\General\SMA_MarineShoreline.gpkg')
shoreline.to_crs(epsg=4236, inplace=True)

# ===============================================================================
# Turn a subset into a geodataframe
# ===============================================================================


# First, find cosmos year of occurence

year_25  = np.full(len(lat_lon), np.nan)
year_50  = np.full(len(lat_lon), np.nan)
year_100 = np.full(len(lat_lon), np.nan)
year_150 = np.full(len(lat_lon), np.nan)

for index, row in lat_lon.iterrows():
    pull = RCP45_mag[index][ProbExceed]
    
    y = pull.index.to_numpy()
    x = pull.to_numpy()
    
    year_25[index] = np.interp(0.25, x, y).round(0)
    year_50[index] = np.interp(0.50, x, y).round(0)
    year_100[index] = np.interp(1.00, x, y).round(0)
    year_150[index] = np.interp(1.50, x, y).round(0)
    
geometry = [Point(xy) for xy in zip(lat_lon['Lon'], lat_lon['Lat'])]

Out = gpd.GeoDataFrame({"year_25":year_25,
                        "year_50":year_50,
                        "year_100":year_100,
                        "year_150":year_150},
                       geometry=geometry,
                       crs = 'EPSG:4236')

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
cnty.plot(ax=ax1, column="JURISDIC_2", edgecolor="black",cmap = "Pastel2", alpha=0.4)
Out.plot(ax = ax1, column="year_25", cmap="viridis", legend=True, edgecolor="black")
ax1.grid()
ax1.set_xlim(-125,-122)
ax1.set_ylim(46,49)
ax1.set_title(f"Year of 0.25m RSLR occurrence ({ProbExceed}% probability)")
plt.savefig(os.path.join(dir_out,"RSLR_CosmosSLR_TimeOf025.tiff"), dpi=300, bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
cnty.plot(ax=ax1, column="JURISDIC_2", edgecolor="black",cmap = "Pastel2", alpha=0.4)
Out.plot(ax = ax1, column="year_50", cmap="viridis", legend=True, edgecolor="black")
ax1.grid()
ax1.set_xlim(-125,-122)
ax1.set_ylim(46,49)
ax1.set_title(f"Year of 0.50m RSLR occurrence ({ProbExceed}% probability)")
plt.savefig(os.path.join(dir_out,"RSLR_CosmosSLR_TimeOf050.tiff"), dpi=300, bbox_inches="tight")


fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
cnty.plot(ax=ax1, column="JURISDIC_2", edgecolor="black",cmap = "Pastel2", alpha=0.4)
Out.plot(ax = ax1, column="year_100", cmap="viridis", legend=True, edgecolor="black")
ax1.grid()
ax1.set_xlim(-125,-122)
ax1.set_ylim(46,49)
ax1.set_title(f"Year of 1.00m RSLR occurrence ({ProbExceed}% probability)")
plt.savefig(os.path.join(dir_out,"RSLR_CosmosSLR_TimeOf100.tiff"), dpi=300, bbox_inches="tight")



# ===============================================================================
# Aggregate by county
# ===============================================================================
# Spatial join: attach county name to each point
Out_with_county = gpd.sjoin(
    Out,
    cnty[['JURISDIC_2', 'geometry']],      # keep only name + geometry
    how='left',
    predicate='within'
)

# Rename column for clarity
Out_with_county.rename(columns={'JURISDIC_2': 'County'}, inplace=True)

sns.set_theme(style="darkgrid", palette="deep")

fig, [ax0, ax1, ax2, ax3] = plt.subplots(4, 1, figsize=(10, 10))
sns.boxplot(ax = ax0, data=Out_with_county, x='County', y='year_25')
ax0.set_title('Distribution of Year for 0.25 m RSLR by County')
ax0.set_xticklabels([])
ax0.grid(True, which='both', axis='both', alpha=0.5)
ax0.set(xlabel=None)
ax0.plot([0,14], [2050,2050], 'k--', label='Year 2050')

sns.boxplot(ax = ax1, data=Out_with_county, x='County', y='year_50')
ax1.set_title('Distribution of Year for 0.50 m RSLR by County')
ax1.set_xticklabels([])
ax1.grid(True, which='both', axis='both', alpha=0.5)
ax1.set(xlabel=None)
ax1.plot([0,14], [2050,2050], 'k--', label='Year 2050')
ax1.plot([0,14], [2100,2100], 'k--', label='Year 2100')


sns.boxplot(ax = ax2, data=Out_with_county, x='County', y='year_100')
ax2.set_title('Distribution of Year for 1.00 m RSLR by County')
ax2.set_xticklabels([])
ax2.grid(True, which='both', axis='both', alpha=0.5)
ax2.set(xlabel=None)
ax2.plot([0,14], [2050,2050], 'k--', label='Year 2050')
ax2.plot([0,14], [2100,2100], 'k--', label='Year 2100')
ax2.legend()

sns.boxplot(ax = ax3, data=Out_with_county, x='County', y='year_150')
ax3.set_title('Distribution of Year for 1.50 m RSLR by County')
ax3.grid(True, which='both', axis='both', alpha=0.5)
ax3.plot([0,14], [2100,2100], 'k--', label='Year 2100')


plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.savefig(os.path.join(dir_out,"RSLR_CosmosSLR_by_county.tiff"), dpi=300, bbox_inches="tight")

plt.show()
