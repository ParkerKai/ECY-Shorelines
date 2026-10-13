#!/usr/bin/env python3
"""
Aggregate point GeoJSONs -> one GeoDataFrame -> plot -> Shapefile.

"""

from pathlib import Path
import re
import geopandas as gpd
import pandas as pd


# -----------------------
# Utility helpers
# -----------------------


def sanitize_columns_for_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """
    ESRI Shapefile limits:
      - Max 10 chars for field names
      - Avoid exotic characters; use letters, numbers, underscores
      - No duplicate names
      - Basic dtypes (numeric, string); convert other objects to strings
    """

    def clean_name(name):
        # keep only letters, numbers, underscore; lower-case; collapse repeats
        base = re.sub(r"[^A-Za-z0-9_]", "_", str(name)).lower()
        base = re.sub(r"_+", "_", base).strip("_")
        # truncate to 10 chars
        return base[:10] if len(base) > 10 else base

    old2new = {}
    used = set()
    for c in df.columns:
        if c == "geometry":  # leave geometry alone
            old2new[c] = c
            continue
        new = clean_name(c)
        # Ensure uniqueness
        i = 1
        candidate = new
        while candidate in used or candidate == "geometry":
            suffix = str(i)
            candidate = (
                (new[: (10 - len(suffix))] + suffix)
                if len(new) + len(suffix) > 10
                else (new + suffix)
            )
            i += 1
        used.add(candidate)
        old2new[c] = candidate

    df = df.rename(columns=old2new)

    # Convert unsupported types to strings
    for c in df.columns:
        if c == "geometry":
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(
            df[c]
        ):
            df[c] = df[c].astype(str)

    return df


# -----------------------
# Main
# -----------------------
input_dir = Path(
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WFLOW\WFLOWgauges"
)
dir_out = Path(r'C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\WFLOW')

shoreline = gpd.read_file(r"N:\SEA\Kai\General\SMA_MarineShoreline.gpkg")
shoreline.to_crs(epsg=4236, inplace=True)


files = sorted(list(input_dir.glob("*.geojson")))
gdfs = []
for fp in files:
    gdf = gpd.read_file(fp)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf['county']  = fp.stem.split("_")[0]

    if "Contour" in gdf.columns:
        gdf = gdf.rename(columns={"Contour": "contour"})
        
    if "gauge_id" in gdf.columns and "wflow_idx" not in gdf.columns:
        gdf = gdf.rename(columns={"gauge_id": "wflow_idx"})
    
    if "Id" in gdf.columns:
        gdf = gdf.rename(columns={"Id": "wflow_idx"})
    
    if 'oid' in gdf.columns:
        gdf = gdf.drop(columns=["oid"])
 
    if 'fid' in gdf.columns:
        gdf = gdf.drop(columns=["fid"])

    if 'gauge_id' in gdf.columns:
        gdf = gdf.drop(columns=["gauge_id"])
    
    gdf = gdf.reset_index(drop=True)   
    
    print(fp.stem.split("_")[0])
    print(gdf.columns)

    gdfs.append(gdf)


Wflow = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

Wflow_PP = Wflow[(Wflow["contour"] == -1)].copy()

# Plot
ax = Wflow.plot(figsize=(6, 6), color="tab:blue", edgecolor="black")
shoreline.plot(ax=ax, color="red", edgecolor="black")
ax.set_aspect("equal")
ax.grid(True)

Out = sanitize_columns_for_shapefile(Wflow)

Out.to_file(dir_out / "WFLOWgauges.shp", driver="ESRI Shapefile")