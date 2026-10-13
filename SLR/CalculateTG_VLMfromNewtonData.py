# -*- coding: utf-8 -*-
"""
Created on 07/01/2026

This script determines vertical land motion (VLM) at tide gauge locations by comparing the observed water level data with a reference model. The script downloads tide gauge data from NOAA CO-OPS, processes the data, and combines it with geospatial information to analyze VLM trends.
Data is from:  https://www.mdpi.com/2073-4441/13/3/281
An Assessment of Vertical Land Movement to Support Coastal Hazards Planning in Washington State
Newtwon et al., 2021

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"


# -------------------------------
# Packages
# -------------------------------

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator, Rbf
from pyproj import CRS, Transformer
import os
import json
from glob import glob

# -------------------------------
# User Input
# -------------------------------

stat_list = [
    "9439040",
    "9440581",
    "9440910",
    "9441102",
    "9442396",
    "9443090",
    "9444090",
    "9444900",
    "9449880",
    "9449424",
    "9447130",
    "9445958",
    "9446484",
    "9440569",
    "9439099",
    "9437540",
    "9435380",
    "9432780",
    "9431647",
]

dir_in_TG = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"
dir_in_VLM = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\VLM"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\VLM"

interp_method = (
    "rbf-thin_plate"  # 'nearest' | 'linear' | 'cubic' | 'idw' | 'rbf-thin_plate'
)

# -------------------------------
# Helpers: grid detection & projection
# -------------------------------


def detect_regular_grid(
    df,
    lon_col="Longitude",
    lat_col="Latitude",
    value_col=None,  # Optional: pass the column to pivot into the grid
    tol=1e-8,  # Absolute tolerance for spacing checks
    snap=False,  # Snap coords to dominant step to counter tiny float jitter
    wrap_longitude=False,  # Normalize longitudes to [-180, 180]
    dupe_agg="error",  # 'error' | 'mean' | 'median' | 'first' | 'last'
    allow_missing_ratio=0.0,  # Fraction of missing grid cells allowed for reshape
    return_pivot=False,  # If True and reshaping is feasible, return the pivoted 2D grid
    lat_first=True,  # If True, pivot returns rows=lat, cols=lon (matrix.shape=(n_lat, n_lon))
):
    """
    Determine whether `df` forms a complete, regularly spaced Lon–Lat grid,
    whether it *can be reshaped* into such a grid (with NaN for missing cells),
    and optionally return the pivoted grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with longitude/latitude columns.
    lon_col : str
        Name of the longitude column.
    lat_col : str
        Name of the latitude column.
    value_col : str or None
        Name of the value column to pivot into the grid. If None, reshaping (pivot)
        will produce a boolean presence grid.
    tol : float
        Absolute tolerance for equal step checks along lon/lat.
    snap : bool
        If True, snap coordinate values to the dominant step before checks.
    wrap_longitude : bool
        If True, wrap longitudes to [-180, 180] to avoid antimeridian artifacts.
    dupe_agg : {'error','mean','median','first','last'}
        How to handle duplicate (lon,lat) pairs:
        - 'error': report duplicates and do not aggregate.
        - 'mean'/'median'/'first'/'last': aggregate duplicate values accordingly.
    allow_missing_ratio : float
        Allow up to this fraction of the full grid to be missing but still
        considered "reshape-able".
    return_pivot : bool
        If True and reshaping is feasible, include a pivoted 2D grid in the result.
    lat_first : bool
        Controls orientation of the returned grid (rows Lat, columns Lon if True).

    Returns
    -------
    dict
        {
            "is_grid": bool,            # complete regular grid
            "can_reshape": bool,        # regular spacing & missing <= allow_missing_ratio
            "full_rect": bool,          # no missing & no extras after de-dup
            "regular": bool,            # regular spacing on both axes
            "lons": np.ndarray,         # unique sorted (possibly snapped) longitudes
            "lats": np.ndarray,         # unique sorted (possibly snapped) latitudes
            "lon_step": float or None,
            "lat_step": float or None,
            "n_expected": int,          # len(lons)*len(lats)
            "n_rows": int,              # rows after cleaning/aggregation
            "has_dupes": bool,          # duplicates before aggregation
            "dupe_agg_applied": str or None,
            "missing_n": int,           # number of missing lon-lat pairs
            "missing_examples": list,   # up to 10 examples of missing pairs
            "extra_n": int,             # pairs not in expected product (should be 0 after snap/agg)
            "bbox": (lon_min, lon_max, lat_min, lat_max),
            "pivot": pd.DataFrame or None,  # 2D grid (lat×lon or lon×lat) if return_pivot and can_reshape
        }
    """
    # --- Validate inputs -------------------------------------------------------
    if lon_col not in df.columns or lat_col not in df.columns:
        raise KeyError(
            f"Input DataFrame must contain '{lon_col}' and '{lat_col}' columns."
        )
    if value_col is not None and value_col not in df.columns:
        raise KeyError(f"Requested value_col '{value_col}' not found in DataFrame.")

    # Work on a copy; drop rows missing coords (keep value NaN—they become NaN in the grid)
    coord_cols = [lon_col, lat_col]
    keep_cols = coord_cols + ([value_col] if value_col else [])
    dfg = df[keep_cols].dropna(subset=coord_cols).copy()

    # Optional wrapping
    if wrap_longitude:
        dfg[lon_col] = ((dfg[lon_col] + 180.0) % 360.0) - 180.0

    # --- Handle duplicates -----------------------------------------------------
    has_dupes = dfg.duplicated(coord_cols).any()
    dupe_agg_applied = None

    if has_dupes:
        if dupe_agg == "error":
            # Keep as-is, but report failure to be a full rectangle
            pass
        else:
            dupe_agg_applied = dupe_agg
            if value_col is None:
                # No value to aggregate—just drop to first/last/unique
                if dupe_agg in ("first", "last"):
                    keep = ~dfg.duplicated(
                        coord_cols, keep=("first" if dupe_agg == "first" else "last")
                    )
                    dfg = dfg.loc[keep]
                elif dupe_agg in ("mean", "median"):
                    # When no value_col, just keep unique coord pairs
                    dfg = dfg.drop_duplicates(coord_cols)
                else:
                    raise ValueError(f"Unsupported dupe_agg: {dupe_agg}")
            else:
                grp = dfg.groupby(coord_cols, as_index=False)
                if dupe_agg == "mean":
                    dfg = grp[value_col].mean()
                elif dupe_agg == "median":
                    dfg = grp[value_col].median()
                elif dupe_agg == "first":
                    dfg = grp[value_col].first()
                elif dupe_agg == "last":
                    dfg = grp[value_col].last()
                else:
                    raise ValueError(f"Unsupported dupe_agg: {dupe_agg}")

            # After aggregation, duplicates are resolved
            has_dupes = dfg.duplicated(coord_cols).any()

    # --- Regularize coordinates (optionally snap) ------------------------------
    def _regularize_unique(values):
        """Return uniq sorted array, step (median diff), regular flag."""
        arr = np.asarray(values, dtype=float)
        uniq = np.unique(arr)
        uniq.sort()
        if uniq.size <= 1:
            return uniq, None, True  # Trivially regular for single value
        diffs = np.diff(uniq)
        step = np.median(diffs)
        regular = (step > 0) and np.allclose(diffs, step, atol=tol, rtol=0)
        return uniq, float(step), bool(regular)

    lons, lon_step, lons_regular = _regularize_unique(dfg[lon_col])
    lats, lat_step, lats_regular = _regularize_unique(dfg[lat_col])

    # Optional snapping to dominant step
    if snap:

        def _snap_coords(series, base, step):
            if step is None or step <= 0:
                return series
            return base + step * np.round((series - base) / step)

        # Pick bases as current minima
        lon_base = lons.min() if lons.size else 0.0
        lat_base = lats.min() if lats.size else 0.0

        dfg[lon_col] = _snap_coords(dfg[lon_col].to_numpy(float), lon_base, lon_step)
        dfg[lat_col] = _snap_coords(dfg[lat_col].to_numpy(float), lat_base, lat_step)

        # Recompute uniques after snapping
        lons, lon_step, lons_regular = _regularize_unique(dfg[lon_col])
        lats, lat_step, lats_regular = _regularize_unique(dfg[lat_col])

        # Snapping may create duplicates—resolve them if an aggregator is provided
        if dfg.duplicated(coord_cols).any():
            if dupe_agg == "error":
                has_dupes = True  # report
            else:
                dupe_agg_applied = dupe_agg
                if value_col is None:
                    dfg = dfg.drop_duplicates(coord_cols)
                else:
                    grp = dfg.groupby(coord_cols, as_index=False)
                    if dupe_agg == "mean":
                        dfg = grp[value_col].mean()
                    elif dupe_agg == "median":
                        dfg = grp[value_col].median()
                    elif dupe_agg == "first":
                        dfg = grp[value_col].first()
                    elif dupe_agg == "last":
                        dfg = grp[value_col].last()

    # --- Completeness: missing / extras relative to Cartesian product ----------
    n_expected = int(lons.size * lats.size)
    n_rows = int(len(dfg))
    coords = pd.MultiIndex.from_frame(dfg[coord_cols][[lat_col, lon_col]])
    expected = pd.MultiIndex.from_product([lats, lons], names=[lat_col, lon_col])

    missing = expected.difference(coords)
    extra = coords.difference(
        expected
    )  # should be empty unless irregular/snap mismatch

    full_rect = (missing.size == 0) and (extra.size == 0) and (not has_dupes)
    regular = lons_regular and lats_regular
    is_grid = bool(full_rect and regular)

    missing_n = int(missing.size)
    extra_n = int(extra.size)
    missing_examples = list(missing)[:10]

    can_reshape = bool(
        regular
        and (n_expected > 0)
        and (missing_n / n_expected <= float(allow_missing_ratio))
    )

    bbox = (
        float(lons.min()) if lons.size else np.nan,
        float(lons.max()) if lons.size else np.nan,
        float(lats.min()) if lats.size else np.nan,
        float(lats.max()) if lats.size else np.nan,
    )

    # --- Optional pivot (2D grid with NaN for missing cells) -------------------
    pivot_df = None
    if return_pivot and can_reshape:
        # Build a pivot with the full expected index/columns to ensure shape
        if lat_first:
            pivot_df = dfg.pivot(index=lat_col, columns=lon_col, values=value_col)
            pivot_df = pivot_df.reindex(index=lats, columns=lons)
        else:
            pivot_df = dfg.pivot(index=lon_col, columns=lat_col, values=value_col)
            pivot_df = pivot_df.reindex(index=lons, columns=lats)

        # If value_col is None, produce a boolean presence grid
        if value_col is None:
            presence = pd.Series(True, index=coords)
            presence_df = presence.unstack(level=1 if lat_first else 0)
            presence_df = presence_df.reindex_like(pivot_df).fillna(False)
            pivot_df = presence_df.astype(bool)

    return {
        "is_grid": is_grid,
        "can_reshape": can_reshape,
        "full_rect": bool(full_rect),
        "regular": bool(regular),
        "lons": lons,
        "lats": lats,
        "lon_step": lon_step,
        "lat_step": lat_step,
        "n_expected": n_expected,
        "n_rows": n_rows,
        "has_dupes": bool(has_dupes),
        "dupe_agg_applied": dupe_agg_applied,
        "missing_n": missing_n,
        "missing_examples": missing_examples,
        "extra_n": extra_n,
        "bbox": bbox,
        "pivot": pivot_df if return_pivot else None,
    }


def make_local_aeqd_crs(lon0, lat0):
    """
    Build a local Azimuthal Equidistant (AEQD) CRS centered at lon0, lat0 (WGS84).
    Good for distance-based interpolation over regional extents.
    """
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return CRS.from_proj4(proj4)


def project_lonlat(lon, lat, transformer):
    """Project lon/lat arrays to x/y using a pyproj Transformer."""
    x, y = transformer.transform(
        lat, lon
    )  # Note: Transformer default is (lat, lon) -> (y, x)
    return np.asarray(x), np.asarray(y)


# -------------------------------
# Interpolators
# -------------------------------


def interpolate_scattered(
    lon, lat, val, target_lon, target_lat, method="linear", use_projection=True
):
    """
    Interpolate scattered points to target lon/lat.
    method: 'nearest' | 'linear' | 'cubic' | 'idw' | 'rbf-thin_plate'
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    val = np.asarray(val)
    target_lon = np.asarray(target_lon)
    target_lat = np.asarray(target_lat)

    if use_projection:
        # Center projection on data centroid for distance fidelity
        lon0, lat0 = np.nanmean(lon), np.nanmean(lat)
        crs_aeqd = make_local_aeqd_crs(lon0, lat0)
        transformer = Transformer.from_crs("EPSG:4326", crs_aeqd, always_xy=False)
        X, Y = project_lonlat(lon, lat, transformer)
        Xt, Yt = project_lonlat(target_lon, target_lat, transformer)
        pts = np.column_stack([X, Y])
        tpts = np.column_stack([Xt, Yt])
    else:
        pts = np.column_stack([lon, lat])
        tpts = np.column_stack([target_lon, target_lat])

    # Built-in methods
    if method in ("nearest", "linear", "cubic"):
        # griddata returns NaN outside convex hull for linear/cubic; nearest fills
        out = griddata(pts, val, tpts, method=method)
        return out

    # Simple inverse-distance weighting (IDW)
    if method == "idw":
        # P = sum(w_i * v_i) / sum(w_i), w_i = 1 / d_i^p with p=2; epsilon to avoid zero distance
        p = 2.0
        epsilon = 1e-12
        out = np.empty(len(tpts))
        for i, tp in enumerate(tpts):
            d = np.linalg.norm(pts - tp, axis=1)
            w = 1.0 / np.maximum(d, epsilon) ** p
            out[i] = np.sum(w * val) / np.sum(w)
        return out

    # Radial Basis Function (thin-plate spline) — smooth surface, good for gentle fields like VLM
    if method.startswith("rbf"):
        func = "thin_plate"  # alternative: 'multiquadric', 'inverse', 'gaussian'
        if use_projection:
            rbf = Rbf(X, Y, val, function=func, neighbors=20)
            return rbf(Xt, Yt)
        else:
            rbf = Rbf(lon, lat, val, function=func, neighbors=20)
            return rbf(target_lon, target_lat)

    raise ValueError(f"Unknown scattered interpolation method: {method}")


def interpolate_gridded(
    df, grid_info, target_lon, target_lat, allow_extrapolation=False
):
    """
    Bilinear interpolation over a complete regular lon-lat grid using RegularGridInterpolator.
    """
    # Build 2D array V[lat_idx, lon_idx] with lat ascending, lon ascending
    pivot = df.pivot(index="Latitude", columns="Longitude", values="VLM")
    # Ensure ordering matches grid_info arrays
    pivot = pivot.reindex(index=grid_info["lats"], columns=grid_info["lons"])
    V = pivot.values.astype(float)
    # RegularGridInterpolator expects axes as (lat, lon) or any order; we’ll keep (lat, lon)
    rgi = RegularGridInterpolator(
        (grid_info["lats"], grid_info["lons"]),
        V,
        bounds_error=not allow_extrapolation,
        fill_value=None if allow_extrapolation else np.nan,
        method="linear",
    )
    # Query points — shape (n, 2) with [lat, lon]
    qpts = np.column_stack([np.asarray(target_lat), np.asarray(target_lon)])
    return rgi(qpts)


# -------------------------------
# High-level function
# -------------------------------


def evaluate_csv_at_points(
    csv_path,
    target_points,  # list/array of (lon, lat)
    prefer_projection=True,  # Use projection for scattered (recommended)
    scattered_method="linear",  # 'nearest' | 'linear' | 'cubic' | 'idw' | 'rbf-thin_plate'
    allow_extrapolation=False,  # For gridded interpolator
):
    """
    Read CSV with columns: Longitude, Latitude, VLM.
    Detect grid vs scattered and interpolate accordingly.
    Returns a DataFrame with target points and interpolated VLM.
    """
    df = pd.read_csv(csv_path)
    # Basic cleaning
    df = df.rename(columns=lambda c: c.strip())
    assert {"Longitude", "Latitude", "VLM"}.issubset(df.columns), (
        "CSV must have Longitude, Latitude, VLM columns"
    )
    df_clean = df.dropna(subset=["Longitude", "Latitude", "VLM"]).copy()

    # Detect grid
    gi = detect_regular_grid(df_clean, "Longitude", "Latitude", "VLM")

    t_lon = np.array([p[0] for p in target_points], dtype=float)
    t_lat = np.array([p[1] for p in target_points], dtype=float)

    if gi["is_grid"]:
        values = interpolate_gridded(
            df_clean, gi, t_lon, t_lat, allow_extrapolation=allow_extrapolation
        )
        mode = "gridded-linear (RegularGridInterpolator)"
    else:
        values = interpolate_scattered(
            df_clean["Longitude"].values,
            df_clean["Latitude"].values,
            df_clean["VLM"].values,
            t_lon,
            t_lat,
            method=scattered_method,
            use_projection=prefer_projection,
        )
        mode = f"scattered-{scattered_method}"

    out = pd.DataFrame(
        {
            "Longitude": t_lon,
            "Latitude": t_lat,
            "VLM_interp": values,
            "interp_mode": mode,
        }
    )
    return out, gi


# -------------------------------
# Run
# -------------------------------

# Read Lat/Lon for stations

lon_lat = np.full((len(stat_list), 2), np.nan)
for cnt, stat in enumerate(stat_list):
    f_in = glob(os.path.join(dir_in_TG, f"NOAA_TG_{stat}", f"metadata_*_{stat}.json"))[
        0
    ]

    with open(f_in, "r", encoding="utf-8") as file:
        data = json.load(file)  # Converts JSON to Python dict/list
        if not isinstance(data, dict):
            raise ValueError("JSON root element is not a dictionary.")

    lon_lat[cnt, 1] = data.get("Latitude")
    lon_lat[cnt, 0] = data.get("Longitude")


# --------------- VLM ------------
VLM_df, grid_info = evaluate_csv_at_points(
    os.path.join(dir_in_VLM, "InterpolationUncertaintyGrid_Data_S5.csv"),
    lon_lat,
    prefer_projection=True,  # set False if your area is extremely small
    scattered_method=interp_method,  # try 'nearest', 'idw', or 'rbf-thin_plate' for robustness
    allow_extrapolation=False,
)


print("Grid detection:", grid_info)
print(VLM_df)

# --------------- VLM Uncertainty ------------

UNC_df, grid_info = evaluate_csv_at_points(
    os.path.join(dir_in_VLM, "InterpolationVLM_Data_S6.csv"),
    lon_lat,
    prefer_projection=True,  # set False if your area is extremely small
    scattered_method=interp_method,  # try 'nearest', 'idw', or 'rbf-thin_plate' for robustness
    allow_extrapolation=False,
)

# Combine
out_df = pd.concat([VLM_df, UNC_df["VLM_interp"].rename("VLM_uncertainty")], axis=1)


# Save results\
out_df["Station"] = stat_list

out_df.to_csv(os.path.join(dir_out, "vlm_at_TGs .csv"), index=False)
