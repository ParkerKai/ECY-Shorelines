"""Shoreline position per scan and change rate at every transect, per county.

For each transect and lidar scan, find the crossing of CROSSING_DATUM closest to
the shoreline, give it a horizontal uncertainty from the scan's vertical RMSE and
the local beach slope, then fit a weighted linear trend of crossing position
against scan year. Scans marked with "Flag Drop" = TRUE in the uncertainty Excel
file are excluded from the analysis.

The rate is reported with the sign flipped relative to distance along the
transect, so negative = erosion and positive = accretion. Each transect is
placed at its base point from the transects.gpkg of its county.

Outputs, in PROFILE_ROOT / <region> / "shoreline" (read by PlotTransectProfiles.py):
  crossings_<datum>.parquet           one row per transect x scan
  rates_<datum>.parquet               one row per transect, with the fit stats
                                      needed to draw the trend confidence band
  shoreline_positions_<datum>.gpkg    crossing points, one layer per scan
  shoreline_positions_polylines_<datum>.gpkg  crossing polylines, one layer per scan
The rate points are written to OUT_ROOT / <region> / OUT_NAME, layer
shoreline_change_<datum> (one point per transect), and as line segments to
OUT_ROOT / <region> / shoreline_change_polylines_<datum>.shp: points are connected
per path_id in sample_id order, with one segment per point (midpoint to midpoint)
carrying that point's attributes.
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from scipy import stats
from shapely.geometry import LineString

PROFILE_ROOT = Path(r"N:\SEA\Kai\Data\lidar\county_change")
OUT_ROOT = Path(r"N:\SEA\Kai\Data\lidar\ShorelineChange")

# Per-scan vertical accuracy (ft). "unavailable" or blank values fall back to
# the mean over the other scans: mean |bias| and mean RMSE.
UNCERTAINTY_XLSX = Path(r"D:\Data\Lidar\lidar_ScanVerticalUncertainty.xlsx")
BIAS_COL = "Absolute Accuracy (average, ft.)"
RMSE_COL = "Absolue Accuracy (RMSE, ft.)"
FLAG_DROP_COL = (
    "Flag Drop"  # Column to drop scans we are suspicous of (marked with a 1)
)


FT_TO_M = 0.3048
TRANSECT_ROOT = Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
TIDAL_DATUM_LAYER = "transect_bases_tidal_datums"

# Counties to process; None = every county with both profiles and transects.
REGIONS = [
    "thurston",
    "jefferson",
    "king",
    "kitsap",
    "mason",
    "pierce",
    "skagit",
    "snohomish",
    "clallam",
    "whatcom",
    "island",
    "san_juan",
]

# Written to PROFILE_ROOT / <region> / OUT_NAME, one layer per crossing datum.
OUT_NAME = "shoreline_change.gpkg"

# Datum whose contour position is tracked through time: "MLLW", "LMSL" or "MHHW".
CROSSING_DATUM = "MHHW"
# MHHW lies above the water plane on every shoreline, so the mask would only remove valid crossings.
USE_WATER_MASK = CROSSING_DATUM != "MHHW"
# Where a flattened water plane lies within this many m above the datum, the dry beach
# slope is extrapolated down to the datum (MHHW only; 0 disables).
EXTRAPOLATE_MAX_M = 0.20 if CROSSING_DATUM == "MHHW" else 0.0

# Transects extend 150 m landward and seaward of the shoreline, so the shoreline is at 150 m.
SHORELINE_DIST_M = 150.0
SLOPE_WINDOW_M = (
    2.0  # half-width of the window used for the local beach slope at a crossing
)

# Trailing 4-digit year in the scan folder name, used for ordering and colour.
YEAR_PATTERN = re.compile(r"(\d{4})")

KEYS = ["path_id", "sample_id"]

# Shapefile field names are limited to 10 characters.
SHP_FIELD_NAMES = {
    "ci95_lower_m_yr": "ci95_lo",
    "ci95_upper_m_yr": "ci95_hi",
    "fit_intercept": "fit_icept",
    "fit_year_mean": "fit_ymean",
}


def scan_year(scan_name):
    years = YEAR_PATTERN.findall(scan_name)
    return int(years[-1]) if years else None


def shoreline_paths(region, datum=CROSSING_DATUM):
    out_dir = PROFILE_ROOT / region / "shoreline"
    return {
        "crossings": out_dir / f"crossings_{datum}.parquet",
        "rates": out_dir / f"rates_{datum}.parquet",
        "positions": out_dir / f"shoreline_positions_{datum}.gpkg",
        "positions_polylines": out_dir / f"shoreline_positions_polylines_{datum}.gpkg",
    }


def load_uncertainty():
    """Vertical bias and RMSE (m) per scan, plus the fallback used for missing values.
    Also loads the Flag Drop indicator for scans to exclude."""
    df = pd.read_excel(UNCERTAINTY_XLSX)

    # Handle optional Flag Drop column
    if FLAG_DROP_COL in df.columns:
        flag_drop_array = (
            pd.to_numeric(df[FLAG_DROP_COL], errors="coerce").to_numpy() == 1
        )
    else:
        # Column doesn't exist, create array of all False (no scans flagged)
        flag_drop_array = np.zeros(len(df), dtype=bool)

    table = pd.DataFrame(
        {
            "bias_z_m": pd.to_numeric(df[BIAS_COL], errors="coerce").to_numpy()
            * FT_TO_M,
            "rmse_z_m": pd.to_numeric(df[RMSE_COL], errors="coerce").to_numpy()
            * FT_TO_M,
            "flag_drop": flag_drop_array,
        },
        index=df["Dataset"].astype(str).str.strip(),
    )
    # No reported bias means no shift, since an average bias has no known sign.
    fallback = {
        "bias_z_m": 0.0,
        "rmse_z_m": table["rmse_z_m"].mean(),
    }
    print(f"Uncertainty fallback: bias 0 m, RMSE {fallback['rmse_z_m']:.3f} m")

    # Report on Flag Drop column
    if FLAG_DROP_COL not in df.columns:
        print(
            f"  Note: '{FLAG_DROP_COL}' column not found in Excel file; no scans will be filtered"
        )
    return table, fallback


def find_regions():
    return sorted(
        p.name
        for p in PROFILE_ROOT.iterdir()
        if (p / "profiles").is_dir()
        and (TRANSECT_ROOT / p.name / "transects.gpkg").exists()
    )


def load_bases(region):
    """Transect base points with the crossing datum level (m NAVD88)."""
    col = f"{CROSSING_DATUM}_navd88_m"
    bases = pyogrio.read_dataframe(
        TRANSECT_ROOT / region / "transects.gpkg",
        layer=TIDAL_DATUM_LAYER,
        columns=KEYS + ["measure_m", col],
    )
    return bases.rename(columns={col: "datum_z"})


def contiguous_runs(df):
    """Sort by path_id/sample_id and label runs of consecutive sample_ids on a path.

    Transects without a crossing or rate are missing from the data, so connecting
    across a sample_id gap would draw a line over the unmeasured stretch.
    """
    df = df.sort_values(KEYS, ignore_index=True)
    new = (df["path_id"] != df["path_id"].shift()) | (
        df["sample_id"].diff() != 1
    )
    return df.assign(run_id=new.cumsum())


def scan_crossings(path, levels):
    """Datum crossing (dist and x/y) and local beach slope per transect for one profile file.

    Wet samples are excluded only when USE_WATER_MASK is set; otherwise the flags only
    locate a flat water plane to extrapolate from.
    """
    names = set(pq.ParquetFile(path).schema_arrow.names)
    has_water = "water" in names
    use_mask = USE_WATER_MASK and has_water
    if (USE_WATER_MASK or EXTRAPOLATE_MAX_M) and not has_water:
        print(
            f"  WARNING: {path.name} has no 'water' column; mask/extrapolation skipped"
        )
    cols = KEYS + ["scan", "dist_m", "x", "y", "z"] + (["water"] if has_water else [])
    df = pq.read_table(path, columns=cols, read_dictionary=["scan"]).to_pandas()

    df = df.merge(levels, on=KEYS, how="inner")
    df["scan_code"] = df["scan"].cat.codes
    df = df.sort_values(
        KEYS + ["scan_code", "dist_m"], kind="stable", ignore_index=True
    )

    pid = df["path_id"].to_numpy()
    sid = df["sample_id"].to_numpy()
    sc = df["scan_code"].to_numpy()
    dist = df["dist_m"].to_numpy(float)
    px = df["x"].to_numpy(float)
    py = df["y"].to_numpy(float)
    z = df["z"].to_numpy(float)
    wet = (
        df["water"].fillna(False).to_numpy(bool)
        if has_water
        else np.zeros(len(df), bool)
    )
    if use_mask:
        z = np.where(wet, np.nan, z)
    level = df["datum_z"].to_numpy(float)

    new = np.r_[
        True, (pid[1:] != pid[:-1]) | (sid[1:] != sid[:-1]) | (sc[1:] != sc[:-1])
    ]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    n_groups = starts.size

    # Crossing closest to SHORELINE_DIST_M, interpolated only between adjacent
    # valid samples of the same profile.
    d = z - level
    d0, d1 = d[:-1], d[1:]
    hit = (
        (gid[1:] == gid[:-1])
        & np.isfinite(d0)
        & np.isfinite(d1)
        & ((d0 == 0) | (np.sign(d0) != np.sign(d1)))
    )
    idx = np.flatnonzero(hit)
    t = np.divide(
        d0[idx], d0[idx] - d1[idx], out=np.zeros(idx.size), where=d0[idx] != 0
    )
    xs = dist[idx] + t * (dist[idx + 1] - dist[idx])
    order = np.lexsort((np.abs(xs - SHORELINE_DIST_M), gid[idx]))
    g_hit, first = np.unique(gid[idx][order], return_index=True)
    pick = idx[order][first]
    t_pick = t[order][first]
    x0 = np.full(n_groups, np.nan)
    x0[g_hit] = xs[order][first]
    cx = np.full(n_groups, np.nan)
    cy = np.full(n_groups, np.nan)
    cx[g_hit] = px[pick] + t_pick * (px[pick + 1] - px[pick])
    cy[g_hit] = py[pick] + t_pick * (py[pick + 1] - py[pick])

    # No crossing, but a flat water plane just above the datum: extend the dry beach
    # slope from the wet/dry boundary down to the datum.
    extrapolated = np.zeros(n_groups, bool)
    slope_ext = np.full(n_groups, np.nan)
    if EXTRAPOLATE_MAX_M and has_water:
        fin = np.isfinite(z)
        i = np.flatnonzero((gid[1:] == gid[:-1]) & (wet[1:] != wet[:-1]))
        ib = np.where(wet[i], i + 1, i)  # dry side of each wet/dry pair
        iw = np.where(wet[i], i, i + 1)
        gap = z[iw] - level[ib]
        cand = (
            np.isnan(x0[gid[ib]])
            & fin[ib]
            & fin[iw]
            & (gap > 0)
            & (gap <= EXTRAPOLATE_MAX_M)
            & (z[ib] > level[ib])
        )
        ib, iw = ib[cand], iw[cand]
        order = np.lexsort((np.abs(dist[ib] - SHORELINE_DIST_M), gid[ib]))
        g_sel, first = np.unique(gid[ib][order], return_index=True)
        ib, iw = ib[order][first], iw[order][first]

        xb = np.full(n_groups, np.nan)
        xb[g_sel] = dist[ib]
        rel_b = dist - xb[gid]
        okb = ~wet & fin & (np.abs(rel_b) <= SLOPE_WINDOW_M)  # NaN xb never passes
        g, r, zz = gid[okb], rel_b[okb], z[okb]
        nb = np.bincount(g, minlength=n_groups)
        sx = np.bincount(g, r, n_groups)
        sy = np.bincount(g, zz, n_groups)
        sxx = np.bincount(g, r * r, n_groups)
        sxy = np.bincount(g, r * zz, n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_b = (nb * sxy - sx * sy) / (nb * sxx - sx * sx)
            a_b = (sy - s_b * sx) / nb
            rel_c = (level[starts] - a_b) / s_b
        # The datum must lie on the wet side of the boundary.
        toward_wet = np.zeros(n_groups)
        toward_wet[g_sel] = np.sign(dist[iw] - dist[ib])
        good = (nb >= 3) & np.isfinite(rel_c) & (rel_c * toward_wet > 0)
        gs = np.flatnonzero(good)
        j = np.searchsorted(g_sel, gs)
        frac = rel_c[gs] / (dist[iw[j]] - dist[ib[j]])
        x0[gs] = xb[gs] + rel_c[gs]
        cx[gs] = px[ib[j]] + frac * (px[iw[j]] - px[ib[j]])
        cy[gs] = py[ib[j]] + frac * (py[iw[j]] - py[ib[j]])
        slope_ext[gs] = s_b[gs]
        extrapolated[gs] = True

    # Least-squares beach slope within +/- SLOPE_WINDOW_M of the crossing.
    rel = dist - x0[gid]
    ok = np.isfinite(z) & (np.abs(rel) <= SLOPE_WINDOW_M)
    g, r, zz = gid[ok], rel[ok], z[ok]
    n = np.bincount(g, minlength=n_groups)
    sx = np.bincount(g, r, n_groups)
    sy = np.bincount(g, zz, n_groups)
    sxx = np.bincount(g, r * r, n_groups)
    sxy = np.bincount(g, r * zz, n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        beach_slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    beach_slope[n < 3] = np.nan
    beach_slope = np.where(extrapolated, slope_ext, beach_slope)

    out = pd.DataFrame(
        {
            "path_id": pid[starts],
            "sample_id": sid[starts],
            "scan": df["scan"].to_numpy()[starts],
            "dist_m": x0,
            "x": cx,
            "y": cy,
            "beach_slope": beach_slope,
            "extrapolated": extrapolated,
        }
    )
    return out[np.isfinite(out["dist_m"])]


def change_rates(crossings):
    """Weighted trend of measured crossing position per transect."""
    c = crossings.dropna(subset=["sigma_x_m"]).copy()
    c["w"] = 1 / c["sigma_x_m"] ** 2
    c["wx"] = c["w"] * c["year"]
    c["wy"] = c["w"] * c["dist_m"]
    grp = c.groupby(KEYS)
    s = grp[["w", "wx", "wy"]].sum()
    s["n"] = grp.size()
    s["year_start"] = grp["year"].min() - 0.5
    s["year_end"] = grp["year"].max() - 0.5
    s = s[s["n"] >= 2]
    c = c.join(
        s[["w", "wx", "wy"]].rename(columns=lambda k: f"sum_{k}"), on=KEYS, how="inner"
    )

    xw = c["sum_wx"] / c["sum_w"]
    yw = c["sum_wy"] / c["sum_w"]
    c["dxx"] = c["w"] * (c["year"] - xw) ** 2
    c["dxy"] = c["w"] * (c["year"] - xw) * (c["dist_m"] - yw)
    m = c.groupby(KEYS)[["dxx", "dxy"]].sum()
    s = s.join(m)
    s = s[s["dxx"] > 0]
    s["slope"] = s["dxy"] / s["dxx"]
    s["intercept"] = s["wy"] / s["w"] - s["slope"] * s["wx"] / s["w"]

    c = c.join(s[["slope", "intercept"]], on=KEYS, how="inner")
    c["chi"] = c["w"] * (c["dist_m"] - (c["slope"] * c["year"] + c["intercept"])) ** 2
    s["chi2"] = c.groupby(KEYS)["chi"].sum()

    # Variance scaled by reduced chi-square (when > 1) so scatter beyond the
    # propagated measurement error also widens the interval.
    dof = s["n"] - 2
    chi2_red = np.where(dof > 0, s["chi2"] / dof.clip(lower=1), 1.0)
    scale = np.maximum(1.0, chi2_red)
    slope_se = np.sqrt(scale / s["dxx"])
    tcrit = np.where(dof > 0, stats.t.ppf(0.975, dof.clip(lower=1)), 1.96)
    ci = tcrit * slope_se

    rate = -s["slope"]
    # The interval is centred on the rate, so significance means the interval excludes zero.
    change = np.select(
        [rate + ci < 0, rate - ci > 0], ["erosion", "accretion"], "NoSignificantChange"
    )
    # The trend/fit_* columns are in distance along the transect, so the band
    # dist(t) +/- tcrit * sqrt(scale * (1/sum_w + (t - year_mean)^2 / sxx)) can be redrawn.
    return pd.DataFrame(
        {
            "rate_m_yr": rate,
            "change": change,
            "ci95_m_yr": ci,
            "ci95_lower_m_yr": rate - ci,
            "ci95_upper_m_yr": rate + ci,
            "n": s["n"].astype(int),
            "year_start": s["year_start"].astype(int),
            "year_end": s["year_end"].astype(int),
            "trend_m_yr": s["slope"],
            "fit_intercept": s["intercept"],
            "fit_year_mean": s["wx"] / s["w"],
            "fit_sum_w": s["w"],
            "fit_sxx": s["dxx"],
            "fit_scale": scale,
            "fit_tcrit": tcrit,
        },
        index=s.index,
    ).reset_index()


def process_region(region, uncertainty, fallback):
    print(f"\n=== {region} ===")
    bases = load_bases(region)
    levels = bases.loc[bases["datum_z"].notna(), KEYS + ["datum_z"]]
    print(f"{len(bases)} transects, {len(levels)} with a {CROSSING_DATUM} level")

    frames = []
    for path in sorted((PROFILE_ROOT / region / "profiles").glob("*_profiles.parquet")):
        c = scan_crossings(path, levels)
        print(f"  {path.name:55s} {len(c):7d} crossings")
        frames.append(c)
    if not frames:
        print("  no profile files, skipping")
        return

    crossings = pd.concat(frames, ignore_index=True)
    crossings["scan"] = crossings["scan"].astype(str)
    years = crossings["scan"].map(scan_year)
    crossings = crossings[years.notna()].assign(year=years.dropna() + 0.5)

    crossings = crossings.join(uncertainty, on="scan")
    missing = crossings[["bias_z_m", "rmse_z_m"]].isna()
    crossings["uncertainty_fallback"] = missing.any(axis=1)

    # Filter out flagged scans (flag_drop = True)
    flagged_scans = uncertainty[uncertainty["flag_drop"]].index.tolist()
    if flagged_scans:
        print(f"  Excluding flagged scans: {', '.join(flagged_scans)}")
        crossings = crossings[~crossings["scan"].isin(flagged_scans)]

    crossings = crossings.fillna(fallback)
    for scan in sorted(
        crossings.loc[crossings["uncertainty_fallback"], "scan"].unique()
    ):
        print(f"  {scan}: no reported accuracy, using fallback")
    # RMSE already includes the bias (RMSE^2 = bias^2 + SD^2), so it is the 1-sigma.
    with np.errstate(divide="ignore"):
        crossings["sigma_x_m"] = crossings["rmse_z_m"] / crossings["beach_slope"].abs()
    crossings.loc[~np.isfinite(crossings["sigma_x_m"]), "sigma_x_m"] = np.nan

    # Bias shifts the interval, not the point or the fit; assumes bias = measured - reference z.
    shift = crossings["dist_m"] + crossings["bias_z_m"] / crossings["beach_slope"]
    crossings["bar_lower_m"] = shift - 1.96 * crossings["sigma_x_m"]
    crossings["bar_upper_m"] = shift + 1.96 * crossings["sigma_x_m"]

    rates = change_rates(crossings)
    rates["datum"] = CROSSING_DATUM

    paths = shoreline_paths(region)
    paths["crossings"].parent.mkdir(parents=True, exist_ok=True)
    crossings.to_parquet(paths["crossings"], index=False)
    rates.to_parquet(paths["rates"], index=False)
    print(f"Wrote {paths['crossings']} and {paths['rates']}")

    crs = pyogrio.read_info(
        TRANSECT_ROOT / region / "transects.gpkg", layer=TIDAL_DATUM_LAYER
    )["crs"]
    paths["positions"].unlink(missing_ok=True)
    paths["positions_polylines"].unlink(missing_ok=True)
    for scan, c in crossings.groupby("scan"):
        points = gpd.GeoDataFrame(
            c.drop(columns=["x", "y"]),
            geometry=gpd.points_from_xy(c["x"], c["y"]),
            crs=crs,
        )
        pyogrio.write_dataframe(points, paths["positions"], layer=scan, driver="GPKG", dataset_options={"OVERWRITE": "YES"})

        # Create polyline version of the crossing positions
        polylines = []
        for _, group in contiguous_runs(c).groupby("run_id", sort=False):
            if len(group) < 2:
                continue

            coords = list(zip(group["x"], group["y"]))
            line_geom = LineString(coords)

            line_attrs = group.iloc[0].drop(["x", "y", "run_id"]).to_dict()
            line_attrs["sample_id_end"] = group["sample_id"].iloc[-1]
            line_attrs["geometry"] = line_geom
            line_attrs["n_points"] = len(group)

            polylines.append(line_attrs)

        if polylines:
            polylines_gdf = gpd.GeoDataFrame(polylines, crs=crs)
            pyogrio.write_dataframe(
                polylines_gdf, paths["positions_polylines"], layer=scan, driver="GPKG", dataset_options={"OVERWRITE": "YES"}
            )

    print(f"Wrote {crossings['scan'].nunique()} scan layers to {paths['positions']}")
    print(
        f"Wrote {crossings['scan'].nunique()} scan layers to {paths['positions_polylines']}"
    )

    out = bases.drop(columns="datum_z").merge(rates, on=KEYS, how="inner")
    out.insert(0, "region", region)

    out_path = OUT_ROOT / region / OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layer = f"shoreline_change_{CROSSING_DATUM}"
    pyogrio.write_dataframe(out, out_path, layer=layer, driver="GPKG", dataset_options={"OVERWRITE": "YES"})
    counts = out["change"].value_counts().to_dict()
    print(f"Wrote {len(out)} transects ({counts}) to {out_path} [{layer}]")

    write_change_polylines(out, OUT_ROOT / region)


def write_change_polylines(out, out_dir):
    """Shapefile with one segment per rate point, carrying that point's attributes.

    Points are connected per path_id in sample_id order; each segment runs from the
    midpoint with the previous point, through the point, to the midpoint with the next.
    """
    segments = []
    for _, group in contiguous_runs(out).groupby("run_id", sort=False):
        if len(group) < 2:
            continue
        xy = np.column_stack([group.geometry.x, group.geometry.y])
        mid = (xy[:-1] + xy[1:]) / 2
        starts = np.vstack([xy[:1], mid])
        ends = np.vstack([mid, xy[-1:]])
        segments.append(
            gpd.GeoDataFrame(
                group.drop(columns=["geometry", "run_id"]),
                geometry=[LineString(v) for v in zip(starts, xy, ends)],
                crs=out.crs,
            )
        )
    if not segments:
        return

    lines = pd.concat(segments, ignore_index=True).rename(columns=SHP_FIELD_NAMES)
    shp_path = out_dir / f"shoreline_change_polylines_{CROSSING_DATUM}.shp"
    pyogrio.write_dataframe(lines, shp_path, driver="ESRI Shapefile")
    print(f"Wrote {len(lines)} polyline segments to {shp_path}")


#################### Main ######################


if __name__ == "__main__":
    uncertainty, fallback = load_uncertainty()
    for region in REGIONS or find_regions():
        process_region(region, uncertainty, fallback)
