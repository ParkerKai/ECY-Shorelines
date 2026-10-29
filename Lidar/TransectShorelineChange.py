"""Shoreline change rate at every transect, written to a GeoPackage per county.

Batch version of the datum-crossing trend (figure 2) in PlotTransectProfiles.py:
for each transect and lidar scan, find the seaward crossing of CROSSING_DATUM
on the dry profile, give it a horizontal uncertainty from the vertical lidar
error and the local beach slope, then fit a weighted linear trend of crossing
position against scan year.

The rate is reported with the sign flipped relative to distance along the
transect, so negative = erosion and positive = accretion (PlotTransectProfiles.py
labels a positive distance trend as erosion). Each transect is placed at its
base point from the transects.gpkg of its county.

The computation is vectorised across all transects, but follows the same steps
as seaward_crossing, local_slope and the weighted fit in PlotTransectProfiles.py.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from scipy import stats

PROFILE_ROOT = Path(r"D:\Data\Lidar\county_change")
TRANSECT_ROOT = Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
TIDAL_DATUM_LAYER = "transect_bases_tidal_datums"

# Counties to process; None = every county with both profiles and transects.
REGIONS = ['Thurston']

# Written to PROFILE_ROOT / <region> / OUT_NAME, one layer per crossing datum.
OUT_NAME = Path(r"N:\SEA\Kai\Data\lidar\ShorelineChange\shoreline_change.gpkg")

# Datum whose contour position is tracked through time: "MLLW", "LMSL" or "MHHW".
CROSSING_DATUM = "MHHW"
# Transects extend 150 m landward and seaward of the shoreline, so the shoreline is at 150 m.
SHORELINE_DIST_M = 150.0
POINT_SIGMA_Z_M = 0.20 / 1.96  # 1-sigma vertical lidar uncertainty (20 cm is the 95% CI)
SLOPE_WINDOW_M = 2.0  # half-width of the window used for the local beach slope at a crossing

# Trailing 4-digit year in the scan folder name, used for ordering and colour.
YEAR_PATTERN = re.compile(r"(\d{4})")

KEYS = ["path_id", "sample_id"]


def scan_year(scan_name):
    years = YEAR_PATTERN.findall(scan_name)
    return int(years[-1]) if years else None


def find_regions():
    return sorted(
        p.name
        for p in PROFILE_ROOT.iterdir()
        if (p / "profiles").is_dir() and (TRANSECT_ROOT / p.name / "transects.gpkg").exists()
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


def scan_crossings(path, levels):
    """Datum crossing, local beach slope and sigma per transect for one profile file."""
    names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = KEYS + ["scan", "dist_m", "z"] + (["water"] if "water" in names else [])
    if "water" not in names:
        print(f"  WARNING: {path.name} has no 'water' column; wet samples were dropped")
    df = pq.read_table(path, columns=cols, read_dictionary=["scan"]).to_pandas()
    if "water" not in df:
        df["water"] = False
    df["water"] = df["water"].fillna(False).astype(bool)

    df = df.merge(levels, on=KEYS, how="inner")
    df["scan_code"] = df["scan"].cat.codes
    df = df.sort_values(KEYS + ["scan_code", "dist_m"], kind="stable", ignore_index=True)

    pid = df["path_id"].to_numpy()
    sid = df["sample_id"].to_numpy()
    sc = df["scan_code"].to_numpy()
    dist = df["dist_m"].to_numpy(float)
    z = df["z"].to_numpy(float)
    wet = df["water"].to_numpy(bool)
    level = df["datum_z"].to_numpy(float)

    new = np.r_[True, (pid[1:] != pid[:-1]) | (sid[1:] != sid[:-1]) | (sc[1:] != sc[:-1])]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    n_groups = starts.size

    # Crossing of the dry profile closest to SHORELINE_DIST_M, interpolated only
    # between adjacent valid samples of the same profile.
    d = np.where(wet, np.nan, z) - level
    d0, d1 = d[:-1], d[1:]
    hit = (
        (gid[1:] == gid[:-1])
        & np.isfinite(d0)
        & np.isfinite(d1)
        & ((d0 == 0) | (np.sign(d0) != np.sign(d1)))
    )
    idx = np.flatnonzero(hit)
    t = np.divide(d0[idx], d0[idx] - d1[idx], out=np.zeros(idx.size), where=d0[idx] != 0)
    xs = dist[idx] + t * (dist[idx + 1] - dist[idx])
    order = np.lexsort((np.abs(xs - SHORELINE_DIST_M), gid[idx]))
    g_hit, first = np.unique(gid[idx][order], return_index=True)
    x0 = np.full(n_groups, np.nan)
    x0[g_hit] = xs[order][first]

    # Least-squares dry beach slope within +/- SLOPE_WINDOW_M of the crossing.
    rel = dist - x0[gid]
    ok = ~wet & np.isfinite(z) & (np.abs(rel) <= SLOPE_WINDOW_M)
    g, r, zz = gid[ok], rel[ok], z[ok]
    n = np.bincount(g, minlength=n_groups)
    sx = np.bincount(g, r, n_groups)
    sy = np.bincount(g, zz, n_groups)
    sxx = np.bincount(g, r * r, n_groups)
    sxy = np.bincount(g, r * zz, n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        beach_slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    beach_slope[n < 3] = np.nan

    out = pd.DataFrame(
        {
            "path_id": pid[starts],
            "sample_id": sid[starts],
            "scan": df["scan"].to_numpy()[starts],
            "dist_m": x0,
            "beach_slope": beach_slope,
        }
    )
    out = out[np.isfinite(out["dist_m"])]
    with np.errstate(divide="ignore"):
        out["sigma_x_m"] = POINT_SIGMA_Z_M / np.abs(out["beach_slope"])
    out.loc[~np.isfinite(out["sigma_x_m"]), "sigma_x_m"] = np.nan
    return out


def change_rates(crossings):
    """Weighted trend of crossing position per transect, as in PlotTransectProfiles.py."""
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
    c = c.join(s[["w", "wx", "wy"]].rename(columns=lambda k: f"sum_{k}"), on=KEYS, how="inner")

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
        },
        index=s.index,
    ).reset_index()


def process_region(region):
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

    rates = change_rates(crossings)
    out = bases.drop(columns="datum_z").merge(rates, on=KEYS, how="inner")
    out.insert(0, "region", region)
    out["datum"] = CROSSING_DATUM

    out_path = PROFILE_ROOT / region / OUT_NAME
    layer = f"shoreline_change_{CROSSING_DATUM}"
    pyogrio.write_dataframe(out, out_path, layer=layer, driver="GPKG")
    counts = out["change"].value_counts().to_dict()
    print(f"Wrote {len(out)} transects ({counts}) to {out_path} [{layer}]")


#################### Main ######################


for region in REGIONS or find_regions():
    process_region(region)
