"""Plot every available lidar scan profile for a single transect on one axis.

Reads the per-scan Parquet files written by TransectLidarProfiles.py and
overlays each scan's elevation profile for the chosen transect, so changes in
the profile over time are visible directly.

Stretches flagged as water are drawn as a band over the line, and the water
plane of the tile under the transect (or the scan plane, failing that) is drawn
as a horizontal rule, so the mask can be checked against the elevations it was
derived from. The mask cuts each scan at its water plane: everything seaward of
where the plane meets the beach is water (wave crests included), everything
landward is land. Each water/land transition is marked on the profile, which
on a shore-normal transect should be a single point at the plane. Reading the
flags requires TransectLidarProfiles.py to have been run with DROP_WATER =
False; with DROP_WATER = True the wet samples are deleted and cannot be told
apart from ordinary gaps in coverage.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from BuildWaterMaskIndex import INDEX_GPKG, TILE_LAYER, is_skipped_scan

REGION_NAME = "island"

PROFILE_DIR = Path(r"D:\Data\Lidar\county_change") / REGION_NAME / "profiles"

# Water planes detected by BuildWaterMaskIndex.py.
WATER_MASK_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
WATER_SCAN_LAYER = "water_scans"
# Each tile can carry its own plane, so the one drawn is that of the tile under
# the transect. Falls back to the scan plane when no tile can be matched.
WATER_TILE_LAYER = "water_tiles"

TIDAL_DATUM_GPKG = (
    Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
    / REGION_NAME
    / "transects.gpkg"
)
TIDAL_DATUM_LAYER = "transect_bases_tidal_datums"

# Transect to plot, matching the id columns carried through from the transects.
PATH_ID = 14
SAMPLE_ID = 139691

# None = show the figure interactively, otherwise a directory to save a PNG in.
OUT_DIR = r"N:\SEA\Kai\Figures\LIdar\TransectPlots"

SHOW_WATER_PLANE = True
# Mark water/land transitions and report samples on the "wrong" side of the
# plane: wet above it (seaward roughness, filled holes) and dry at or below it
# (low ground not connected to open water).
SHOW_MASK_DIAGNOSTICS = True

# Datum whose contour position is tracked through time: "MLLW", "LMSL" or "MHHW".
CROSSING_DATUM = "MHHW"
# Transects extend 150 m landward and seaward of the shoreline, so the shoreline is at 150 m.
SHORELINE_DIST_M = 150.0
POINT_SIGMA_Z_M = (
    0.20 / 1.96
)  # 1-sigma vertical lidar uncertainty (20 cm is the 95% CI)
SLOPE_WINDOW_M = (
    2.0  # half-width of the window used for the local beach slope at a crossing
)

# Trailing 4-digit year in the scan folder name, used for ordering and colour.
YEAR_PATTERN = re.compile(r"(\d{4})")


def scan_year(scan_name):
    years = YEAR_PATTERN.findall(scan_name)
    return int(years[-1]) if years else None


def load_profiles(path_id, sample_id):
    """Every scan's profile for one transect, plus files written without flags."""
    wanted = ["path_id", "sample_id", "scan", "dist_m", "x", "y", "z", "water"]
    frames, unflagged = [], []
    for path in sorted(PROFILE_DIR.glob("*_profiles.parquet")):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if "water" not in names:
            unflagged.append(path.name)
        df = pd.read_parquet(
            path,
            columns=[c for c in wanted if c in names],
            filters=[("path_id", "==", path_id), ("sample_id", "==", sample_id)],
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit(
            f"No profiles found for path_id={path_id}, sample_id={sample_id}\n"
            f"searched {PROFILE_DIR}"
        )
    profiles = pd.concat(frames, ignore_index=True)
    if "water" not in profiles:
        profiles["water"] = False
    profiles["water"] = profiles["water"].fillna(False).astype(bool)
    return profiles.sort_values(["scan", "dist_m"]), unflagged


def load_water_planes():
    """Detected scan-level water level and method, keyed by scan name."""
    if not SHOW_WATER_PLANE or not WATER_MASK_GPKG.exists():
        return {}
    import geopandas as gpd

    water = gpd.read_file(WATER_MASK_GPKG, layer=WATER_SCAN_LAYER)
    if "z_tolerance" in water:
        print(
            f"WARNING: {WATER_MASK_GPKG.name} still has a z_tolerance column, so "
            "it predates the water-plane cut. Re-run BuildWaterMaskIndex.py"
        )
    return {
        r["scan"]: (r["water_z"], r["method"] if "method" in water else "")
        for _, r in water.iterrows()
    }


def load_tile_planes(profiles):
    """(water_z, plane source) of the tile under this transect, keyed by scan name.

    Tiles overlap, so the tile whose footprint holds the most of a scan's
    samples is used.
    """
    if not SHOW_WATER_PLANE or not WATER_MASK_GPKG.exists():
        return {}
    if not {"x", "y"} <= set(profiles) or not TIDAL_DATUM_GPKG.exists():
        return {}
    import geopandas as gpd
    import pyogrio
    import shapely

    if WATER_TILE_LAYER not in pyogrio.list_layers(WATER_MASK_GPKG)[:, 0]:
        return {}
    tile_water = pyogrio.read_dataframe(
        WATER_MASK_GPKG, layer=WATER_TILE_LAYER, read_geometry=False
    )
    if not {"scan", "path", "water_z"} <= set(tile_water):
        return {}
    if "plane" not in tile_water:
        tile_water["plane"] = ""

    # Assumes the profile x/y share the CRS of the transect layers in this GeoPackage.
    crs = pyogrio.read_info(TIDAL_DATUM_GPKG, layer=TIDAL_DATUM_LAYER)["crs"]
    points = gpd.GeoSeries(
        gpd.points_from_xy(profiles["x"], profiles["y"]), crs=crs
    )
    index_crs = pyogrio.read_info(INDEX_GPKG, layer=TILE_LAYER)["crs"]
    points = points.to_crs(index_crs)
    footprints = gpd.read_file(
        INDEX_GPKG,
        layer=TILE_LAYER,
        bbox=tuple(points.total_bounds),
        columns=["scan", "path"],
    )
    footprints = footprints.merge(
        tile_water[["scan", "path", "water_z", "plane"]], on=["scan", "path"]
    )

    scans = profiles["scan"].to_numpy()
    planes = {}
    for scan, tiles in footprints.groupby("scan"):
        pts = points.values[scans == scan]
        if not len(pts):
            continue
        counts = [
            int(shapely.contains(geom, pts).sum()) for geom in tiles.geometry.values
        ]
        best = int(np.argmax(counts))
        if counts[best]:
            row = tiles.iloc[best]
            planes[scan] = (row["water_z"], row["plane"])
    return planes


def load_tidal_datums(path_id, sample_id):
    """(MLLW, LMSL, MHHW) in m NAVD88 for one transect, or None if unavailable."""
    if not TIDAL_DATUM_GPKG.exists():
        return None
    import pyogrio

    cols = ["MLLW_navd88_m", "LMSL_navd88_m", "MHHW_navd88_m"]
    df = pyogrio.read_dataframe(
        TIDAL_DATUM_GPKG,
        layer=TIDAL_DATUM_LAYER,
        columns=cols,
        where=f"path_id = {int(path_id)} AND sample_id = {int(sample_id)}",
        read_geometry=False,
    )
    if df.empty or df[cols].iloc[0].isna().any():
        return None
    return tuple(df[cols].iloc[0])


def mask_transitions(dist, wet):
    """dist_m midway between each pair of adjacent samples where wet flips."""
    flip = np.flatnonzero(wet[:-1] != wet[1:])
    return (dist[flip] + dist[flip + 1]) / 2


def seaward_crossing(dist, z, wet, level):
    """dist_m of the crossing of `level` closest to SHORELINE_DIST_M, linearly interpolated.
    
    Note: wet flag is NOT applied here because datum crossings may occur at elevations
    where the water mask is active. The water mask is designed to account for tidal
    differences in satellite imagery, not to exclude data from topographic analysis.
    """
    d = z - level
    d0, d1 = d[:-1], d[1:]
    # Only adjacent valid samples, so crossings are never interpolated across gaps.
    hit = np.isfinite(d0) & np.isfinite(d1) & ((d0 == 0) | (np.sign(d0) != np.sign(d1)))
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return np.nan
    t = np.divide(
        d0[idx], d0[idx] - d1[idx], out=np.zeros(idx.size), where=d0[idx] != 0
    )
    xs = dist[idx] + t * (dist[idx + 1] - dist[idx])
    return xs[np.argmin(np.abs(xs - SHORELINE_DIST_M))]


def local_slope(dist, z, wet, x0, window):
    """Dry beach slope (m/m) by least squares within +/- window m of x0."""
    ok = ~wet & np.isfinite(z) & (np.abs(dist - x0) <= window)
    if ok.sum() < 3:
        return np.nan
    return np.polyfit(dist[ok], z[ok], 1)[0]


#################### Main ######################


profiles, unflagged = load_profiles(PATH_ID, SAMPLE_ID)
planes = load_water_planes()
tile_planes = load_tile_planes(profiles)
datums = load_tidal_datums(PATH_ID, SAMPLE_ID)


def plane_for(scan):
    """(water_z, method, source) drawn for a scan; tile plane first."""
    water_z, method = planes.get(scan, (None, ""))
    if scan in tile_planes:
        tile_z, source = tile_planes[scan]
        return tile_z, method, source or "tile"
    return water_z, method, "scan" if water_z is not None else ""


if unflagged:
    print(
        f"WARNING: {len(unflagged)} profile file(s) have no 'water' column, so "
        "their wet samples were dropped rather than flagged.\n"
        "         Re-run TransectLidarProfiles.py with DROP_WATER = False to "
        "QA the mask."
    )

scans = sorted(profiles["scan"].unique(), key=lambda s: (scan_year(s) or 0, s))
colours = plt.cm.viridis(np.linspace(0, 1, max(2, len(scans))))

fig, ax = plt.subplots(figsize=(12, 7))
if datums is not None:
    mllw, lmsl, mhhw = datums
    ax.axhspan(mllw, mhhw, color="tab:blue", alpha=0.12, linewidth=0, zorder=0)
    ax.axhline(lmsl, color="tab:blue", linewidth=1.0, linestyle="--", zorder=0)
    print(f"Tidal datums: MLLW {mllw:+.2f}  LMSL {lmsl:+.2f}  MHHW {mhhw:+.2f} m")
else:
    print("WARNING: no tidal datums found for this transect")

for scan, colour in zip(scans, colours):
    profile = profiles[profiles["scan"] == scan]
    dist = profile["dist_m"].to_numpy()
    z = profile["z"].to_numpy()
    wet = profile["water"].to_numpy()

    water_z, method, source = plane_for(scan)
    short_method = (
        "fallback"
        if method.startswith("fallback")
        else "detected"
        if method.startswith("detected")
        else method
    )
    if source.startswith("tile"):
        short_method = "tile"
    label = scan if water_z is None else f"{scan}  ({water_z:+.2f} m {short_method})"

    # Masked stretches sit under the line as a band, so they read as an
    # annotation of the profile rather than as a separate series.
    if wet.any():
        ax.plot(
            dist,
            np.where(wet, z, np.nan),
            color=colour,
            linewidth=6,
            alpha=0.30,
            solid_capstyle="butt",
            zorder=1,
        )
    ax.plot(dist, z, color=colour, linewidth=1.3, label=label, zorder=2)

    if water_z is not None:
        ax.axhline(water_z, color=colour, linewidth=0.7, alpha=0.5, zorder=0)

    if SHOW_MASK_DIAGNOSTICS:
        for x in mask_transitions(dist, wet):
            ax.axvline(x, color=colour, linewidth=0.8, linestyle=":", zorder=1)

print(f"{len(scans)} scans, {len(profiles)} samples for this transect")
for scan in scans:
    sub = profiles[profiles["scan"] == scan]
    water_z, method, source = plane_for(scan)
    if water_z is not None:
        plane = f"{water_z:+6.2f} {method} ({source} plane)"
    elif is_skipped_scan(scan):
        plane = "unmasked by design (topobathy)"
    else:
        plane = "no water in mask"
    print(f"  {scan:40s} {int(sub['water'].sum()):5d}/{len(sub):5d} wet   {plane}")
    if SHOW_MASK_DIAGNOSTICS and water_z is not None:
        dist = sub["dist_m"].to_numpy()
        z = sub["z"].to_numpy()
        wet = sub["water"].to_numpy()
        flips = mask_transitions(dist, wet)
        wet_above = int((wet & (z > water_z)).sum())
        dry_below = int((~wet & (z <= water_z)).sum())
        print(
            f"  {'':40s} waterline at {', '.join(f'{x:.1f}' for x in flips) or 'none'} m; "
            f"{wet_above} wet above plane, {dry_below} dry at/below plane"
        )

handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="0.4", linewidth=6, alpha=0.30))
labels.append("flagged wet")
if planes or tile_planes:
    handles.append(Line2D([], [], color="0.4", linewidth=0.7, alpha=0.5))
    labels.append("water plane")
if SHOW_MASK_DIAGNOSTICS and profiles["water"].any():
    handles.append(Line2D([], [], color="0.4", linewidth=0.8, linestyle=":"))
    labels.append("water/land boundary")
if datums is not None:
    handles += [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.12, linewidth=0),
        Line2D([], [], color="tab:blue", linewidth=1.0, linestyle="--"),
    ]
    labels += ["tidal region (MLLW\u2013MHHW)", "LMSL"]

ax.set_xlabel("Distance along transect (m)")
ax.set_ylabel("Elevation (m NAVD88)")
ax.set_title(f"Transect path_id={PATH_ID}, sample_id={SAMPLE_ID}")
ax.grid(alpha=0.3)
ax.legend(handles, labels, fontsize=10, loc="best", framealpha=0.9)
ax.set_xlim(50, 200)

fig.tight_layout()

# Datum contour position through time.
fig2 = None
if datums is not None:
    level = dict(zip(["MLLW", "LMSL", "MHHW"], datums))[CROSSING_DATUM]
    rows = []
    for scan in scans:
        year = scan_year(scan)
        if year is None:
            continue
        p = profiles[profiles["scan"] == scan]
        x = seaward_crossing(
            p["dist_m"].to_numpy(), p["z"].to_numpy(), p["water"].to_numpy(), level
        )
        if np.isfinite(x):
            beach_slope = local_slope(
                p["dist_m"].to_numpy(float),
                p["z"].to_numpy(float),
                p["water"].to_numpy(bool),
                x,
                SLOPE_WINDOW_M,
            )
            rows.append(
                {
                    "scan": scan,
                    "year": year + 0.5,
                    "dist_m": x,
                    "beach_slope": beach_slope,
                    "sigma_x_m": POINT_SIGMA_Z_M / abs(beach_slope)
                    if np.isfinite(beach_slope) and beach_slope != 0
                    else np.nan,
                }
            )
    crossings = pd.DataFrame(rows)
    print(f"\n{CROSSING_DATUM} ({level:+.2f} m) crossings:")
    print(crossings.to_string(index=False) if not crossings.empty else "  none")

    if len(crossings) >= 2 and crossings["sigma_x_m"].isna().any():
        print("Dropping crossings without a usable local beach slope")
        crossings = crossings.dropna(subset=["sigma_x_m"])

    if len(crossings) >= 2:
        from scipy import stats

        x = crossings["year"].to_numpy(float)
        y = crossings["dist_m"].to_numpy(float)
        sig = crossings["sigma_x_m"].to_numpy(float)
        n = len(x)
        w = 1 / sig**2

        # Weighted fit; each point's horizontal sigma = vertical sigma / beach slope.
        # The variance is scaled by reduced chi-square (when > 1) so scatter beyond
        # the propagated measurement error also widens the interval.
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        xw = np.sum(w * x) / np.sum(w)
        sxx = np.sum(w * (x - xw) ** 2)
        dof = n - 2
        chi2_red = (
            np.sum(w * (y - (slope * x + intercept)) ** 2) / dof if dof > 0 else 1.0
        )
        scale = max(1.0, chi2_red)
        slope_se = np.sqrt(scale / sxx)
        tcrit = stats.t.ppf(0.975, dof) if dof > 0 else 1.96
        slope_ci = tcrit * slope_se

        xs = np.linspace(x.min(), x.max(), 100)
        band_se = np.sqrt(scale * (1 / np.sum(w) + (xs - xw) ** 2 / sxx))

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.fill_between(
            xs,
            slope * xs + intercept - tcrit * band_se,
            slope * xs + intercept + tcrit * band_se,
            alpha=0.2,
            color="C0",
            label="95% CI of trend",
        )
        ax2.plot(xs, slope * xs + intercept, color="C0")
        ax2.errorbar(
            x,
            y,
            yerr=1.96 * sig,
            fmt="o",
            color="C0",
            capsize=3,
            label="Crossing (95% CI from vertical error)",
        )
        change = "Erosion" if slope > 0 else "Accretion"
        ax2.text(
            0.02,
            0.97,
            f"Shoreline Change = {abs(slope) * 100:.0f} ± {slope_ci * 100:.0f} cm/yr "
            f"{change}\n(95% CI, n={n})",
            transform=ax2.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
        ax2.set_xlabel("Scan year")
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax2.set_ylabel(f"Distance along transect of {CROSSING_DATUM} (m)")
        ax2.set_title(
            f"{CROSSING_DATUM} ({level:+.2f} m NAVD88) position, "
            f"path_id={PATH_ID}, sample_id={SAMPLE_ID}"
        )
        ax2.grid(alpha=0.3)
        ax2.legend(loc="lower right")
        fig2.tight_layout()
    else:
        print("Fewer than 2 crossings, skipping regression plot")

if OUT_DIR is None:
    plt.show()
else:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"transect_{REGION_NAME}{PATH_ID}_{SAMPLE_ID}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    if fig2 is not None:
        out_path = (
            out_dir / f"transect_{REGION_NAME}_{PATH_ID}_{SAMPLE_ID}_{CROSSING_DATUM}_trend.png"
        )
        fig2.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
