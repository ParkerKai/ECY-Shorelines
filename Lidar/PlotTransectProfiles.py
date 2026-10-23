"""Plot every available lidar scan profile for a single transect on one axis.

Reads the per-scan Parquet files written by TransectLidarProfiles.py and
overlays each scan's elevation profile for the chosen transect, so changes in
the profile over time are visible directly.

Stretches flagged as water are drawn as a band over the line, and each scan's
detected water plane is drawn as a horizontal rule, so the mask can be checked
against the elevations it was derived from. Reading the flags requires
TransectLidarProfiles.py to have been run with DROP_WATER = False; with
DROP_WATER = True the wet samples are deleted and cannot be told apart from
ordinary gaps in coverage.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib.lines import Line2D

from BuildWaterMaskIndex import (
    CLOSING_M,
    REFERENCE_SIMPLIFY_M,
    REFERENCE_WATER,
    SEED_EROSION_M,
    Z_TOLERANCE,
)

REGION_NAME = "king"

PROFILE_DIR = Path(r"D:\Data\Lidar\county_change") / REGION_NAME / "profiles"

# Water planes detected by BuildWaterMaskIndex.py.
WATER_MASK_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
WATER_SCAN_LAYER = "water_scans"

TIDAL_DATUM_GPKG = (
    Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
    / REGION_NAME
    / "transects.gpkg"
)
TIDAL_DATUM_LAYER = "transect_bases_tidal_datums"

# Transect to plot, matching the id columns carried through from the transects.
PATH_ID = 0
SAMPLE_ID = 100

# None = show the figure interactively, otherwise a directory to save a PNG in.
OUT_DIR = None

SHOW_WATER_PLANE = True
# Attribute samples flagged wet above the water plane to the mask step that caught them.
SHOW_MASK_DIAGNOSTICS = True
CAUSE_MARKERS = {"reference core": "x", "closing": "^", "fill/other": "s"}

# Datum whose contour position is tracked through time: "MLLW", "LMSL" or "MHHW".
CROSSING_DATUM = "MHHW"

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
    """Detected water level, method and Z tolerance per scan, keyed by scan name."""
    if not SHOW_WATER_PLANE or not WATER_MASK_GPKG.exists():
        return {}
    import geopandas as gpd

    water = gpd.read_file(WATER_MASK_GPKG, layer=WATER_SCAN_LAYER)
    # Masks built before the per-scan tolerance existed used Z_TOLERANCE.
    return {
        r["scan"]: (
            r["water_z"],
            r["method"] if "method" in water else "",
            r["z_tolerance"] if "z_tolerance" in water else Z_TOLERANCE,
        )
        for _, r in water.iterrows()
    }


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


def load_reference_core(profiles):
    """Reference water eroded to the always-masked core, as BuildWaterMaskIndex.py builds it."""
    if REFERENCE_WATER is None or not Path(REFERENCE_WATER).exists():
        return None
    import geopandas as gpd
    import pyogrio
    import shapely
    from shapely.geometry import box

    # Assumes the profile x/y share the CRS of the transect layers in this GeoPackage.
    crs = pyogrio.read_info(TIDAL_DATUM_GPKG, layer=TIDAL_DATUM_LAYER)["crs"]
    area = box(
        profiles["x"].min(),
        profiles["y"].min(),
        profiles["x"].max(),
        profiles["y"].max(),
    ).buffer(2 * SEED_EROSION_M)
    ref = gpd.read_file(REFERENCE_WATER, bbox=gpd.GeoSeries([area], crs=crs))
    if ref.empty:
        return None
    geom = shapely.force_2d(ref.geometry.to_crs(crs).values)
    geom = shapely.make_valid(shapely.simplify(geom, REFERENCE_SIMPLIFY_M))
    return shapely.union_all(geom).buffer(-SEED_EROSION_M, quad_segs=2)


def wet_causes(dist, z, wet, in_core, water_z, tolerance):
    """Label each wet sample with the mask step most likely responsible."""
    cut = water_z + tolerance
    cause = np.full(z.shape, "", dtype=object)
    cause[wet & (z <= cut)] = "below plane"
    above = wet & (z > cut)
    cause[above & in_core] = "reference core"
    rest = np.flatnonzero(above & ~in_core)
    low = dist[z <= cut]
    if rest.size:
        # Along-profile distance only bounds the planar one, so "fill/other" may
        # still be closing across wet cells beside the transect. Closing only
        # bridges gaps up to 2 * CLOSING_M wide, so CLOSING_M from the nearest
        # below-plane sample is the most it can reach.
        gap = (
            np.abs(dist[rest, None] - low[None, :]).min(axis=1)
            if low.size
            else np.full(rest.size, np.inf)
        )
        cause[rest] = np.where(gap <= CLOSING_M, "closing", "fill/other")
    return cause


def seaward_crossing(dist, z, wet, level):
    """Smallest dist_m where the dry profile crosses `level`, linearly interpolated."""
    z = np.where(wet, np.nan, z)
    d = z - level
    d0, d1 = d[:-1], d[1:]
    # Only adjacent valid samples, so crossings are never interpolated across gaps.
    hit = np.isfinite(d0) & np.isfinite(d1) & ((d0 == 0) | (np.sign(d0) != np.sign(d1)))
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if d0[i] == 0:
        return dist[i]
    t = d0[i] / (d0[i] - d1[i])
    return dist[i] + t * (dist[i + 1] - dist[i])


#################### Main ######################


profiles, unflagged = load_profiles(PATH_ID, SAMPLE_ID)
planes = load_water_planes()
datums = load_tidal_datums(PATH_ID, SAMPLE_ID)

core = None
if SHOW_MASK_DIAGNOSTICS and {"x", "y"} <= set(profiles):
    import shapely

    core = load_reference_core(profiles)
    profiles["in_core"] = (
        shapely.intersects_xy(core, profiles["x"], profiles["y"])
        if core is not None
        else False
    )

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

    water_z, method, tolerance = planes.get(scan, (None, "", Z_TOLERANCE))
    label = scan if water_z is None else f"{scan}  ({water_z:+.2f} m {method})"

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

    if "in_core" in profile and water_z is not None and wet.any():
        cause = wet_causes(
            dist, z, wet, profile["in_core"].to_numpy(), water_z, tolerance
        )
        for name, marker in CAUSE_MARKERS.items():
            sel = cause == name
            if sel.any():
                ax.scatter(
                    dist[sel], z[sel], marker=marker, s=14, color=colour, zorder=3
                )
        counts = pd.Series(cause[wet]).value_counts()
        print(
            f"  {scan:40s} wet causes: "
            + ", ".join(f"{k} {v}" for k, v in counts.items())
        )

print(f"{len(scans)} scans, {len(profiles)} samples for this transect")
if SHOW_MASK_DIAGNOSTICS and core is None:
    print("  (no reference core here, or profiles lack x/y; core attribution skipped)")
for scan in scans:
    sub = profiles[profiles["scan"] == scan]
    water_z, method, tolerance = planes.get(scan, (None, "", None))
    plane = (
        f"{water_z:+6.2f} +{tolerance:.2f} {method}"
        if water_z is not None
        else "no mask entry"
    )
    print(f"  {scan:40s} {int(sub['water'].sum()):5d}/{len(sub):5d} wet   {plane}")

handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="0.4", linewidth=6, alpha=0.30))
labels.append("flagged wet")
if planes:
    handles.append(Line2D([], [], color="0.4", linewidth=0.7, alpha=0.5))
    labels.append("water plane")
if "in_core" in profiles:
    for name, marker in CAUSE_MARKERS.items():
        handles.append(Line2D([], [], color="0.4", marker=marker, linestyle=""))
        labels.append(f"wet above plane: {name}")
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
ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.9)
ax.set_xlim(100, 200)

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
            rows.append({"scan": scan, "year": year + 0.5, "dist_m": x})
    crossings = pd.DataFrame(rows)
    print(f"\n{CROSSING_DATUM} ({level:+.2f} m) crossings:")
    print(crossings.to_string(index=False) if not crossings.empty else "  none")

    if len(crossings) >= 2:
        slope, _ = np.polyfit(crossings["year"], crossings["dist_m"], 1)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.regplot(data=crossings, x="year", y="dist_m", ax=ax2)
        ax2.text(
            0.02,
            0.97,
            f"slope = {slope * 1000:+.1f} mm/yr  (n={len(crossings)})",
            transform=ax2.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
        ax2.set_xlabel("Scan year")
        ax2.set_ylabel(f"Distance along transect of {CROSSING_DATUM} (m)")
        ax2.set_title(
            f"{CROSSING_DATUM} ({level:+.2f} m NAVD88) position, "
            f"path_id={PATH_ID}, sample_id={SAMPLE_ID}"
        )
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
    else:
        print("Fewer than 2 crossings, skipping regression plot")

if OUT_DIR is None:
    plt.show()
else:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"transect_{PATH_ID}_{SAMPLE_ID}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    if fig2 is not None:
        out_path = (
            out_dir / f"transect_{PATH_ID}_{SAMPLE_ID}_{CROSSING_DATUM}_trend.png"
        )
        fig2.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
