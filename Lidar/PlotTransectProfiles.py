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
from matplotlib.lines import Line2D

REGION_NAME = "king"

PROFILE_DIR = Path(r"D:\Data\Lidar_CoastalChange\county_change") / REGION_NAME

# Water planes detected by BuildWaterMaskIndex.py.
WATER_MASK_GPKG = Path(r"D:\Data\Lidar\lidar_water_mask_dtm.gpkg")
WATER_SCAN_LAYER = "water_scans"

# Transect to plot, matching the id columns carried through from the transects.
PATH_ID = 1
SAMPLE_ID = 21848

# None = show the figure interactively, otherwise a directory to save a PNG in.
OUT_DIR = None

SHOW_WATER_PLANE = True
# Tolerance BuildWaterMaskIndex.py added above the plane before masking. Drawn
# as a second rule so the level the mask actually cut at is visible.
Z_TOLERANCE = 0.25

# Trailing 4-digit year in the scan folder name, used for ordering and colour.
YEAR_PATTERN = re.compile(r"(\d{4})")


def scan_year(scan_name):
    years = YEAR_PATTERN.findall(scan_name)
    return int(years[-1]) if years else None


def load_profiles(path_id, sample_id):
    """Every scan's profile for one transect, plus files written without flags."""
    wanted = ["path_id", "sample_id", "scan", "dist_m", "z", "water"]
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
    """Detected water level and method per scan, keyed by scan name."""
    if not SHOW_WATER_PLANE or not WATER_MASK_GPKG.exists():
        return {}
    import geopandas as gpd

    water = gpd.read_file(WATER_MASK_GPKG, layer=WATER_SCAN_LAYER)
    return {
        r["scan"]: (r["water_z"], r["method"] if "method" in water else "")
        for _, r in water.iterrows()
    }


def main():
    profiles, unflagged = load_profiles(PATH_ID, SAMPLE_ID)
    planes = load_water_planes()

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
    for scan, colour in zip(scans, colours):
        profile = profiles[profiles["scan"] == scan]
        dist = profile["dist_m"].to_numpy()
        z = profile["z"].to_numpy()
        wet = profile["water"].to_numpy()

        water_z, method = planes.get(scan, (None, ""))
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
            ax.axhline(
                water_z + Z_TOLERANCE,
                color=colour,
                linewidth=0.7,
                alpha=0.5,
                linestyle=":",
                zorder=0,
            )

    print(f"{len(scans)} scans, {len(profiles)} samples for this transect")
    for scan in scans:
        sub = profiles[profiles["scan"] == scan]
        water_z, method = planes.get(scan, (None, ""))
        plane = f"{water_z:+6.2f} {method}" if water_z is not None else "no mask entry"
        print(f"  {scan:40s} {int(sub['water'].sum()):5d}/{len(sub):5d} wet   {plane}")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", linewidth=6, alpha=0.30))
    labels.append("flagged wet")
    if planes:
        handles += [
            Line2D([], [], color="0.4", linewidth=0.7, alpha=0.5),
            Line2D([], [], color="0.4", linewidth=0.7, alpha=0.5, linestyle=":"),
        ]
        labels += ["water plane", f"plane + {Z_TOLERANCE} m"]

    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("Elevation (m NAVD88)")
    ax.set_title(f"Transect path_id={PATH_ID}, sample_id={SAMPLE_ID}")
    ax.grid(alpha=0.3)
    ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()

    if OUT_DIR is None:
        plt.show()
    else:
        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"transect_{PATH_ID}_{SAMPLE_ID}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
