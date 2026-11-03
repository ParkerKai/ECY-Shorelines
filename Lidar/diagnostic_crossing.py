"""
Diagnostic script to debug seaward_crossing() detection.
Run this to understand why crossings aren't being found.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Copy these from PlotTransectProfiles.py
REGION_NAME = "island"
PROFILE_DIR = Path(r"D:\Data\Lidar\county_change") / REGION_NAME / "profiles"
CROSSING_DATUM = "MHHW"
SHORELINE_DIST_M = 150.0
PATH_ID = 14
SAMPLE_ID = 139691

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
    return profiles, unflagged


def load_tidal_datums(path_id, sample_id):
    """(MLLW, LMSL, MHHW) in m NAVD88 for one transect, or None if unavailable."""
    import pyogrio
    
    TIDAL_DATUM_GPKG = (
        Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
        / REGION_NAME
        / "transects.gpkg"
    )
    TIDAL_DATUM_LAYER = "transect_bases_tidal_datums"
    
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


# Load data
print(f"Loading profiles for path_id={PATH_ID}, sample_id={SAMPLE_ID}...")
profiles, unflagged = load_profiles(PATH_ID, SAMPLE_ID)
datums = load_tidal_datums(PATH_ID, SAMPLE_ID)

if datums is None:
    raise SystemExit("No tidal datums found")

mllw, lmsl, mhhw = datums
print(f"Tidal datums: MLLW {mllw:+.2f}  LMSL {lmsl:+.2f}  MHHW {mhhw:+.2f} m")

scans = sorted(profiles["scan"].unique(), key=lambda s: (scan_year(s) or 0, s))

# Test each datum level
for datum_name, level in [("MLLW", mllw), ("LMSL", lmsl), ("MHHW", mhhw)]:
    print(f"\n{'='*80}")
    print(f"TESTING {datum_name} at level={level:+.2f} m")
    print(f"{'='*80}")
    
    for scan in scans:
        p = profiles[profiles["scan"] == scan]
        dist = p["dist_m"].to_numpy()
        z = p["z"].to_numpy()
        wet = p["water"].to_numpy()
        
        # Show what the function sees
        z_masked = np.where(wet, np.nan, z)
        d = z_masked - level
        d0, d1 = d[:-1], d[1:]
        
        print(f"\n{scan}:")
        print(f"  Total samples: {len(z)}")
        print(f"  Wet samples: {wet.sum()}")
        print(f"  Finite pairs (d0 & d1): {(np.isfinite(d0) & np.isfinite(d1)).sum()}")
        
        # Find what should be crossings
        potential = np.isfinite(d0) & np.isfinite(d1)
        if not potential.any():
            print(f"  → NO FINITE PAIRS (likely all data is wet)")
            continue
        
        idx = np.where(potential)[0]
        
        # Look for sign changes and near-zero values
        sign_changes = []
        for i in idx:
            sign_d0 = np.sign(d0[i])
            sign_d1 = np.sign(d1[i])
            is_zero = (d0[i] == 0)
            sign_change = (sign_d0 != sign_d1)
            
            if is_zero or sign_change:
                t = d0[i] / (d0[i] - d1[i]) if (d0[i] - d1[i]) != 0 else 0
                x = dist[i] + t * (dist[i + 1] - dist[i])
                sign_changes.append({
                    'i': i,
                    'dist_range': (dist[i], dist[i+1]),
                    'z_range': (z[i], z[i+1]),
                    'd_range': (d0[i], d1[i]),
                    'crossing_x': x,
                    'hit_reason': 'zero' if is_zero else 'sign_change'
                })
        
        if sign_changes:
            print(f"  → FOUND {len(sign_changes)} crossing(s):")
            for hit in sign_changes:
                print(f"      i={hit['i']:3d}: dist=[{hit['dist_range'][0]:7.1f}, {hit['dist_range'][1]:7.1f}], "
                      f"z=[{hit['z_range'][0]:+7.2f}, {hit['z_range'][1]:+7.2f}], "
                      f"d=[{hit['d_range'][0]:+.3f}, {hit['d_range'][1]:+.3f}] → "
                      f"crossing at x={hit['crossing_x']:7.1f} ({hit['hit_reason']})")
                
                # Find closest to SHORELINE
                closest_idx = np.argmin([abs(hit['crossing_x'] - SHORELINE_DIST_M) for hit in sign_changes])
                closest = sign_changes[closest_idx]
                print(f"  → CLOSEST TO SHORELINE ({SHORELINE_DIST_M} m): x={closest['crossing_x']:.1f} m")
        else:
            print(f"  → NO SIGN CHANGES DETECTED")
            # Show some near-misses
            print(f"  → Showing first 10 finite pairs (looking for crossings):")
            for i in idx[:10]:
                print(f"      i={i:3d}: z=[{z[i]:+7.2f}, {z[i+1]:+7.2f}], "
                      f"d=[{d0[i]:+.3f}, {d1[i]:+.3f}], "
                      f"sign=[{int(np.sign(d0[i])):+d}, {int(np.sign(d1[i])):+d}]")
