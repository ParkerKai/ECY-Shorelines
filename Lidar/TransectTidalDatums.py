"""Compute NOAA tidal datum elevations at each transect base point.

Loads the `transect_bases` layer from a county's transects.gpkg and uses
vyperdatum (NOAA VDatum grids in PROJ) to find the elevation of every available
tidal datum relative to NAVD88 at each point. Results are written as a new
layer in the same transects.gpkg.

RUN IN TERMINAL, NOT IN AN INTERACTIVE KERNEL.

Requires vyperdatum with the NOAA grids/proj.db downloaded to VYPER_GRIDS_DIR.
"""

import os
import re
from pathlib import Path

VYPER_GRIDS_DIR = Path(r"D:\Data\VLM\vyper_grids")
# vyperdatum reads VYPER_GRIDS at import time; PROJ_DATA must match before pyproj loads.
os.environ["VYPER_GRIDS"] = str(VYPER_GRIDS_DIR)
os.environ["PROJ_DATA"] = str(VYPER_GRIDS_DIR)

import pyproj

# Pin PROJ to the vyper grids alone. pyproj caches its data dir on first use, so
# PROJ_DATA is ignored if pyproj was already loaded in this kernel, and
# vyperdatum only prepends its dir to the env default. If the vyper proj.db then
# cannot be opened, PROJ silently falls back to the env proj.db, which has no
# NOAA authority, and vyperdatum fails its import with an unhelpful assertion.
pyproj.datadir.set_data_dir(str(VYPER_GRIDS_DIR))
if "NOAA" not in pyproj.database.get_authorities():
    raise RuntimeError(
        f"PROJ could not load the NOAA authority from {VYPER_GRIDS_DIR / 'proj.db'}. "
        "Check the file exists, is readable and not locked by another process, "
        "then restart the kernel."
    )

from vyperdatum.transformer import Transformer

import geopandas as gpd
import numpy as np
from pyproj import Transformer as PyprojTransformer
from pyproj.database import query_crs_info
from pyproj.enums import PJType
from shapely import STRtree, points

# vyperdatum enables PROJ CDN fetching, which fails here; use only the local grids.
pyproj.network.set_network_enabled(False)

COUNTY_ROOT = Path(r"N:\SEA\Kai\Data\BarriersToMigration\counties_v4")
TRANSECT_GPKG_NAME = "transects.gpkg"
TRANSECT_LAYER = "transect_bases"
TRANSECT_LINES_LAYER = "transects_all"
KEYS = ["path_id", "sample_id"]

# Seaward search for bases outside NOAA grid coverage; may extend past the transect end.
SEAWARD_STEP_M = 10.0
MAX_SEAWARD_M = 1000.0
# method: grid (at base), seaward (along transect), nearest (QAQC: nearest grid sample).
METHOD_COL = "tidal_fill_method"
DIST_COL = "tidal_fill_dist_m"
QAQC_COL = "tidal_qaqc"

county_list = [
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

# NAD83(2011) geographic; datum elevations are reported relative to NAVD88.
GEOG_CRS = "EPSG:6318"
REFERENCE_VCRS = "EPSG:5703"
REFERENCE_LABEL = "navd88"

TIDAL_DATUMS = ["MLLW", "MLW", "MTL", "DTL", "LMSL", "MSL", "MHW", "MHHW"]

OUT_LAYER = "transect_bases_tidal_datums"


def find_datum_crs_codes() -> dict[str, list[str]]:
    """Map each tidal datum to candidate NOAA vertical CRS codes in vyperdatum's proj.db."""
    infos = query_crs_info(auth_name="NOAA", pj_types=PJType.VERTICAL_CRS)
    candidates: dict[str, list] = {d: [] for d in TIDAL_DATUMS}
    for info in infos:
        name = info.name.upper()
        # Foot-unit CRSs have no transformation to NAVD88 (meters) in proj.db.
        if "(FT)" in name:
            continue
        for datum in TIDAL_DATUMS:
            if re.search(rf"\b{datum}\b", name):
                candidates[datum].append((info, name))

    # Prefer height (up) CRSs over depth variants; z=0 gives the same answer either way.
    codes = {}
    for datum, matches in candidates.items():
        matches.sort(key=lambda m: ("DEPTH" in m[1], m[0].code))
        codes[datum] = [f"{m[0].auth_name}:{m[0].code}" for m in matches]
        if codes[datum]:
            print(f"{datum}: {[(c, m[0].name) for c, m in zip(codes[datum], matches)]}")
    return {d: c for d, c in codes.items() if c}


def build_transformer(vcrs: str) -> Transformer | None:
    try:
        return Transformer(
            crs_from=f"{GEOG_CRS}+{vcrs}",
            crs_to=f"{GEOG_CRS}+{REFERENCE_VCRS}",
        )
    except Exception as exc:
        print(f"    {vcrs}: could not build transformer ({exc})")
        return None


def datum_elevation(
    tf: Transformer, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray | None:
    """Elevation of the datum surface (z=0 in the source VCRS) relative to the reference VCRS."""
    success, _, _, z = tf.transform_points(
        x=lon.copy(), y=lat.copy(), z=np.zeros_like(lon), always_xy=True
    )
    if not success or z is None:
        return None
    z = np.asarray(z, dtype=float)
    z[~np.isfinite(z)] = np.nan  # outside grid coverage
    return z


def seaward_search(
    bases: gpd.GeoDataFrame,
    transects: gpd.GeoDataFrame,
    datum_tfs: dict[str, Transformer],
    to_geog: PyprojTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Step seaward along each transect until all datums have grid values.

    Returns the x/y of the grid sample used for each base (NaN if none).
    """
    cols = list(datum_tfs)
    missing = bases[cols].isna().any(axis=1).to_numpy()
    bases[METHOD_COL] = np.where(missing, "none", "grid")
    bases[DIST_COL] = np.where(missing, np.nan, 0.0)
    sx = np.where(missing, np.nan, bases.geometry.x.to_numpy())
    sy = np.where(missing, np.nan, bases.geometry.y.to_numpy())
    if not missing.any():
        return sx, sy

    sub = (
        bases.loc[missing, KEYS]
        .reset_index()
        .merge(transects[KEYS + ["geometry"]], on=KEYS, how="inner")
    )
    rows = sub["index"].to_numpy()
    ox = bases.geometry.x.to_numpy()[rows]
    oy = bases.geometry.y.to_numpy()[rows]
    # transects_all vertex order is seaward -> origin -> landward.
    sea = np.array([g.coords[0] for g in sub.geometry])
    dx, dy = sea[:, 0] - ox, sea[:, 1] - oy
    norm = np.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm

    pending = np.ones(len(sub), dtype=bool)
    for d in np.arange(SEAWARD_STEP_M, MAX_SEAWARD_M + 1e-9, SEAWARD_STEP_M):
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        px, py = ox[idx] + ux[idx] * d, oy[idx] + uy[idx] * d
        lon, lat = to_geog.transform(px, py)
        lon, lat = np.asarray(lon), np.asarray(lat)
        vals = {}
        for c, tf in datum_tfs.items():
            z = datum_elevation(tf, lon, lat)
            vals[c] = np.full(idx.size, np.nan) if z is None else z
        good = np.all([np.isfinite(v) for v in vals.values()], axis=0)
        hit = rows[idx[good]]
        for c, v in vals.items():
            bases.loc[hit, c] = v[good]
        bases.loc[hit, DIST_COL] = d
        bases.loc[hit, METHOD_COL] = "seaward"
        sx[hit], sy[hit] = px[good], py[good]
        pending[idx[good]] = False

    found = bases.loc[rows, DIST_COL].dropna()
    print(
        f"Seaward search: filled {len(found)}/{missing.sum()} points "
        f"(median {found.median():.0f} m, max {found.max():.0f} m); "
        f"{missing.sum() - len(found)} still missing"
    )
    return sx, sy


def nearest_fill(
    bases: gpd.GeoDataFrame, cols: list[str], sx: np.ndarray, sy: np.ndarray
) -> None:
    """Fill remaining bases from the nearest grid sample location and flag them for QAQC."""
    missing = bases[cols].isna().any(axis=1).to_numpy()
    bases[QAQC_COL] = 0
    has = np.isfinite(sx)
    if not missing.any() or not has.any():
        return
    src_idx = np.flatnonzero(has)
    tree = STRtree(points(sx[src_idx], sy[src_idx]))
    (q, t), dist = tree.query_nearest(
        bases.geometry.values[missing], return_distance=True, all_matches=False
    )
    miss_idx = np.flatnonzero(missing)[q]
    src = src_idx[t]
    for c in cols:
        bases.loc[miss_idx, c] = bases[c].to_numpy()[src]
    bases.loc[miss_idx, DIST_COL] = dist
    bases.loc[miss_idx, METHOD_COL] = "nearest"
    bases.loc[miss_idx, QAQC_COL] = 1
    print(
        f"Nearest fill: {len(miss_idx)} points flagged for QAQC "
        f"(median {np.median(dist):.0f} m, max {dist.max():.0f} m)"
    )


def main() -> None:
    
    for COUNTY in county_list:
        gpkg = COUNTY_ROOT / COUNTY / TRANSECT_GPKG_NAME
        print(f"Loading {TRANSECT_LAYER} from {gpkg}")
        bases = gpd.read_file(gpkg, layer=TRANSECT_LAYER)
        bases = bases[bases.geometry.notna() & ~bases.geometry.is_empty].reset_index(
            drop=True
        )
        if bases.empty:
            raise SystemExit("No transect base points found.")

        # Fall back to the ballpark NAD83 -> NAD83(2011) shift (~1 m) when the HARN grid is not local.
        to_geog = PyprojTransformer.from_crs(
            bases.crs, GEOG_CRS, always_xy=True, only_best=False
        )
        lon, lat = to_geog.transform(
            bases.geometry.x.to_numpy(), bases.geometry.y.to_numpy()
        )

        datum_codes = find_datum_crs_codes()
        if not datum_codes:
            raise SystemExit(
                f"No NOAA tidal datum CRSs found; check {VYPER_GRIDS_DIR} / proj.db."
            )

        datum_tfs: dict[str, Transformer] = {}
        for datum, codes in datum_codes.items():
            best, best_code, best_tf = None, None, None
            for code in codes:
                tf = build_transformer(code)
                if tf is None:
                    continue
                z = datum_elevation(tf, lon, lat)
                if z is not None and (
                    best is None or np.isfinite(z).sum() > np.isfinite(best).sum()
                ):
                    best, best_code, best_tf = z, code, tf
            col = f"{datum}_{REFERENCE_LABEL}_m"
            if best is None or not np.isfinite(best).any():
                print(f"{datum}: no valid values")
                continue
            bases[col] = best
            datum_tfs[col] = best_tf
            print(f"{datum} ({best_code}): {np.isfinite(best).sum()}/{len(best)} points")

        print(f"Loading {TRANSECT_LINES_LAYER} from {gpkg}")
        transects = gpd.read_file(gpkg, layer=TRANSECT_LINES_LAYER).drop_duplicates(KEYS)
        sx, sy = seaward_search(bases, transects, datum_tfs, to_geog)
        nearest_fill(bases, list(datum_tfs), sx, sy)

        # Writing a named layer replaces only that layer; other layers in the gpkg are kept.
        bases.to_file(gpkg, layer=OUT_LAYER, driver="GPKG")
        print(f"Wrote {len(bases)} points to {gpkg} (layer '{OUT_LAYER}')")


if __name__ == "__main__":
    main()
