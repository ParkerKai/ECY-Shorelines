"""Shoreline change from MHHW contours drawn on each lidar scan, per county.

Alternative to TransectShorelineChange.py. Instead of finding the datum crossing
on each transect profile and joining the crossings into a polyline, the MHHW
shoreline is contoured directly from each scan's DEM and then intersected with
the transects:

  1. Find every scan in the tile index whose tiles intersect the county's
     barrier_region (shorezone) polygon. Scans flagged in the uncertainty Excel
     file ("Flag Drop") are skipped.
  2. Mosaic each scan's tiles with gdal.Warp at the scan's native resolution
     (EPSG:26910, bilinear), cut to the shorezone, and convert z to metres.
  3. MHHW varies along the coast, so the transect-base MHHW values are
     interpolated (IDW) into a smooth surface that is subtracted from the DEM;
     the MHHW shoreline is then the 0 contour of that residual (gdal_contour,
     via gdal.ContourGenerateEx). Contour pieces shorter than MIN_CONTOUR_LEN_M
     are dropped.
  4. Each transect's crossing is its intersection with the contour; with several
     intersections, the one closest to the transect base is kept. The local
     beach slope is sampled from the scan DEM within +/- SLOPE_WINDOW_M of the
     crossing.
  5. Uncertainty and the weighted change-rate fit are the same as in
     TransectShorelineChange.py.

Scans are processed in parallel (N_WORKERS processes). Run from a terminal, not
an interactive kernel, so the worker processes can import this module.

Outputs, in PROFILE_ROOT / <region> / "shoreline":
  crossings_<datum>.parquet                   one row per transect x scan
  rates_<datum>.parquet                       one row per transect
  shoreline_positions_<datum>.gpkg            crossing points, one layer per scan
  shoreline_positions_polylines_<datum>.gpkg  MHHW contours, one layer per scan
  shoreline_positions_polylines_<datum>.shp   MHHW contours, all scans in one shapefile
  datum_surface_<datum>.tif                   interpolated datum surface (m NAVD88)
and in OUT_ROOT / <region>: shoreline_change.gpkg (layer shoreline_change_<datum>)
and shoreline_change_polylines_<datum>.shp, as in TransectShorelineChange.py.
"""

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from osgeo import gdal, ogr
from pyproj import CRS
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from TransectShorelineChange import (
    CROSSING_DATUM,
    KEYS,
    REGIONS,
    SHORELINE_DIST_M,
    SLOPE_WINDOW_M,
    TIDAL_DATUM_LAYER,
    TRANSECT_ROOT,
    change_rates,
    load_uncertainty,
    scan_year,
    write_change_polylines,
)

gdal.UseExceptions()

PROFILE_ROOT = Path(r"N:\SEA\Kai\Data\lidar\county_change_contour")
OUT_ROOT = Path(r"N:\SEA\Kai\Data\lidar\ShorelineChange_contour")
OUT_NAME = "shoreline_change.gpkg"

INDEX_GPKG = Path(r"D:\Data\Lidar\lidar_tile_index_dtm.gpkg")
TILE_LAYER = "tiles"

REGION_GPKG_NAME = "county_data.gpkg"
REGION_LAYER = "barrier_region"
TRANSECT_LAYER = "transects_all"

OUTPUT_CRS = "EPSG:26910"

# Per-scan DEMs are large; they are written here and deleted unless KEEP_DEMS.
TMP_ROOT = Path(r"D:\Data\Lidar\_contour_tmp")
KEEP_DEMS = False

N_WORKERS = 5

# Contour pieces shorter than this (m) are dropped: rings around rocks, logs, puddles.
MIN_CONTOUR_LEN_M = 10.0

# Elevations outside this range (m) are undeclared nodata sentinels.
VALID_Z_RANGE = (-200.0, 5000.0)
NODATA = -9999.0

# Datum surface: IDW of the transect-base values on a coarse grid, bilinear in between.
DATUM_GRID_RES_M = 50.0
DATUM_GRID_PAD_M = 1000.0
IDW_NEIGHBOURS = 8
IDW_POWER = 2.0

# Pixels processed at once when converting the DEM to the datum residual.
CHUNK = 4096


def shoreline_paths(region, datum=CROSSING_DATUM):
    out_dir = PROFILE_ROOT / region / "shoreline"
    return {
        "crossings": out_dir / f"crossings_{datum}.parquet",
        "rates": out_dir / f"rates_{datum}.parquet",
        "positions": out_dir / f"shoreline_positions_{datum}.gpkg",
        "positions_polylines": out_dir / f"shoreline_positions_polylines_{datum}.gpkg",
        "positions_polylines_shp": out_dir / f"shoreline_positions_polylines_{datum}.shp",
        "datum_surface": out_dir / f"datum_surface_{datum}.tif",
    }


def z_scale(crs):
    """Factor converting the tile's (horizontal = vertical) units to metres."""
    axis = CRS.from_user_input(crs).axis_info
    return axis[0].unit_conversion_factor if axis else 1.0


#################### Region inputs ######################


def load_region_geometry(region):
    gdf = pyogrio.read_dataframe(
        TRANSECT_ROOT / region / REGION_GPKG_NAME, layer=REGION_LAYER
    ).to_crs(OUTPUT_CRS)
    return shapely.make_valid(gdf.geometry.union_all())


def load_transects(region):
    """Transect lines with the datum level and the distance of the base along the line."""
    gpkg = TRANSECT_ROOT / region / "transects.gpkg"
    col = f"{CROSSING_DATUM}_navd88_m"
    bases = pyogrio.read_dataframe(
        gpkg, layer=TIDAL_DATUM_LAYER, columns=KEYS + ["measure_m", col]
    ).to_crs(OUTPUT_CRS)
    bases = bases.rename(columns={col: "datum_z"})

    lines = pyogrio.read_dataframe(gpkg, layer=TRANSECT_LAYER, columns=KEYS)
    lines = lines.to_crs(OUTPUT_CRS).drop_duplicates(KEYS)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    lines = lines.merge(
        pd.DataFrame(
            {
                "path_id": bases["path_id"],
                "sample_id": bases["sample_id"],
                "base_geom": bases.geometry.values,
            }
        ),
        on=KEYS,
        how="left",
    )
    has_base = lines["base_geom"].notna().to_numpy()
    base_dist = np.full(len(lines), SHORELINE_DIST_M)
    base_dist[has_base] = shapely.line_locate_point(
        lines.geometry.values[has_base], lines["base_geom"].values[has_base]
    )
    lines = gpd.GeoDataFrame(
        lines[KEYS].assign(base_dist=base_dist),
        geometry=lines.geometry.values,
        crs=OUTPUT_CRS,
    ).reset_index(drop=True)
    return bases, lines


def build_datum_surface(bases, bounds, path):
    """IDW of the base-point datum levels onto a coarse grid covering bounds."""
    ok = bases["datum_z"].notna().to_numpy()
    xy = np.column_stack(
        [bases.geometry.x.to_numpy()[ok], bases.geometry.y.to_numpy()[ok]]
    )
    z = bases["datum_z"].to_numpy(float)[ok]
    if z.size == 0:
        raise ValueError(f"no transect base has a {CROSSING_DATUM} level")

    res = DATUM_GRID_RES_M
    left = np.floor((bounds[0] - DATUM_GRID_PAD_M) / res) * res
    bottom = np.floor((bounds[1] - DATUM_GRID_PAD_M) / res) * res
    right = np.ceil((bounds[2] + DATUM_GRID_PAD_M) / res) * res
    top = np.ceil((bounds[3] + DATUM_GRID_PAD_M) / res) * res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))

    gx = left + (np.arange(width) + 0.5) * res
    gy = top - (np.arange(height) + 0.5) * res
    xx, yy = np.meshgrid(gx, gy)
    k = min(IDW_NEIGHBOURS, z.size)
    dist, idx = cKDTree(xy).query(np.column_stack([xx.ravel(), yy.ravel()]), k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1.0) ** IDW_POWER
    grid = ((w * z[idx]).sum(axis=1) / w.sum(axis=1)).reshape(height, width)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=OUTPUT_CRS,
        transform=rasterio.transform.from_origin(left, top, res, res),
        compress="deflate",
    ) as dst:
        dst.write(grid.astype("float32"), 1)
    print(f"  {CROSSING_DATUM} surface {width}x{height} at {res:g} m -> {path.name}")


#################### Per-scan worker ######################


def datum_interpolator(path):
    with rasterio.open(path) as src:
        grid = src.read(1).astype("float64")
        t = src.transform
    gx = t.c + (np.arange(grid.shape[1]) + 0.5) * t.a
    gy = t.f + (np.arange(grid.shape[0]) + 0.5) * t.e
    # RegularGridInterpolator needs ascending axes; rows run north to south.
    return RegularGridInterpolator(
        (gy[::-1], gx), grid[::-1], bounds_error=False, fill_value=None
    )


def mosaic_scan(tile_paths, cut_path, bounds, res, dem_path):
    gdal.Warp(
        str(dem_path),
        [str(p) for p in tile_paths],
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS=OUTPUT_CRS,
            xRes=res,
            yRes=res,
            targetAlignedPixels=True,
            outputBounds=bounds,
            cutlineDSName=str(cut_path),
            resampleAlg="bilinear",
            dstNodata=NODATA,
            outputType=gdal.GDT_Float32,
            warpMemoryLimit=1024,
            warpOptions=["INIT_DEST=NO_DATA"],
            creationOptions=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "SPARSE_OK=TRUE",
                "BIGTIFF=YES",
            ],
        ),
    )


def to_datum_residual(dem_path, zscale, interp):
    """Rewrite the DEM in place as z (m) minus the datum surface, so MHHW is 0."""
    valid = 0
    low, high = VALID_Z_RANGE
    with rasterio.open(dem_path, "r+") as src:
        t = src.transform
        for row in range(0, src.height, CHUNK):
            for col in range(0, src.width, CHUNK):
                win = Window(
                    col, row, min(CHUNK, src.width - col), min(CHUNK, src.height - row)
                )
                z = src.read(1, window=win)
                ok = z != NODATA
                if not ok.any():
                    continue
                zm = z.astype("float64") * zscale
                ok &= np.isfinite(zm) & (zm >= low) & (zm <= high)
                r, c = np.nonzero(ok)
                xs = t.c + (col + c + 0.5) * t.a
                ys = t.f + (row + r + 0.5) * t.e
                out = np.full(z.shape, NODATA, dtype="float32")
                out[r, c] = zm[r, c] - interp(np.column_stack([ys, xs]))
                src.write(out, 1, window=win)
                valid += r.size
    return valid


def contour_zero(dem_path, tmp_gpkg):
    src = gdal.Open(str(dem_path))
    band = src.GetRasterBand(1)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(tmp_gpkg))
    layer = ds.CreateLayer("contour", src.GetSpatialRef(), ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn("contour_id", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("level", ogr.OFTReal))
    gdal.ContourGenerateEx(
        band,
        layer,
        options=["FIXED_LEVELS=0", f"NODATA={NODATA}", "ID_FIELD=0", "ELEV_FIELD=1"],
    )
    ds = None
    src = None
    return pyogrio.read_dataframe(tmp_gpkg, layer="contour")


def contour_crossings(contours, transects):
    """Intersection of each transect with the contour closest to its base."""
    lines = contours.geometry.values
    tgeom = transects.geometry.values
    ti, ci = shapely.STRtree(lines).query(tgeom, predicate="intersects")
    if ti.size == 0:
        return pd.DataFrame()
    inter = shapely.intersection(tgeom[ti], lines[ci])
    xy, part = shapely.get_coordinates(inter, return_index=True)
    t = ti[part]
    df = pd.DataFrame(
        {
            "t": t,
            "dist_m": shapely.line_locate_point(tgeom[t], shapely.points(xy)),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "contour_id": contours["contour_id"].to_numpy()[ci[part]],
        }
    )
    # A crossing at a shared contour vertex can be reported twice.
    df = df.assign(d_round=df["dist_m"].round(3)).drop_duplicates(["t", "d_round"])
    df["base_offset_m"] = df["dist_m"] - transects["base_dist"].to_numpy()[df["t"]]
    df["n_crossings"] = df.groupby("t")["t"].transform("size")
    df = df.reindex(df["base_offset_m"].abs().sort_values(kind="stable").index)
    df = df.drop_duplicates("t").sort_values("t")
    return df.drop(columns="d_round")


def sample_dem(dem_path, xs, ys):
    """DEM values at points, reading one block at a time; NaN off the data."""
    out = np.full(xs.size, np.nan)
    with rasterio.open(dem_path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        bh, bw = src.block_shapes[0]
        idx = np.flatnonzero(inside)
        key = (rows[idx] // bh) * (src.width // bw + 1) + cols[idx] // bw
        order = np.argsort(key, kind="stable")
        idx, key = idx[order], key[order]
        for group in np.split(idx, np.flatnonzero(np.diff(key)) + 1):
            if group.size == 0:
                continue
            r0 = (rows[group[0]] // bh) * bh
            c0 = (cols[group[0]] // bw) * bw
            win = Window(c0, r0, min(bw, src.width - c0), min(bh, src.height - r0))
            block = src.read(1, window=win)
            out[group] = block[rows[group] - r0, cols[group] - c0]
    out[out == NODATA] = np.nan
    return out


def beach_slopes(dem_path, crossings, transects, res):
    """Least-squares slope (dz/d dist) of the DEM within +/- SLOPE_WINDOW_M of each crossing."""
    k = max(1, int(np.floor(SLOPE_WINDOW_M / res)))
    offsets = np.arange(-k, k + 1) * res
    tgeom = transects.geometry.values[crossings["t"].to_numpy()]
    lengths = shapely.length(tgeom)
    d = crossings["dist_m"].to_numpy()[:, None] + offsets[None, :]
    inside = (d >= 0) & (d <= lengths[:, None])
    pts = shapely.line_interpolate_point(
        np.repeat(tgeom, offsets.size), np.clip(d, 0, lengths[:, None]).ravel()
    )
    xy = shapely.get_coordinates(pts)
    z = sample_dem(dem_path, xy[:, 0], xy[:, 1]).reshape(d.shape)
    z[~inside] = np.nan

    ok = np.isfinite(z)
    r = np.where(ok, offsets[None, :], 0.0)
    zz = np.where(ok, z, 0.0)
    n = ok.sum(axis=1)
    sx, sy = r.sum(axis=1), zz.sum(axis=1)
    sxx, sxy = (r * r).sum(axis=1), (r * zz).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    slope[n < 3] = np.nan
    return slope


def process_scan(job):
    """Mosaic, contour and intersect one scan. Runs in a worker process."""
    region, scan = job["region"], job["scan"]
    tag = f"[{region}/{scan}]"
    work = TMP_ROOT / region / scan
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        cut_path = work / "cutline.gpkg"
        gpd.GeoDataFrame(geometry=[job["footprint"]], crs=OUTPUT_CRS).to_file(
            cut_path, layer="cutline", driver="GPKG"
        )
        dem_path = work / f"{scan}_dem.tif"
        res = job["res"]
        mosaic_scan(job["tiles"], cut_path, job["footprint"].bounds, res, dem_path)
        valid = to_datum_residual(
            dem_path, job["zscale"], datum_interpolator(job["datum_surface"])
        )
        if valid == 0:
            print(f"{tag} no valid DEM pixels in the shorezone")
            return {"scan": scan, "contours": None, "crossings": None}

        contours = contour_zero(dem_path, work / "contour.gpkg")
        n_raw = len(contours)
        contours = contours[contours.geometry.length >= MIN_CONTOUR_LEN_M]
        contours = contours.reset_index(drop=True)
        contours = contours.assign(
            scan=scan, year=scan_year(scan), length_m=contours.geometry.length
        )

        transects = job["transects"]
        crossings = (
            contour_crossings(contours, transects) if len(contours) else pd.DataFrame()
        )
        if len(crossings):
            crossings["beach_slope"] = beach_slopes(dem_path, crossings, transects, res)
            ids = transects[KEYS].iloc[crossings["t"].to_numpy()].reset_index(drop=True)
            crossings = pd.concat(
                [ids, crossings.drop(columns="t").reset_index(drop=True)], axis=1
            )
            crossings.insert(2, "scan", scan)

        print(
            f"{tag} {res:.2f} m, {valid} px, {len(contours)}/{n_raw} contours "
            f">= {MIN_CONTOUR_LEN_M:g} m, {len(crossings)}/{len(transects)} transects "
            f"crossed ({time.time() - started:.0f} s)"
        )
        return {"scan": scan, "contours": contours, "crossings": crossings}
    finally:
        if not KEEP_DEMS:
            shutil.rmtree(work, ignore_errors=True)


#################### Per-region driver ######################


def scan_jobs(region, region_geom, tiles, transects, datum_surface, flagged):
    hits = tiles.iloc[
        np.unique(tiles.sindex.query(region_geom, predicate="intersects"))
    ]
    jobs = []
    for scan, group in hits.groupby("scan", sort=True):
        if scan_year(scan) is None:
            print(f"  {scan}: no year in name, skipped")
            continue
        if scan in flagged:
            print(f"  {scan}: Flag Drop, skipped")
            continue
        footprint = shapely.make_valid(
            region_geom.intersection(group.geometry.union_all())
        )
        footprint = shapely.union_all(
            [
                g
                for g in shapely.get_parts(footprint)
                if g.geom_type in ("Polygon", "MultiPolygon")
            ]
        )
        if footprint.is_empty:
            continue
        zscale = z_scale(group["native_crs"].iloc[0])
        res = float(group["pixel_size_x"].median()) * zscale
        sub = transects.iloc[
            np.unique(transects.sindex.query(footprint, predicate="intersects"))
        ]
        jobs.append(
            {
                "region": region,
                "scan": scan,
                "tiles": group["path"].tolist(),
                "footprint": footprint,
                "res": res,
                "zscale": zscale,
                "datum_surface": str(datum_surface),
                "transects": sub.reset_index(drop=True),
            }
        )
    return jobs


def add_uncertainty(crossings, uncertainty, fallback):
    """Year, vertical accuracy and horizontal error bars, as in TransectShorelineChange.py."""
    crossings = crossings.copy()
    crossings["year"] = crossings["scan"].map(scan_year) + 0.5
    crossings = crossings.join(uncertainty.drop(columns="flag_drop"), on="scan")
    missing = crossings[["bias_z_m", "rmse_z_m"]].isna()
    crossings["uncertainty_fallback"] = missing.any(axis=1)
    crossings[["bias_z_m", "rmse_z_m"]] = crossings[["bias_z_m", "rmse_z_m"]].fillna(
        fallback
    )
    for scan in sorted(
        crossings.loc[crossings["uncertainty_fallback"], "scan"].unique()
    ):
        print(f"  {scan}: no reported accuracy, using fallback")
    # RMSE already includes the bias (RMSE^2 = bias^2 + SD^2), so it is the 1-sigma.
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings["sigma_x_m"] = crossings["rmse_z_m"] / crossings["beach_slope"].abs()
        shift = crossings["dist_m"] + crossings["bias_z_m"] / crossings["beach_slope"]
    crossings.loc[~np.isfinite(crossings["sigma_x_m"]), "sigma_x_m"] = np.nan
    # Bias shifts the interval, not the point or the fit; assumes bias = measured - reference z.
    crossings["bar_lower_m"] = shift - 1.96 * crossings["sigma_x_m"]
    crossings["bar_upper_m"] = shift + 1.96 * crossings["sigma_x_m"]
    return crossings


def process_region(region, uncertainty, fallback, tiles, pool):
    print(f"\n=== {region} ===")
    region_geom = load_region_geometry(region)
    bases, transects = load_transects(region)
    print(
        f"{len(transects)} transects, {bases['datum_z'].notna().sum()} bases with a "
        f"{CROSSING_DATUM} level; shorezone {region_geom.area / 1e6:.1f} km2"
    )

    paths = shoreline_paths(region)
    build_datum_surface(bases, region_geom.bounds, paths["datum_surface"])

    flagged = set(uncertainty.index[uncertainty["flag_drop"]])
    jobs = scan_jobs(
        region, region_geom, tiles, transects, paths["datum_surface"], flagged
    )
    print(f"  {len(jobs)} scans intersect the shorezone")
    if not jobs:
        return

    paths["positions_polylines"].unlink(missing_ok=True)
    contours_all = []
    frames = []
    futures = {pool.submit(process_scan, job): job["scan"] for job in jobs}
    for future in as_completed(futures):
        scan = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            print(f"  [{region}/{scan}] FAILED: {exc!r}")
            continue
        contours = result["contours"]
        if contours is not None and len(contours):
            contours_all.append(contours)
        if result["crossings"] is not None and len(result["crossings"]):
            frames.append(result["crossings"])
    if not frames:
        print("  no crossings, skipping")
        return

    crossings = add_uncertainty(
        pd.concat(frames, ignore_index=True), uncertainty, fallback
    )
    crossings = crossings.sort_values(KEYS + ["scan"], ignore_index=True)
    rates = change_rates(crossings)
    rates["datum"] = CROSSING_DATUM

    crossings.to_parquet(paths["crossings"], index=False)
    rates.to_parquet(paths["rates"], index=False)
    print(f"Wrote {paths['crossings']} and {paths['rates']}")

    # Write contour polylines to GPKG (each scan as a separate layer)
    # This was originally done during worker processing but SQLite locking caused issues
    # when multiple workers tried to write simultaneously, so we batch write here instead
    if contours_all:
        paths["positions_polylines"].unlink(missing_ok=True)
        first_write = True
        for contours in contours_all:
            if len(contours):
                scan = contours["scan"].iloc[0]
                pyogrio.write_dataframe(
                    gpd.GeoDataFrame(contours, geometry="geometry", crs=OUTPUT_CRS),
                    paths["positions_polylines"],
                    layer=scan,
                    driver="GPKG",
                    mode="a" if not first_write else "w",
                    dataset_options={"OVERWRITE": "YES"} if first_write else {},
                )
                first_write = False
        print(f"Wrote {len(contours_all)} scan layers to {paths['positions_polylines']}")

    paths["positions"].unlink(missing_ok=True)
    first_write = True
    for scan, c in crossings.groupby("scan"):
        points = gpd.GeoDataFrame(
            c.drop(columns=["x", "y"]),
            geometry=gpd.points_from_xy(c["x"], c["y"]),
            crs=OUTPUT_CRS,
        )
        pyogrio.write_dataframe(
            points,
            paths["positions"],
            layer=scan,
            driver="GPKG",
            mode="a" if not first_write else "w",
            dataset_options={"OVERWRITE": "YES"} if first_write else {},
        )
        first_write = False
    print(f"Wrote {crossings['scan'].nunique()} scan layers to {paths['positions']}")

    # Write all contours to a single shapefile
    if contours_all:
        all_contours_gdf = gpd.GeoDataFrame(
            pd.concat(contours_all, ignore_index=True),
            geometry="geometry",
            crs=OUTPUT_CRS,
        )
        paths["positions_polylines_shp"].unlink(missing_ok=True)
        # Also delete associated shapefile files
        for ext in [".shx", ".dbf", ".prj", ".cpg"]:
            (paths["positions_polylines_shp"].parent / (paths["positions_polylines_shp"].stem + ext)).unlink(missing_ok=True)
        pyogrio.write_dataframe(
            all_contours_gdf,
            str(paths["positions_polylines_shp"]),
            driver="ESRI Shapefile",
            dataset_options={"OVERWRITE": "YES"},
        )
        print(f"Wrote {len(all_contours_gdf)} contour polylines to {paths['positions_polylines_shp']}")

    out = bases.drop(columns="datum_z").merge(rates, on=KEYS, how="inner")
    out.insert(0, "region", region)
    out_path = OUT_ROOT / region / OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer = f"shoreline_change_{CROSSING_DATUM}"
    pyogrio.write_dataframe(
        out, out_path, layer=layer, driver="GPKG", dataset_options={"OVERWRITE": "YES"}
    )
    counts = out["change"].value_counts().to_dict()
    print(f"Wrote {len(out)} transects ({counts}) to {out_path} [{layer}]")

    write_change_polylines(out, OUT_ROOT / region)


def load_tiles():
    tiles = pyogrio.read_dataframe(
        INDEX_GPKG,
        layer=TILE_LAYER,
        columns=["scan", "path", "native_crs", "pixel_size_x"],
    )
    return tiles.to_crs(OUTPUT_CRS)


#################### Main ######################


if __name__ == "__main__":
    uncertainty, fallback = load_uncertainty()
    tiles = load_tiles()
    print(f"{len(tiles)} tiles in {INDEX_GPKG.name}")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for region in REGIONS:
            try:
                process_region(region, uncertainty, fallback, tiles, pool)
            except Exception as exc:
                print(f"=== {region} FAILED: {exc!r}")
