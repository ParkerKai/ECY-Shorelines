"""Most-recent-lidar DEM mosaic and hillshade, per county.

For each county the scans in the tile index that intersect the barrier_region
(shorezone) polygon are mosaicked into one continuous DEM. At every pixel the
most recent scan with valid data wins; older scans only fill what newer ones do
not cover. Scans flagged "Flag Drop" in the uncertainty Excel file are skipped,
as in TransectShorelineChangeContour.py.

How it runs:
  1. Each scan's tiles are wrapped in a VRT and then a warped VRT (bilinear,
     EPSG:26910) on one common grid at the finest native resolution among the
     scans used. Nothing is resampled until a block is read.
  2. The output grid is split into CHUNK x CHUNK blocks; only blocks touching the
     shorezone are processed, in parallel (N_WORKERS processes). Each block
     rasterizes the shorezone mask, then reads scans newest first, filling only
     still-empty shorezone pixels and stopping once the block is full.
  3. Each block is written once to the output DEM (z converted to metres).
  4. The scan supplying each pixel is subsampled to ~COVERAGE_RES_M and
     polygonized into the coverage shapefile.
  5. gdal.DEMProcessing makes the hillshade.

Outputs, in OUT_ROOT:
  <region>_dem.tif            elevation (m, native vertical datum), nodata -9999
  <region>_scan_coverage.shp  polygons (scan, year, area_km2) outlining the area
                              of the DEM supplied by each scan
  <region>_hillshade.tif      gdal.DEMProcessing hillshade (uint8, nodata 0)

Run from a terminal (worker processes must import this module):
  python MostRecentDEM.py
"""

import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
from osgeo import gdal, ogr, osr
from rasterio.features import rasterize
from rasterio.windows import Window

from TransectShorelineChange import REGIONS, load_uncertainty, scan_year
from TransectShorelineChangeContour import (
    NODATA,
    OUTPUT_CRS,
    VALID_Z_RANGE,
    load_region_geometry,
    load_tiles,
    z_scale,
)

gdal.UseExceptions()

OUT_ROOT = Path(r"D:\Data\Lidar_MostRecentDEM")
TMP_ROOT = Path(r"D:\Data\Lidar\_mostrecent_tmp")

N_WORKERS = 12

# Output block edge (pixels); a multiple of the GeoTIFF tile size so each tile
# is written exactly once.
CHUNK = 2048
TILE = 512

# Approximate cell size (m) of the scan-coverage polygons.
COVERAGE_RES_M = 5.0

GDAL_CACHE_MB = 1024
CREATION = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]


def select_scans(region_geom, tiles, flagged):
    """Scans with tiles intersecting the region, newest first."""
    hits = tiles.iloc[
        np.unique(tiles.sindex.query(region_geom, predicate="intersects"))
    ]
    rb = region_geom.bounds
    scans = []
    for scan, group in hits.groupby("scan", sort=True):
        year = scan_year(scan)
        if year is None:
            print(f"  {scan}: no year in name, skipped")
            continue
        if scan in flagged:
            print(f"  {scan}: Flag Drop, skipped")
            continue
        tb = group.total_bounds
        zscale = z_scale(group["native_crs"].iloc[0])
        scans.append(
            {
                "scan": scan,
                "year": year,
                "tiles": group["path"].tolist(),
                "bounds": (
                    max(tb[0], rb[0]),
                    max(tb[1], rb[1]),
                    min(tb[2], rb[2]),
                    min(tb[3], rb[3]),
                ),
                "zscale": zscale,
                "res": float(group["pixel_size_x"].median()) * zscale,
            }
        )
    scans.sort(key=lambda s: (s["year"], s["scan"]), reverse=True)
    return scans


def snap_bounds(bounds, res):
    """Expand bounds outward to multiples of res so every scan shares one pixel grid."""
    return (
        np.floor(bounds[0] / res) * res,
        np.floor(bounds[1] / res) * res,
        np.ceil(bounds[2] / res) * res,
        np.ceil(bounds[3] / res) * res,
    )


def build_scan_vrt(scan, res, out_bounds, work):
    """Warped VRT of one scan on the output grid; returns its pixel offset in it."""
    tiles_vrt = work / f"{scan['scan']}_tiles.vrt"
    gdal.BuildVRT(str(tiles_vrt), [str(p) for p in scan["tiles"]])
    bounds = snap_bounds(scan["bounds"], res)
    warped = work / f"{scan['scan']}.vrt"
    gdal.Warp(
        str(warped),
        str(tiles_vrt),
        options=gdal.WarpOptions(
            format="VRT",
            dstSRS=OUTPUT_CRS,
            xRes=res,
            yRes=res,
            outputBounds=bounds,
            resampleAlg="bilinear",
            dstNodata=NODATA,
            outputType=gdal.GDT_Float32,
            warpOptions=["INIT_DEST=NO_DATA"],
        ),
    )
    ds = gdal.Open(str(warped))
    info = {
        "vrt": str(warped),
        "col_off": int(round((bounds[0] - out_bounds[0]) / res)),
        "row_off": int(round((out_bounds[3] - bounds[3]) / res)),
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "zscale": scan["zscale"],
    }
    ds = None
    return info


#################### Block worker ######################

_W = {}


def _init_worker(region_wkb, scans, out_bounds, res, k):
    gdal.SetCacheMax(GDAL_CACHE_MB * 1024 * 1024)
    _W.update(
        region=shapely.from_wkb(region_wkb),
        scans=scans,
        datasets=[gdal.Open(s["vrt"]) for s in scans],
        left=out_bounds[0],
        top=out_bounds[3],
        res=res,
        k=k,
    )
    shapely.prepare(_W["region"])


def process_block(block):
    """Fill one output block newest scan first. Returns (row, col, dem, coarse ids)."""
    r0, c0, h, w = block
    res, k = _W["res"], _W["k"]
    left = _W["left"] + c0 * res
    top = _W["top"] - r0 * res
    piece = shapely.clip_by_rect(_W["region"], left, top - h * res, left + w * res, top)
    if piece.is_empty:
        return None
    need = rasterize(
        [piece],
        out_shape=(h, w),
        transform=rasterio.transform.from_origin(left, top, res, res),
        dtype="uint8",
    ).astype(bool)
    if not need.any():
        return None

    dem = np.full((h, w), NODATA, dtype="float32")
    ids = np.zeros((h, w), dtype="uint8")
    low, high = VALID_Z_RANGE
    for i, (s, ds) in enumerate(zip(_W["scans"], _W["datasets"]), start=1):
        # Overlap of the block with this scan's VRT, in block coordinates.
        br0 = max(0, s["row_off"] - r0)
        bc0 = max(0, s["col_off"] - c0)
        br1 = min(h, s["row_off"] + s["height"] - r0)
        bc1 = min(w, s["col_off"] + s["width"] - c0)
        if br0 >= br1 or bc0 >= bc1 or not need[br0:br1, bc0:bc1].any():
            continue
        z = ds.GetRasterBand(1).ReadAsArray(
            c0 + bc0 - s["col_off"], r0 + br0 - s["row_off"], bc1 - bc0, br1 - br0
        )
        zm = z.astype("float64") * s["zscale"]
        ok = (
            need[br0:br1, bc0:bc1]
            & (z != NODATA)
            & np.isfinite(zm)
            & (zm >= low)
            & (zm <= high)
        )
        if not ok.any():
            continue
        dem[br0:br1, bc0:bc1][ok] = zm[ok]
        ids[br0:br1, bc0:bc1][ok] = i
        need[br0:br1, bc0:bc1][ok] = False
        if not need.any():
            break

    if not ids.any():
        return None
    # Centre pixel of each coarse cell, on a grid shared by all blocks.
    fr = (k // 2 - r0) % k
    fc = (k // 2 - c0) % k
    return r0, c0, dem, ids[fr::k, fc::k], (r0 + fr) // k, (c0 + fc) // k


#################### Outputs ######################


def polygonize_coverage(coarse, scans, out_bounds, cres, work, out_shp):
    """Polygons outlining where each scan supplies the DEM (shapefile with scan, year)."""
    src = gdal.GetDriverByName("MEM").Create(
        "", coarse.shape[1], coarse.shape[0], 1, gdal.GDT_Byte
    )
    src.SetGeoTransform((out_bounds[0], cres, 0, out_bounds[3], 0, -cres))
    srs = osr.SpatialReference()
    srs.SetFromUserInput(OUTPUT_CRS)
    src.SetProjection(srs.ExportToWkt())
    band = src.GetRasterBand(1)
    band.WriteArray(coarse)
    band.SetNoDataValue(0)

    tmp_gpkg = work / "coverage.gpkg"
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(tmp_gpkg))
    layer = ds.CreateLayer("coverage", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("scan_id", ogr.OFTInteger))
    gdal.Polygonize(band, band.GetMaskBand(), layer, 0)
    ds = layer = band = src = None

    gdf = pyogrio.read_dataframe(tmp_gpkg, layer="coverage")
    ids = gdf["scan_id"].to_numpy()
    gdf["scan"] = [scans[i - 1]["scan"] for i in ids]
    gdf["year"] = [scans[i - 1]["year"] for i in ids]
    gdf = gdf.drop(columns="scan_id").dissolve(by=["scan", "year"], as_index=False)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf = gdf.sort_values("year", ignore_index=True)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        out_shp.with_suffix(ext).unlink(missing_ok=True)
    pyogrio.write_dataframe(gdf, out_shp, driver="ESRI Shapefile")
    return gdf


def build_hillshade(dem_path, hillshade_path):
    # DEM is in metres in a metric CRS, so the default scale of 1 is correct.
    gdal.SetCacheMax(GDAL_CACHE_MB * 4 * 1024 * 1024)
    gdal.DEMProcessing(
        str(hillshade_path),
        str(dem_path),
        "hillshade",
        options=gdal.DEMProcessingOptions(
            format="GTiff",
            computeEdges=True,
            zFactor=1.0,
            azimuth=315,
            altitude=45,
            creationOptions=CREATION + ["SPARSE_OK=TRUE"],
        ),
    )


#################### Per-region driver ######################


def region_blocks(region_geom, width, height, out_bounds, res):
    blocks = [
        (r, c, min(CHUNK, height - r), min(CHUNK, width - c))
        for r in range(0, height, CHUNK)
        for c in range(0, width, CHUNK)
    ]
    boxes = shapely.box(
        *np.array(
            [
                [
                    out_bounds[0] + c * res,
                    out_bounds[3] - (r + h) * res,
                    out_bounds[0] + (c + w) * res,
                    out_bounds[3] - r * res,
                ]
                for r, c, h, w in blocks
            ]
        ).T
    )
    shapely.prepare(region_geom)
    keep = shapely.intersects(region_geom, boxes)
    return [b for b, kp in zip(blocks, keep) if kp], len(blocks)


def process_region(region, tiles, flagged):
    print(f"\n=== {region} ===")
    started = time.time()
    region_geom = load_region_geometry(region)
    scans = select_scans(region_geom, tiles, flagged)
    print(f"  {len(scans)} scans intersect the shorezone")
    if not scans:
        return
    if len(scans) > 255:
        raise ValueError("more than 255 scans; widen the scan-id dtype")

    res = min(s["res"] for s in scans)
    out_bounds = snap_bounds(region_geom.bounds, res)
    width = int(round((out_bounds[2] - out_bounds[0]) / res))
    height = int(round((out_bounds[3] - out_bounds[1]) / res))
    k = max(1, int(round(COVERAGE_RES_M / res)))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dem_path = OUT_ROOT / f"{region}_dem.tif"
    coverage_path = OUT_ROOT / f"{region}_scan_coverage.shp"
    hillshade_path = OUT_ROOT / f"{region}_hillshade.tif"
    for p in (dem_path, hillshade_path):
        p.unlink(missing_ok=True)

    work = TMP_ROOT / region
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        vrts = [build_scan_vrt(s, res, out_bounds, work) for s in scans]
        blocks, n_all = region_blocks(region_geom, width, height, out_bounds, res)
        print(
            f"  grid {width}x{height} at {res:.3f} m; "
            f"{len(blocks)}/{n_all} blocks touch the shorezone"
        )

        coarse = np.zeros((-(-height // k), -(-width // k)), dtype="uint8")
        profile = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=NODATA,
            crs=OUTPUT_CRS,
            transform=rasterio.transform.from_origin(
                out_bounds[0], out_bounds[3], res, res
            ),
            tiled=True,
            blockxsize=TILE,
            blockysize=TILE,
            compress="deflate",
            predictor=3,
            BIGTIFF="YES",
            sparse_ok=True,
            num_threads="ALL_CPUS",
        )
        done = 0
        with (
            rasterio.open(dem_path, "w", **profile) as dst,
            ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_worker,
                initargs=(shapely.to_wkb(region_geom), vrts, out_bounds, res, k),
            ) as pool,
        ):
            todo = iter(blocks)
            running = set()
            while True:
                # Bounded queue so finished blocks do not pile up in memory.
                while len(running) < 2 * N_WORKERS:
                    b = next(todo, None)
                    if b is None:
                        break
                    running.add(pool.submit(process_block, b))
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    out = f.result()
                    done += 1
                    if out is not None:
                        r0, c0, dem, cids, cr, cc = out
                        dst.write(
                            dem, 1, window=Window(c0, r0, dem.shape[1], dem.shape[0])
                        )
                        coarse[cr : cr + cids.shape[0], cc : cc + cids.shape[1]] = cids
                    if done % 100 == 0 or done == len(blocks):
                        print(
                            f"  {done}/{len(blocks)} blocks "
                            f"({time.time() - started:.0f} s)"
                        )

        cov = polygonize_coverage(
            coarse, scans, out_bounds, k * res, work, coverage_path
        )
        for _, row in cov.iterrows():
            print(f"  {row['scan']} ({row['year']}): {row['area_km2']:.2f} km2")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    t = time.time()
    build_hillshade(dem_path, hillshade_path)
    print(f"  hillshade {time.time() - t:.0f} s")
    print(
        f"  wrote {dem_path.name}, {coverage_path.name}, {hillshade_path.name} "
        f"({time.time() - started:.0f} s)"
    )


#################### Main ######################


if __name__ == "__main__":
    uncertainty, _ = load_uncertainty()
    flagged = set(uncertainty.index[uncertainty["flag_drop"]])
    tiles = load_tiles()
    print(f"{len(tiles)} tiles")
    for region in REGIONS:
        try:
            process_region(region, tiles, flagged)
        except Exception as exc:
            print(f"=== {region} FAILED: {exc!r}")
