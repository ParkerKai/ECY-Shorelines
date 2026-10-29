#!/usr/bin/env python3
"""
Combine the C-CAP 30 m land cover raster with the tiled high-res (1 m) land cover,
preferring the high-res data where both exist, and polygonize all classes whose
description contains "wetland" into one GeoPackage per county.

Processing is done per county and in pixel blocks to limit RAM. Both datasets are
resampled (nearest) onto the high-res grid, so the output keeps the finest resolution.
"""

# -------------------------
# CONFIG
# -------------------------
CCAP_TIF = r"N:\SEA\Kai\Data\CCAP\2021_CCAP_J1422365\2021_CCAP_J1422365tR0_C0.tif"
HIRES_DIR = r"N:\SEA\Kai\Data\CCAP\2021_PugetWA_HighResLandCoverv2_J1420157"
HIRES_TILEINDEX = "2021_PugetWA_HighResLandCoverv2_J1420157_tileindex.shp"
COUNTY_SHP = r"N:\SEA\Kai\Data\Gis_General\CountyBounds.shp"
COUNTY_FIELD = "COUNTY_NM"
COUNTIES = None  # e.g. ["King", "Pierce"]; None = every county that overlaps the data
OUTDIR = r"N:\SEA\Kai\Data\CCAP\Extracted_Polygons\wetlands"

KEYWORD = "wetland"
# Output layer name -> normalized class name (lowercase, spaces and slashes removed).
WETLAND_TYPES = {
    "Estuarine_Emergent": "estuarineemergentwetland",
    "Estuarine_ScrubShrub": "estuarinescrubshrubwetland",
    "Estuarine_Forested": "estuarineforestedwetland",
    "Palustrine_Emergent": "palustrineemergentwetland",
    "Palustrine_ScrubShrub": "palustrinescrubshrubwetland",
    "Palustrine_Forested": "palustrineforestedwetland",
}
OTHER_WETLAND = len(WETLAND_TYPES) + 1  # wetland class not in WETLAND_TYPES
BLOCK_PIXELS = 8000  # block edge in high-res pixels; lower if RAM is tight
# -------------------------

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box, shape
from shapely.ops import unary_union


def log(msg):
    print(f"[wetlands] {msg}", flush=True)


def wetland_lut(raster_path):
    """Lookup table: raster value -> wetland type code (1..n), OTHER_WETLAND, or 0."""
    vat = gpd.read_file(Path(raster_path).with_suffix(".vat.dbf"))
    codes = {key: i for i, key in enumerate(WETLAND_TYPES.values(), start=1)}
    lut = np.zeros(256, dtype=np.uint8)
    for value, name in zip(vat["VALUE"].astype(int), vat["CLASS_NAME"].fillna("")):
        norm = name.lower().replace("/", "").replace(" ", "")
        if norm in codes:
            lut[value] = codes[norm]
        elif KEYWORD in norm:
            lut[value] = OTHER_WETLAND
    return lut


def combine_block(bounds, transform, shape_hw, ccap, ccap_lut, tiles, hires_lut, res):
    """Wetland type-code array for one block; high-res wins where it has data (value > 1)."""
    left, bottom, right, top = bounds
    h, w = shape_hw

    ccap_arr = np.zeros((h, w), dtype=np.uint8)
    if box(*bounds).intersects(box(*ccap.bounds)):
        reproject(
            rasterio.band(ccap, 1),
            ccap_arr,
            dst_transform=transform,
            dst_crs=ccap.crs,
            resampling=Resampling.nearest,
        )

    hires_arr = np.zeros((h, w), dtype=np.uint8)
    paths = tiles[tiles.intersects(box(*bounds))]["path"].tolist()
    if paths:
        srcs = [rasterio.open(p) for p in paths]
        try:
            mosaic, _ = merge(
                srcs, bounds=bounds, res=res, nodata=0, resampling=Resampling.nearest
            )
        finally:
            for s in srcs:
                s.close()
        mh, mw = min(h, mosaic.shape[1]), min(w, mosaic.shape[2])
        hires_arr[:mh, :mw] = mosaic[0, :mh, :mw]

    return np.where(hires_arr > 1, hires_lut[hires_arr], ccap_lut[ccap_arr])


def main():
    out_dir = Path(OUTDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = gpd.read_file(Path(HIRES_DIR) / HIRES_TILEINDEX)
    tiles["path"] = tiles["filename"].apply(lambda n: str(Path(HIRES_DIR) / f"{n}.tif"))
    with rasterio.open(tiles["path"].iloc[0]) as s0:
        crs, res = s0.crs, s0.res
        hires_lut = wetland_lut(tiles["path"].iloc[0])
    xres, yres = res
    tiles = tiles.to_crs(crs)

    ccap = rasterio.open(CCAP_TIF)
    ccap_lut = wetland_lut(CCAP_TIF)
    log(
        f"Wetland values - C-CAP: {np.flatnonzero(ccap_lut).tolist()}, "
        f"high-res: {np.flatnonzero(hires_lut).tolist()}"
    )
    missing = set(range(1, len(WETLAND_TYPES) + 1)) - set(ccap_lut) | (
        set(range(1, len(WETLAND_TYPES) + 1)) - set(hires_lut)
    )
    if missing:
        log(f"WARNING: type codes missing from a VAT: {sorted(missing)}")

    # Grid origin snapped to the high-res tile extent so blocks share one pixel grid.
    gx0, _, _, gy0 = tiles.total_bounds

    data_extent = box(*ccap.bounds).union(box(*tiles.total_bounds))
    counties = gpd.read_file(COUNTY_SHP).to_crs(crs)
    counties = counties[counties.intersects(data_extent)]
    if COUNTIES:
        counties = counties[counties[COUNTY_FIELD].isin(COUNTIES)]

    for _, row in counties.iterrows():
        name = str(row[COUNTY_FIELD]).replace(" ", "_")
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        # Align county window to the high-res grid.
        x0 = gx0 + np.floor((minx - gx0) / xres) * xres
        y0 = gy0 - np.floor((gy0 - maxy) / yres) * yres
        ncols = int(np.ceil((maxx - x0) / xres))
        nrows = int(np.ceil((y0 - miny) / yres))
        log(f"{name}: {ncols} x {nrows} px")

        parts = {"wetlands": []}
        parts.update({layer: [] for layer in WETLAND_TYPES})
        layer_by_code = dict(enumerate(WETLAND_TYPES, start=1))
        for r0 in range(0, nrows, BLOCK_PIXELS):
            for c0 in range(0, ncols, BLOCK_PIXELS):
                h, w = min(BLOCK_PIXELS, nrows - r0), min(BLOCK_PIXELS, ncols - c0)
                bx0, by1 = x0 + c0 * xres, y0 - r0 * yres
                bounds = (bx0, by1 - h * yres, bx0 + w * xres, by1)
                bbox = box(*bounds)
                if not bbox.intersects(geom):
                    continue
                transform = from_origin(bx0, by1, xres, yres)

                inside = rasterize(
                    [(geom, 1)], out_shape=(h, w), transform=transform, dtype="uint8"
                ).astype(bool)
                if not inside.any():
                    continue

                codes = np.where(
                    inside,
                    combine_block(
                        bounds,
                        transform,
                        (h, w),
                        ccap,
                        ccap_lut,
                        tiles,
                        hires_lut,
                        (xres, yres),
                    ),
                    0,
                ).astype(np.uint8)
                wet = codes > 0
                if not wet.any():
                    continue

                parts["wetlands"].extend(
                    shape(g)
                    for g, v in shapes(
                        wet.astype(np.uint8), mask=wet, transform=transform
                    )
                )
                for code, layer in layer_by_code.items():
                    mask = codes == code
                    if mask.any():
                        parts[layer].extend(
                            shape(g)
                            for g, v in shapes(
                                mask.astype(np.uint8), mask=mask, transform=transform
                            )
                        )

        if not parts["wetlands"]:
            log(f"{name}: no wetland cells")
            continue

        out = out_dir / f"Wetlands_{name}.gpkg"
        out.unlink(missing_ok=True)  # avoid appending to layers from a previous run
        for layer, geoms in parts.items():
            if not geoms:
                continue
            merged = unary_union(geoms)  # dissolves polygons split at block edges
            gdf = gpd.GeoDataFrame(geometry=[merged], crs=crs).explode(
                index_parts=False
            )
            gdf = gdf.reset_index(drop=True)
            gdf["County"] = row[COUNTY_FIELD]
            gdf["Area_m2"] = gdf.area
            gdf.to_file(out, layer=layer, driver="GPKG")
            log(f"{name}: wrote {len(gdf)} polygons -> {out} [{layer}]")

    ccap.close()


if __name__ == "__main__":
    main()
