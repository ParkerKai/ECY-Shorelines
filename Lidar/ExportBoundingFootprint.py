"""Export the bounding footprint GeoPackage layer to an ESRI Shapefile."""

from pathlib import Path
import geopandas as gpd


Region_list = ['hood_canal', 'north_sound', 'south_sound', 'strait_of_juan_de_fuca', 'whidbey']


IN_DIR = Path(r"L:\PROJECTS\Shoreline_Mapping_Unit\Outside_Data\WDNR\aerial_lidar")
OUT_DIR = Path(r"N:\SEA\Kai\Data\lidar\DNR_Shapefiles")

SHAPEFILE_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"}


def remove_existing_shapefile(shapefile_path):
    """Remove existing sidecar files for the target shapefile stem."""
    for path in shapefile_path.parent.glob(f"{shapefile_path.stem}.*"):
        if path.suffix.lower() in SHAPEFILE_EXTENSIONS:
            path.unlink()

############### Main ##############

for Region in Region_list:
    
    print(f"Processing region: {Region}")
    
    IN_GPKG = IN_DIR / Region/ "footprints" /f"{Region}_footprints.gpkg"
    BOUNDING_LAYER_NAME = f"{Region}_bounding_footprint"

    OUT_SHP = OUT_DIR / f"{BOUNDING_LAYER_NAME}.shp"
        
    if not IN_GPKG.exists():
        raise SystemExit(f"Input GeoPackage does not exist: {IN_GPKG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bounding_gdf = gpd.read_file(IN_GPKG, layer=BOUNDING_LAYER_NAME)
    if bounding_gdf.empty:
        raise SystemExit(f"Layer is empty: {BOUNDING_LAYER_NAME}")

    if "raster_count" in bounding_gdf.columns:
        bounding_gdf = bounding_gdf.rename(columns={"raster_count": "rast_count"})

    remove_existing_shapefile(OUT_SHP)
    bounding_gdf.to_file(OUT_SHP, driver="ESRI Shapefile")
    print(f"Wrote {len(bounding_gdf)} feature(s) to {OUT_SHP}")

