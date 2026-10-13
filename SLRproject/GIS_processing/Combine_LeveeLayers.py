# -*- coding: utf-8 -*-
"""
Combine levee/dike datasets into a single GeoPackage layer using GeoPandas.

Author: Kai (ECY)
"""

from pathlib import Path
import warnings
import geopandas as gpd
import fiona
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---- INPUT PATHS (edit if needed) ----
gpkg_path = Path(r"N:\SEA\Kai\Data\Levees\SEA_4930722724246232045.gpkg")

shp1_path = Path(r"N:\SEA\CDon\SLR_layers\CramerTidalRestrict\Tidal_Restrictions_PugetSound.shp")
shp2_path = Path(r"N:\SEA\CDon\SLR_layers\CramerTidalRestrict\Cramer_Tidal_Restrict_connectivity.shp")
shp3_path = Path(r"N:\SEA\CDon\SLR_layers\CramerTidalRestrict\Cramer_Tidal_Restrictions.shp")

# ---- OUTPUT ----
out_gpkg = Path(r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\temp\Levees_combined.gpkg")
out_layer_name = "levees_combined"

# ---- OPTIONS ----
# If your agency prefers a specific CRS (e.g., EPSG:2285 Washington North ftUS or EPSG:2927 WA South),
# set it here; otherwise we'll adopt the CRS of the GPKG layer we load.
target_epsg = None  # e.g., 2285 or 2927 or 4326. Leave as None to use the GPKG layer CRS.

# If your GeoPackage contains multiple layers, set the one you want here.
# Leave as None to auto-pick the first layer and print available options.
gpkg_layer_to_use = None  # e.g., "levees" or whatever the layer is called

# ---- FUNCTIONS ----

def list_gpkg_layers(gpkg_file: Path):
    """Return a list of layer names in a GeoPackage."""
    try:
        return fiona.listlayers(str(gpkg_file))
    except Exception as e:
        raise RuntimeError(f"Could not list layers in {gpkg_file}: {e}")

def read_gpkg_layer(gpkg_file: Path, layer_name: str | None = None) -> gpd.GeoDataFrame:
    """Read a GeoPackage layer; if layer_name is None, read the first layer."""
    layers = list_gpkg_layers(gpkg_file)
    if not layers:
        raise RuntimeError(f"No layers found in {gpkg_file}")

    if layer_name is None:
        print(f"[Info] Available layers in {gpkg_file.name}: {layers}")
        layer_name = layers[0]
        print(f"[Info] Using first layer: {layer_name}")

    if layer_name not in layers:
        raise ValueError(f"Layer '{layer_name}' not found in {gpkg_file}. Available: {layers}")

    gdf = gpd.read_file(gpkg_file, layer=layer_name)
    gdf["source"] = f"{gpkg_file.name}:{layer_name}"
    return gdf

def safe_filter(gdf: gpd.GeoDataFrame, primary_field: str, tidal_field: str,
                primary_value: str = "Dike/Levee", tidal_value: str = "Complete") -> gpd.GeoDataFrame:
    """
    Robust attribute filter:
    - Ignores case and leading/trailing whitespace.
    - Handles missing fields gracefully.
    """
    cols = {c.lower(): c for c in gdf.columns}
    pf = cols.get(primary_field.lower())
    tf = cols.get(tidal_field.lower())

    if pf is None or tf is None:
        print(f"[Warn] Missing expected fields in {gdf.attrs.get('source', 'unknown source')}."
              f" Expected '{primary_field}' and '{tidal_field}'. Returning empty selection.")
        return gdf.iloc[0:0].copy()

    # Normalize to string for comparison
    gdf["_pf_norm"] = gdf[pf].astype(str).str.strip().str.lower()
    gdf["_tf_norm"] = gdf[tf].astype(str).str.strip().str.lower()

    sel = (gdf["_pf_norm"] == primary_value.strip().lower()) & \
          (gdf["_tf_norm"] == tidal_value.strip().lower())

    out = gdf.loc[sel].copy()
    # Cleanup temporary columns
    out.drop(columns=["_pf_norm", "_tf_norm"], inplace=True)
    return out

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Trim whitespace in column names.
    - Rename columns to lowercase (safer for OGR drivers).
    - Merge case-insensitive duplicates (prefer non-null values).
    """
    df = df.copy()
    # 1) Trim whitespace
    df.columns = [c.strip() for c in df.columns]

    # 2) Lowercase + collapse duplicates
    seen = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in seen:
            # Merge values: keep existing where not null, otherwise take from duplicate
            df[seen[lc]] = df[seen[lc]].combine_first(df[c])
            df.drop(columns=c, inplace=True)
        else:
            # rename to lowercase to avoid case collisions downstream
            if c != lc:
                df.rename(columns={c: lc}, inplace=True)
            seen[lc] = lc

    return df

def standardize_schema(gdf: gpd.GeoDataFrame, primary_field: str | None, tidal_field: str | None) -> gpd.GeoDataFrame:
    """
    Create a simple, consistent schema:
      - 'primary' for the levee/dike classification (from provided field)
      - 'tidal_conn' for tidal connectivity status (from provided field)
      - 'source' already set by the reader/filter
    Keep all original fields too, but ensure these key fields exist.
    """
    gdf = gdf.copy()
    if primary_field and primary_field in gdf.columns:
        gdf["primary"] = gdf[primary_field]
    elif "primary" not in gdf.columns:
        gdf["primary"] = None

    if tidal_field and tidal_field in gdf.columns:
        gdf["tidal_conn"] = gdf[tidal_field]
    elif "tidal_conn" not in gdf.columns:
        gdf["tidal_conn"] = None

    # Ensure 'source' exists
    if "source" not in gdf.columns:
        gdf["source"] = "unknown"

    return gdf

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int | None, template_crs) -> gpd.GeoDataFrame:
    """Reproject to target EPSG if provided; else to the template CRS."""
    if gdf.crs is None:
        print("[Warn] Input layer has no CRS. Assuming template CRS.")
        gdf.set_crs(template_crs, inplace=True)

    target_crs = f"EPSG:{epsg}" if epsg else template_crs
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf

# ---- PIPELINE ----

# 1) Read the GeoPackage layer
gpkg_gdf = read_gpkg_layer(gpkg_path, gpkg_layer_to_use)

# Adopt CRS from GPKG unless user set target_epsg
template_crs = gpkg_gdf.crs
if target_epsg is not None:
    template_crs = f"EPSG:{target_epsg}"
    gpkg_gdf = gpkg_gdf.to_crs(template_crs)

# Standardize schema (we don't know its field names; keep as-is but ensure key fields exist)
gpkg_gdf["source"] = gpkg_gdf.get("source", f"{gpkg_path.name}:{gpkg_layer_to_use or 'first_layer'}")
gpkg_gdf = standardize_schema(gpkg_gdf, primary_field=None, tidal_field=None)

# 2) Read & filter Shapefile 1 (Primary_1 / Tidal_conn)
shp1 = gpd.read_file(shp1_path)
shp1["source"] = f"{shp1_path.name}"
shp1_f = safe_filter(shp1, primary_field="Primary__1", tidal_field="Tidal_conn")
shp1_f = standardize_schema(shp1_f, primary_field="Primary__1", tidal_field="Tidal_conn")
shp1_f = ensure_crs(shp1_f, target_epsg, template_crs)

# 3) Read & filter Shapefile 2 (Primary_Fe / Tidal_conn)
shp2 = gpd.read_file(shp2_path)
shp2["source"] = f"{shp2_path.name}"
shp2_f = safe_filter(shp2, primary_field="Primary_Fe", tidal_field="Tidal_conn")
shp2_f = standardize_schema(shp2_f, primary_field="Primary_Fe", tidal_field="Tidal_conn")
shp2_f = ensure_crs(shp2_f, target_epsg, template_crs)

# 4) Read & filter Shapefile 3 (Primary_Fe / Tidal_conn)
shp3 = gpd.read_file(shp3_path)
shp3["source"] = f"{shp3_path.name}"
shp3_f = safe_filter(shp3, primary_field="Primary_Fe", tidal_field="Tidal_conn")
shp3_f = standardize_schema(shp3_f, primary_field="Primary_Fe", tidal_field="Tidal_conn")
shp3_f = ensure_crs(shp3_f, target_epsg, template_crs)

# 5) Combine
combined = gpd.GeoDataFrame(
    pd.concat([gpkg_gdf, shp1_f, shp2_f, shp3_f], ignore_index=True),
    crs=template_crs
)

# Optional: drop exact duplicates by geometry & key fields
# (Comment out if you prefer to keep all)
# Note: equals() can be slow; for large datasets consider hashing WKB or buffering tolerance
try:
    combined["__wkb__"] = combined.geometry.apply(lambda geom: geom.wkb if geom is not None else None)
    combined = combined.drop_duplicates(subset=["__wkb__", "primary", "tidal_conn", "source"])
    combined = combined.drop(columns="__wkb__")
except Exception:
    pass

# 6) Export to GeoPackage
# If the output GPKG exists and the layer name already exists, GeoPandas will overwrite by default.
out_gpkg.parent.mkdir(parents=True, exist_ok=True)


# Apply just before writing
combined = sanitize_columns(combined)

combined.to_file(out_gpkg, layer=out_layer_name, driver="GPKG")
print(f"[Done] Wrote {len(combined)} features to {out_gpkg} (layer='{out_layer_name}').")


