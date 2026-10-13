# %% [markdown]
# Global ML Building Footprints → county-selected workflow
# - Latest index host + legacy fallback
# - Multiple files per quadkey
# - Gzipped line-delimited GeoJSON handling
# - Tiles from county bounding box (L9), final clip to county polygon
# - Threads, progress bars, flexible predicate, GeoJSON/GPKG output

# %%
import os
import io
import json
import gzip
import requests
import pandas as pd
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union
import mercantile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- TLS + requests imports (place at the very top of the file) ---
import ssl

# Use system trust store for Python/requests
import truststore  # pip install truststore

truststore.inject_into_ssl()  # make stdlib ssl use OS trust store 【1-dead64】【2-4ba8b8】
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# =========================
# CONFIG – EDIT THESE
# =========================
COUNTY_SHP = r"N:\SEA\Kai\Data\BarriersToMigration\CountyBounds.shp"
county_name = "Wahkiakum"  # <-- set the county you want (case-insensitive)

# If your shapefile lacks a defined CRS, set this to the correct EPSG (e.g., 2285 or 2927 for WA local systems; otherwise None)
COUNTY_CRS_EPSG = None

# Target CRS for tiling and output (dataset uses EPSG:4326)
TARGET_CRS = "EPSG:4326"

# Early bounding-box filter: 'intersects' tends to include edge-touching buildings
BOUNDING_BOX_PREDICATE = "intersects"  # or "within"

# Final county clip predicate
FINAL_PREDICATE = "within"  # or "intersects"

# Output
OUTPUT_FN = os.path.join(
    r"N:\SEA\Kai\Data\Assets\Microsoft", f"{county_name}_building_footprints.gpkg"
)  # or .geojson
OUTPUT_DRIVER = "GPKG"  # "GeoJSON" or "GPKG"

# Threads for network I/O
MAX_WORKERS = 8

# Latest index (July 24, 2026) and fallback to legacy host
INDEX_URLS = [
    "https://bfppub.blob.core.windows.net/$web/2026-07-24/dataset-links.csv",  # latest index host
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",  # legacy fallback
]
# =========================


# %% [markdown]
# Helper functions: CRS, cleaning, safe parsing


# %%
def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int | None):
    """
    Ensure the GeoDataFrame has a CRS. If none, assign epsg (if provided).
    """
    if gdf.crs is None:
        if epsg is None:
            raise ValueError(
                "County shapefile has no CRS. Please set COUNTY_CRS_EPSG (e.g., 4326)."
            )
        gdf = gdf.set_crs(epsg, allow_override=True)
    return gdf


def finalize_clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Light cleaning: drop null geometries, fix invalids, keep polygons/multipolygons.
    """
    gdf = gdf[~gdf.geometry.isna()].copy()
    # Fix invalid geometries (buffer(0) trick); ignore non-polygons safely
    fixed = []
    for geom in gdf.geometry:
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            pass
        fixed.append(geom)
    gdf.geometry = fixed
    # Keep only polygonal geometries
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gdf


def load_index(urls: list[str]) -> pd.DataFrame:
    for u in urls:
        try:
            resp = SESSION.get(u, timeout=90)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), dtype=str)
            print(f"Loaded index: {u}")
            return df
        except Exception as e:
            print(f"Failed index {u}: {e}")
    raise RuntimeError("Could not load dataset-links.csv from any known location.")


def load_geojsonl(url: str) -> gpd.GeoDataFrame:
    resp = SESSION.get(url, stream=True, timeout=90)
    resp.raise_for_status()
    raw = resp.raw
    # streamed gzip if needed
    if url.endswith(".gz") or (resp.headers.get("Content-Encoding", "") == "gzip"):
        gz = gzip.GzipFile(fileobj=raw)
        bytestream = io.TextIOWrapper(gz, encoding="utf-8")
    else:
        bytestream = io.TextIOWrapper(raw, encoding="utf-8")

    features = []
    ld_failed = False
    try:
        for line in bytestream:
            line = line.strip()
            if not line:
                continue
            feat = json.loads(line)
            geom = geometry.shape(feat["geometry"])
            props = feat.get("properties", {})
            features.append({"geometry": geom, **props})
    except Exception:
        ld_failed = True

    if ld_failed and not features:
        text = SESSION.get(url, timeout=90).text
        fc = json.loads(text)
        if fc.get("type") == "FeatureCollection":
            for f in fc.get("features", []):
                geom = geometry.shape(f["geometry"])
                props = f.get("properties", {})
                features.append({"geometry": geom, **props})
        else:
            raise RuntimeError(f"GeoJSON parse failure for {url}")

    return gpd.GeoDataFrame(features, crs=TARGET_CRS)


def filter_gdf(gdf: gpd.GeoDataFrame, poly, predicate: str) -> gpd.GeoDataFrame:
    """
    Spatial filter by predicate: 'within' or 'intersects'.
    """
    if gdf.empty:
        return gdf
    if predicate == "within":
        return gdf[gdf.geometry.within(poly)]
    else:
        return gdf[gdf.geometry.intersects(poly)]


def concat_gdfs(parts: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if not parts:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)
    df = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=TARGET_CRS)


# ===============================================================================
# SSL tools
# ===============================================================================
# --- relaxed SSLContext (keep verification ON, relax strict checks) ---
def make_tls_relaxed_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()  # loads OS trust via truststore injection
    # OpenSSL 3+: relax strict extension/name-constraints checks
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        try:
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        except Exception:
            # If the platform doesn't support verify_flags or this flag, continue gracefully
            pass
    return ctx


class SSLContextAdapter(HTTPAdapter):
    """Requests adapter that passes a custom SSLContext into urllib3 PoolManager."""

    def __init__(self, ssl_context: ssl.SSLContext | None = None, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        if self._ssl_context is not None:
            pool_kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(
            connections, maxsize, block=block, **pool_kwargs
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        if self._ssl_context is not None:
            proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def make_session() -> requests.Session:
    s = requests.Session()

    # Retries for transient issues
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )

    # Mount HTTPS with our relaxed SSLContext
    relaxed_ctx = make_tls_relaxed_context()
    s.mount("https://", SSLContextAdapter(ssl_context=relaxed_ctx, max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))

    # Optional: inherit env proxies (common in ECY networks)
    # requests uses HTTPS_PROXY/HTTP_PROXY/NO_PROXY automatically if present. 【4-51204b】

    # Polite UA
    s.headers.update({"User-Agent": "msbf-county-client/1.0"})
    return s


# %% [markdown]
# 1) Read county bounds, pick county by name, build bbox and union polygon


# Global session
SESSION = make_session()

# %%
county_bnds = gpd.read_file(COUNTY_SHP)  # read
county_bnds = ensure_crs(county_bnds, COUNTY_CRS_EPSG)
county_bnds = finalize_clean(county_bnds)

# Find a likely name column
name_cols = [
    c
    for c in county_bnds.columns
    if c.lower() in {"name", "county", "county_name", "county_nm"}
]
if len(name_cols) == 0:
    raise ValueError(
        "County bounds must have a NAME/County column to filter by county."
    )

name_col = name_cols[0]
county_name_lower = county_name.lower().strip()
county_poly = county_bnds[county_bnds[name_col].str.lower() == county_name_lower]
if len(county_poly) == 0:
    raise ValueError(f"County '{county_name}' not found in county bounds.")

# Reproject county polygon to target CRS (EPSG:4326)
county_poly_target = county_poly.to_crs(TARGET_CRS)

# Union in case multipart polygons; then get bounding box
county_union = unary_union(county_poly_target.geometry)
county_bbox = county_union.envelope  # shapely box

# bbox → mercantile tiles at zoom 9 (dataset partitioning level)
minx, miny, maxx, maxy = county_bbox.bounds
quad_keys = {
    mercantile.quadkey(t) for t in mercantile.tiles(minx, miny, maxx, maxy, zooms=9)
}
quad_keys = list(quad_keys)
print(f"County bbox spans {len(quad_keys)} L9 tiles: {quad_keys}")

# %% [markdown]
# 2) Load dataset index (latest → legacy), download & filter per tile

# %%
index_df = load_index(INDEX_URLS)


def process_quadkey(qk: str) -> gpd.GeoDataFrame:
    rows = index_df[index_df["QuadKey"] == qk]
    if rows.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)

    parts = []
    for _, r in rows.iterrows():  # handle >1 file per quadkey
        url = r["Url"]
        gdf = load_geojsonl(url)

        # Early bbox filter to reduce size
        gdf = filter_gdf(gdf, county_bbox, BOUNDING_BOX_PREDICATE)

        # Final county-polygon clip
        gdf = filter_gdf(gdf, county_union, FINAL_PREDICATE)
        if not gdf.empty:
            parts.append(gdf)

    if parts:
        return concat_gdfs(parts)
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)


combined_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)
id_counter = 0

with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(quad_keys)))) as ex:
    futures = {ex.submit(process_quadkey, qk): qk for qk in quad_keys}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiles"):
        gdf_tile = fut.result()
        if not gdf_tile.empty:
            gdf_tile["id"] = range(id_counter, id_counter + len(gdf_tile))
            id_counter += len(gdf_tile)
            combined_gdf = concat_gdfs([combined_gdf, gdf_tile])

print(f"Total buildings in '{county_name}' county: {len(combined_gdf)}")

# %% [markdown]
# 3) Save results (GeoJSON or GeoPackage)

# %%
if combined_gdf.empty:
    print("No buildings found for selected county.")
else:
    combined_gdf = combined_gdf.to_crs(TARGET_CRS)
    combined_gdf.to_file(OUTPUT_FN, driver=OUTPUT_DRIVER)
    print(f"Wrote {len(combined_gdf)} features to {OUTPUT_FN}")
