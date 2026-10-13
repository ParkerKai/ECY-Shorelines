# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Interpolate meteo data onto tide gauge locations using multiple geographical domains
Author: Kai Parker, WA Dept. of Ecology (kaip461@ecy.wa.gov)
"""

import os
import json
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr

# -------------------------------
# User Config
# -------------------------------

dir_in = r'C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Meteo'   # root folder
domains = ['puget_sound', 'Cascadia']           # domain names to search in order of preference
dir_in_TG = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"

# Output options
dir_out = r'C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\Meteo\TG'
os.makedirs(dir_out, exist_ok=True)
save_per_station = False                          # if True: one NetCDF per station; else: one combined file

# Tide gauges
TG = [
    {"station": '9419750', "name": 'CrescentCity'},
    {"station": '9431647', "name": 'PortOrford'},
    {"station": '9432780', "name": 'Charleston'},
    {"station": '9435380', "name": 'SouthBeach'},
    {"station": '9437540', "name": 'Garibaldi'},
    {"station": '9439099', "name": 'Wauna'},
    {"station": '9440569', "name": 'Skamokawa'},
    {"station": '9440581', "name": 'CapeDisappointment'},
    {"station": '9439040', "name": 'Astoria'},
    {"station": '9440910', "name": 'TokePoint'},
    {"station": '9441102', "name": 'Westport'},
    {"station": '9442396', "name": 'LaPush'},
    {"station": '9443090', "name": 'NeahBay'},
    {"station": '9444090', "name": 'PortAngeles'},
    {"station": '9444900', "name": 'PortTownsend'},
    {"station": '9449880', "name": 'FridayHarbor'},
    {"station": '9449424', "name": 'CherryPoint'},
    {"station": '9447130', "name": 'Seattle'},
    {"station": '9446484', "name": 'Tacoma'},
]
TG = pd.DataFrame(TG).set_index("station")


# -------------------------------
# Helpers
# -------------------------------

def read_tg_metadata(dir_in_TG: str, TG_df: pd.DataFrame) -> pd.DataFrame:
    """Read lat/lon from NOAA TG metadata JSONs and add as columns."""
    lats, lons = [], []
    for stat in TG_df.index:
        files_meta = glob(os.path.join(dir_in_TG, f"NOAA_TG_{stat}", f"metadata_*_{stat}.json"))
        if not files_meta:
            raise FileNotFoundError(f"No metadata JSON found for station {stat}")
        with open(files_meta[0], "r", encoding="utf-8") as f:
            md = json.load(f)
        if not isinstance(md, dict):
            raise ValueError(f"Metadata for station {stat} is not a dict")
        lat = md.get("Latitude")
        lon = md.get("Longitude")
        if lat is None or lon is None:
            raise ValueError(f"Missing lat/lon in metadata for station {stat}")
        lats.append(float(lat))
        lons.append(float(lon))
    TG_out = TG_df.copy()
    TG_out["lat"] = lats
    TG_out["lon"] = lons
    return TG_out

def open_domain_ds(root: str, domain: str) -> xr.Dataset:
    """Open a domain dataset with reasonable chunks; raise if no files found."""
    files = os.path.join(root, domain, '*.nc')
    if not files:
        raise FileNotFoundError(
            f"No .nc files found for domain '{domain}''. "
        )
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True)
    return ds


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename common coordinate names to lat/lon; return updated dataset."""
    rename_map = {}
    if 'latitude' in ds.coords: rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_map['longitude'] = 'lon'
    # Some models store lat/lon as data variables (e.g., WRF):
    if 'XLAT' in ds: rename_map['XLAT'] = 'lat'
    if 'XLONG' in ds: rename_map['XLONG'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def lon_style(ds: xr.Dataset) -> str:
    """Return '360' if longitudes in [0,360], else 'negpos' for [-180,180]."""
    if 'lon' in ds.coords:
        lonv = ds['lon'].values
    elif 'lon' in ds:
        lonv = ds['lon'].values
    else:
        # default assume negpos if not present; caller should ensure presence
        return 'negpos'
    # use nanmin/nanmax for safety
    mn = np.nanmin(lonv)
    mx = np.nanmax(lonv)
    if mx > 180.0 and mn >= 0.0:
        return '360'
    return 'negpos'


def normalize_lon_for_ds(lon: float, ds: xr.Dataset) -> float:
    """Convert input lon in [-180,180] to dataset style, if needed."""
    style = lon_style(ds)
    if style == '360':
        # map [-180,180] to [0,360]
        return lon % 360.0
    else:
        # ensure within [-180,180]
        lon_mod = ((lon + 180.0) % 360.0) - 180.0
        return lon_mod


def domain_bbox(ds: xr.Dataset):
    """Return (lat_min, lat_max, lon_min, lon_max) using available lat/lon."""
    latv = None
    lonv = None
    if 'lat' in ds.coords:
        latv = ds['lat'].values
    elif 'lat' in ds:
        latv = ds['lat'].values
    if 'lon' in ds.coords:
        lonv = ds['lon'].values
    elif 'lon' in ds:
        lonv = ds['lon'].values
    if latv is None or lonv is None:
        raise ValueError("Dataset lacks 'lat'/'lon' coordinates or variables.")
    return (np.nanmin(latv), np.nanmax(latv), np.nanmin(lonv), np.nanmax(lonv))


def point_in_bbox(lat: float, lon: float, bbox, tol=0.1) -> bool:
    """Check if (lat, lon) lies within bbox with tolerance (degrees)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat_min - tol <= lat <= lat_max + tol) and (lon_min - tol <= lon <= lon_max + tol)


def haversine_factor(lat1, lon1, lat2, lon2):
    """
    Return 'a' from haversine formula ~ (distance / 2R)^2.
    Useful for nearest comparisons without computing arcsin.
    """
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2) - np.deg2rad(lon1)
    return np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2


def nearest_grid_distance(ds: xr.Dataset, tlat: float, tlon: float) -> float:
    """
    Compute minimal haversine 'a' to any grid point of the dataset (coarse measure of distance).
    For 1-D grids, check nearest coord values; for 2-D grids, flatten and search.
    """
    # Normalize TG lon to dataset style
    tlon_mod = normalize_lon_for_ds(tlon, ds)

    # 1-D grid
    if 'lat' in ds.coords and ds['lat'].ndim == 1 and 'lon' in ds.coords and ds['lon'].ndim == 1:
        # nearest lat
        ilat = np.nanargmin(np.abs(ds['lat'].values - tlat))
        # nearest lon (wrap-aware already via normalize_lon_for_ds)
        ilon = np.nanargmin(np.abs(ds['lon'].values - tlon_mod))
        lat_near = float(ds['lat'].values[ilat])
        lon_near = float(ds['lon'].values[ilon])
        return haversine_factor(tlat, tlon_mod, lat_near, lon_near)

    # 2-D grid
    lat2d = (ds['lat'].values if 'lat' in ds else None)
    lon2d = (ds['lon'].values if 'lon' in ds else None)
    if lat2d is None or lon2d is None or lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("Cannot compute nearest distance: lat/lon not found or not 2-D.")
    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()
    # compute per-point 'a' and take min
    phi1 = np.deg2rad(flat_lat)
    phi2 = np.deg2rad(tlat)
    dphi = phi1 - phi2
    dlambda = np.deg2rad(flat_lon) - np.deg2rad(tlon_mod)
    a = np.sin(dphi/2.0)**2 + np.cos(phi2)*np.cos(phi1)*np.sin(dlambda/2.0)**2
    return float(np.nanmin(a))


def interp_regular_latlon(ds: xr.Dataset, TG_df: pd.DataFrame, method='linear') -> xr.Dataset:
    """
    Vectorized interpolation for regular (1-D) lat/lon grids (e.g., ERA5).
    Returns a Dataset with a new 'station' dimension.
    """
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise ValueError("Dataset must have 'lat' and 'lon' coordinates for regular interpolation.")
    if ds['lat'].ndim != 1 or ds['lon'].ndim != 1:
        raise ValueError("Expected 1-D lat/lon coordinates. Got curvilinear (2-D) grid.")

    # Normalize longitudes to dataset style
    lon_mod = np.array([normalize_lon_for_ds(lon, ds) for lon in TG_df['lon'].values])

    station = xr.DataArray(TG_df.index, dims=('station',), name='station')
    lat_da = xr.DataArray(TG_df['lat'].values, dims=('station',), coords={'station': station})
    lon_da = xr.DataArray(lon_mod, dims=('station',), coords={'station': station})

    out = ds.interp(lat=lat_da, lon=lon_da, method=method)
    out = out.assign_coords(station=station)
    return out


def nearest_on_curvilinear(ds: xr.Dataset, TG_df: pd.DataFrame,
                           lat_name='lat', lon_name='lon',
                           y_name=None, x_name=None) -> xr.Dataset:
    """
    Nearest neighbor selection for 2-D curvilinear lat/lon grids (e.g., WRF/Conus404).
    Returns a Dataset with 'station' dimension.
    """
    # Guess grid dimension names if not provided
    if y_name is None:
        for cand in ['y', 'Y', 'south_north', 'lat_dim', 'j']:
            if cand in ds.dims:
                y_name = cand; break
    if x_name is None:
        for cand in ['x', 'X', 'west_east', 'lon_dim', 'i']:
            if cand in ds.dims:
                x_name = cand; break

    if y_name is None or x_name is None:
        raise ValueError("Could not infer grid dim names. Provide y_name/x_name explicitly.")
    if lat_name not in ds or lon_name not in ds:
        raise ValueError(f"Dataset must contain '{lat_name}' and '{lon_name}' variables.")

    lat2d = ds[lat_name].values
    lon2d = ds[lon_name].values
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("Expected 2-D lat/lon for curvilinear grid.")

    # Normalize TG longitudes to dataset style
    lon_mod = np.array([normalize_lon_for_ds(lon, ds) for lon in TG_df['lon'].values])

    ny, nx = lat2d.shape
    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()

    def nearest_idx(tlat, tlon):
        phi1 = np.deg2rad(flat_lat)
        phi2 = np.deg2rad(tlat)
        dphi = phi1 - phi2
        dlambda = np.deg2rad(flat_lon) - np.deg2rad(tlon)
        a = np.sin(dphi/2.0)**2 + np.cos(phi2)*np.cos(phi1)*np.sin(dlambda/2.0)**2
        idx = np.nanargmin(a)
        return np.unravel_index(idx, (ny, nx))

    jj, ii = [], []
    for tlat, tlon in zip(TG_df['lat'].values, lon_mod):
        j, i = nearest_idx(tlat, tlon)
        jj.append(j); ii.append(i)

    station = xr.DataArray(TG_df.index, dims=('station',), name='station')
    sel = ds.isel(**{
        y_name: xr.DataArray(jj, dims=('station',), coords={'station': station}),
        x_name: xr.DataArray(ii, dims=('station',), coords={'station': station})
    })
    sel = sel.assign_coords(station=station)
    return sel


# -------------------------------
# Main workflow
# -------------------------------

# 1) Tide gauge locations
TG = read_tg_metadata(dir_in_TG, TG)

# 2) Open & prep datasets per domain
ds_map = {}
bbox_map = {}
grid_type = {}  # 'regular' or 'curvilinear'
for dom in domains:
    ds = open_domain_ds(dir_in, dom)
    ds = standardize_coords(ds)
    # classify grid type
    if ('lat' in ds.coords and 'lon' in ds.coords and ds['lat'].ndim == 1 and ds['lon'].ndim == 1):
        grid_type[dom] = 'regular'
    else:
        # treat as curvilinear if lat/lon exist but not 1-D
        grid_type[dom] = 'curvilinear'
    ds_map[dom] = ds
    bbox_map[dom] = domain_bbox(ds)

# 3) Assign each station to a best domain
tol_deg = 0.15  # small tolerance to include near-boundary stations
station_domain = {}
for stat, row in TG.iterrows():
    tlat = float(row['lat'])
    tlon = float(row['lon'])

    # candidates whose bbox contains the point (with tol)
    candidates = [dom for dom in domains if point_in_bbox(tlat, normalize_lon_for_ds(tlon, ds_map[dom]), bbox_map[dom], tol=tol_deg)]

    if len(candidates) == 1:
        station_domain[stat] = candidates[0]
    elif len(candidates) > 1:
        # choose nearest grid among candidates
        best_dom = None
        best_a = np.inf
        for dom in candidates:
            a = nearest_grid_distance(ds_map[dom], tlat, tlon)
            if a < best_a:
                best_a = a; best_dom = dom
        station_domain[stat] = best_dom
    else:
        # outside all bboxes: pick nearest grid across all domains
        best_dom = None
        best_a = np.inf
        for dom in domains:
            a = nearest_grid_distance(ds_map[dom], tlat, tlon)
            if a < best_a:
                best_a = a; best_dom = dom
        station_domain[stat] = best_dom

# (Optional) print assignment summary
print("Station -> Domain assignment:")
for stat in TG.index:
    print(f"  {stat} ({TG.loc[stat,'name']}) -> {station_domain[stat]}")

# 4) Vectorized interpolation per domain
results = []
for dom in domains:
    # subset TG for this domain
    sel_stats = [s for s in TG.index if station_domain[s] == dom]
    if not sel_stats:
        continue
    TG_dom = TG.loc[sel_stats]

    ds = ds_map[dom]

    # OPTIONAL: subset variables to speed up
    # Example for ERA5:
    # if Data_type == 'ERA5':
    #     keep_vars = ['u10', 'v10', 'msl']  # adjust as needed
    #     existing_vars = [v for v in keep_vars if v in ds.data_vars]
    #     ds = ds[existing_vars] if existing_vars else ds

    if grid_type[dom] == 'regular':
        out_dom = interp_regular_latlon(ds, TG_dom, method='linear')
    else:
        out_dom = nearest_on_curvilinear(ds, TG_dom)

    # Add a coord for domain
    out_dom = out_dom.assign_coords(domain=dom)
    results.append(out_dom)

# 5) Combine across domains (preserve original station order)
if not results:
    raise RuntimeError("No interpolation results produced. Check domain inputs and TG metadata.")
combined = xr.concat(results, dim='station')
# Ensure station order matches TG index order
combined = combined.sel(station=TG.index)

# 6) Save outputs
if save_per_station:
    for stat in TG.index:
        fn = os.path.join(dir_out, f"{stat}_ERA5met_on_TG.nc")
        combined.sel(station=stat).to_netcdf(fn)
        print(f"Saved: {fn}")
else:
    fn = os.path.join(dir_out, "AllStations_ERA5met_on_TG.nc")
    combined.to_netcdf(fn)
    print(f"Saved combined file: {fn}")