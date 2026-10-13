import geopandas as gpd
import rasterio 
import os


dir_in = r'C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\CCAP'


counties = gpd.read_file(r'O:\boundaries\Counties.lyr')
ax = counties.plot()

dataset = '2021_ccap_v2_impervious_wa_J1420160'
file_in = os.path.join(dir_in,dataset,'2021_ccap_v2_impervious_wa_J1420160_tileindex.shp')

tile_index = gpd.read_file(os.path.join(file_in)) 

tile_index.plot(ax = ax)



#src = rasterio.open("tests/data/RGB.byte.tif")
