# ---- PROJ/GDAL Fix (Windows + conda) ----
import os
CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ["PROJ_LIB"]  = fr"{CONDA}\Library\share\proj"
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]         # <- auch setzen
os.environ["GDAL_DATA"] = fr"{CONDA}\Library\share\gdal"

# wichtig unter Windows: GDAL/PROJ-DLLs finden
try:
    os.add_dll_directory(fr"{CONDA}\Library\bin")
except Exception:
    pass

# pyproj explizit auf proj.db zeigen
from pyproj import datadir as _pd
_pd.set_data_dir(os.environ["PROJ_LIB"])
# ---- Ende Fix ----

import rasterio, geopandas as gpd
from pathlib import Path

GEOTIFF_PATH = r"C:\git\Master-Thesis-GEOAI\data\ortho_4.tif"
GPKG_PATH    = r"C:\git\Master-Thesis-GEOAI\outputs\buildings_sam_tiles.gpkg"
LAYER_NAME   = "buildings_sam"

with rasterio.open(GEOTIFF_PATH) as src:
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)

gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
print("Vector CRS:", gdf.crs)
print("Vector bounds:", gdf.total_bounds)
