import geopandas as gpd
import rasterio as rio
import numpy as np
from rasterio.windows import from_bounds, Window
from shapely.ops import transform
from pyproj import Transformer


def extract_patch_from_gdb(
    geometry,
    geom_crs,
    tif_path,
    buffer_m
):
    """
    geometry : shapely geometry
    geom_crs : GeoDataFrame CRS
    buffer_m : buffer in REAL meters (auto UTM)
    """

    with rio.open(tif_path) as src:

        geom = geometry

        # ---------- centroid ALWAYS from WGS84 ----------
        to_wgs = Transformer.from_crs(geom_crs, "EPSG:4326", always_xy=True).transform
        centroid_wgs = transform(to_wgs, geom).centroid

        lon, lat = centroid_wgs.x, centroid_wgs.y

        zone = int((lon + 180) / 6) + 1

        if lat >= 0:
            utm_epsg = 32600 + zone
        else:
            utm_epsg = 32700 + zone

        utm_crs = f"EPSG:{utm_epsg}"

        # ---------- now project geometry into raster CRS ----------
        if geom_crs != src.crs:
            to_raster = Transformer.from_crs(
                geom_crs, src.crs, always_xy=True
            ).transform
            geom = transform(to_raster, geom)

        zone = int((lon + 180) / 6) + 1

        if lat >= 0:
            utm_epsg = 32600 + zone   # north
        else:
            utm_epsg = 32700 + zone   # south

        utm_crs = f"EPSG:{utm_epsg}"

        to_utm = Transformer.from_crs(src.crs, utm_crs, always_xy=True).transform
        to_src = Transformer.from_crs(utm_crs, src.crs, always_xy=True).transform

        geom_utm = transform(to_utm, geom)
        geom_buf = geom_utm.buffer(buffer_m)
        geom_buf = transform(to_src, geom_buf)

        # ---------- bounding box ----------
        minx, miny, maxx, maxy = geom_buf.bounds
        win = from_bounds(minx, miny, maxx, maxy, src.transform)

        crop = win.round_offsets().round_lengths()

        data = src.read([1, 2, 3], window=crop, boundless=True, masked=True)
        img = np.moveaxis(data.filled(0), 0, -1)

        # ---------- normalize ----------
        if img.max() > 0:
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        else:
            img[:] = 128

        window_transform = src.window_transform(crop)

        return img, geom, window_transform


def load_gdb_polygons(gdb_path, layer, limit=20):
    gdf = gpd.read_file(gdb_path, layer=layer).head(limit)
    print(f"✅ {len(gdf)} polys from {layer}, CRS={gdf.crs}")
    return gdf
