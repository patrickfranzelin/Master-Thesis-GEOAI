import cv2
import rasterio as rio
import numpy as np
import geopandas as gpd

from shapely.ops import unary_union, transform
from rasterio.windows import from_bounds
from pyproj import Transformer

# -----------------------------------------------------
# Load + preprocess building footprints (RUN ONCE)
# -----------------------------------------------------

def load_gdb_polygons(gdb_path, layer, min_area=20, merge_dist=0):
    gdf = gpd.read_file(gdb_path, layer=layer)

    print(f"Loaded {len(gdf)} raw polygons")

    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    gdf = gdf[gdf.area > min_area]
    print(f"After area filter: {len(gdf)}")

    # Only merge buildings closer than merge_dist meters
    merged = unary_union(gdf.geometry.buffer(merge_dist))

    if merged.geom_type == "Polygon":
        geoms = [merged]
    else:
        geoms = list(merged.geoms)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=utm)

    print(f"After dissolve: {len(gdf)} building objects")

    return gdf



# -----------------------------------------------------
# Patch extraction
# -----------------------------------------------------

def extract_patch(geom, utm_crs, raster_path, out_size=512):

    with rio.open(raster_path) as src:

        # -------------------------------
        # Geometry → raster CRS
        # -------------------------------

        to_raster = Transformer.from_crs(utm_crs, src.crs, always_xy=True).transform
        geom_raster = transform(to_raster, geom)

        # -------------------------------
        # Back to UTM for metric sizing
        # -------------------------------

        to_utm = Transformer.from_crs(src.crs, utm_crs, always_xy=True).transform
        geom_m = transform(to_utm, geom_raster)

        minx, miny, maxx, maxy = geom_m.bounds
        size = max(maxx - minx, maxy - miny)

        margin = 8
        radius = np.clip(size / 2 + margin, 6, 50)

        box_m = geom_m.centroid.buffer(radius).envelope

        box = transform(
            Transformer.from_crs(utm_crs, src.crs, always_xy=True).transform,
            box_m
        )

        # -------------------------------
        # Raster crop
        # -------------------------------

        win = from_bounds(*box.bounds, src.transform)
        win = win.round_offsets().round_lengths()

        data = src.read([1, 2, 3], window=win, boundless=True, fill_value=128)
        img = np.moveaxis(data, 0, -1)

        # -------------------------------
        # Convert polygon → pixel coords
        # -------------------------------

        inv = ~src.window_transform(win)

        def to_px(x, y):
            return inv * (x, y)

        geom_px = transform(lambda x, y: to_px(x, y), geom_raster)

        # -------------------------------
        # Square pad + resize
        # -------------------------------

        h, w, _ = img.shape
        s = max(h, w)

        canvas = np.full((s, s, 3), 128, np.uint8)
        canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = img

        scale = out_size / s

        canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)

        # scale polygon too
        geom_px = transform(lambda x, y: (x + (s-w)//2, y + (s-h)//2), geom_px)
        geom_px = transform(lambda x, y: (x * scale, y * scale), geom_px)

        return canvas, geom_px

