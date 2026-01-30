import cv2
import rasterio as rio
import numpy as np
from rasterio.windows import from_bounds
from shapely.ops import transform, unary_union
from pyproj import Transformer
import geopandas as gpd


# ---------------------------------------------------------
# Grow touching / almost-touching buildings
# ---------------------------------------------------------

def grow_building(seed_geom, gdf, gdf_crs, utm_crs, tol=1.5):

    gdf_utm = gdf.to_crs(utm_crs)
    seed = gpd.GeoSeries([seed_geom], crs=gdf_crs).to_crs(utm_crs).iloc[0]

    visited = set()
    stack = [seed]
    parts = []

    while stack:
        geom = stack.pop()

        hits = gdf_utm[
            (gdf_utm.geometry.intersects(geom)) |
            (gdf_utm.geometry.distance(geom) < tol)
        ]

        for idx, row in hits.iterrows():
            if idx not in visited:
                visited.add(idx)
                parts.append(row.geometry)
                stack.append(row.geometry)

    if len(parts) == 0:
        return seed

    return unary_union(parts)


# ---------------------------------------------------------
# Main extraction
# ---------------------------------------------------------

def extract_patch_from_gdb(seed_geom, gdf, gdf_crs, tif_path, out_size=512):

    with rio.open(tif_path) as src:

        # -------------------------------------------------
        # 1) Auto UTM from centroid
        # -------------------------------------------------
        to_wgs = Transformer.from_crs(gdf_crs, "EPSG:4326", always_xy=True).transform
        lon, lat = transform(to_wgs, seed_geom).centroid.coords[0]

        zone = int((lon + 180) / 6) + 1
        utm = f"EPSG:{32600+zone if lat>=0 else 32700+zone}"

        # -------------------------------------------------
        # 2) Conditional grow
        # -------------------------------------------------
        gdf_utm = gdf.to_crs(utm)
        seed_utm = gpd.GeoSeries([seed_geom], crs=gdf_crs).to_crs(utm).iloc[0]

        neighbors = gdf_utm[
            (gdf_utm.geometry.intersects(seed_utm)) |
            (gdf_utm.geometry.distance(seed_utm) < 1.5)
        ]

        if len(neighbors) > 1:
            geom = grow_building(seed_geom, gdf, gdf_crs, utm)
        else:
            geom = seed_geom

        # -------------------------------------------------
        # 3) Project merged geometry to raster CRS
        # -------------------------------------------------
        if gdf_crs != src.crs:
            to_raster = Transformer.from_crs(gdf_crs, src.crs, always_xy=True).transform
            geom = transform(to_raster, geom)

        # -------------------------------------------------
        # 4) Metric geometry
        # -------------------------------------------------
        to_utm = Transformer.from_crs(src.crs, utm, always_xy=True).transform
        to_src = Transformer.from_crs(utm, src.crs, always_xy=True).transform

        geom_utm = transform(to_utm, geom)

        if geom_utm.is_empty:
            geom_utm = seed_utm

        # -------------------------------------------------
        # 5) Adaptive buffer
        # -------------------------------------------------
        minx, miny, maxx, maxy = geom_utm.bounds

        if not np.all(np.isfinite([minx, miny, maxx, maxy])):
            buffer_m = 30
        else:
            building_size = max(maxx-minx, maxy-miny)

            if not np.isfinite(building_size) or building_size <= 0:
                buffer_m = 30
            else:
                buffer_m = float(np.clip(building_size * 0.6, 10, 150))

        geom_buf = transform(to_src, geom_utm.buffer(buffer_m))

        # -------------------------------------------------
        # 6) Crop raster
        # -------------------------------------------------
        win = from_bounds(*geom_buf.bounds, src.transform)
        win = win.round_offsets().round_lengths()

        data = src.read([1,2,3], window=win, boundless=True, masked=True)
        img = np.moveaxis(data.filled(0), 0, -1)

        # -------------------------------------------------
        # 7) Square pad (no distortion)
        # -------------------------------------------------
        h, w, _ = img.shape
        s = max(h, w)
        canvas = np.full((s, s, 3), 128, np.uint8)

        y0 = (s-h)//2
        x0 = (s-w)//2
        canvas[y0:y0+h, x0:x0+w] = img

        img = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)

        # normalize
        if img.max() > 0:
            p2,p98 = np.percentile(img,(2,98))
            img = np.clip((img-p2)/(p98-p2)*255,0,255).astype(np.uint8)

        return img


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

def load_gdb_polygons(gdb_path, layer, limit=20):
    gdf = gpd.read_file(gdb_path, layer=layer).head(limit)
    print(f"✅ {len(gdf)} polys from {layer}, CRS={gdf.crs}")
    return gdf
