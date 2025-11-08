from __future__ import annotations
import numpy as np, cv2
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from typing import Iterator, Tuple
import numpy as np, random
from shapely.geometry import Polygon, MultiPolygon, mapping

def local_to_global_points(local_points, src_transform, c0, r0):
    global_pts = []
    for x, y in local_points:
        gx = src_transform.c + (c0 + x) * src_transform.a
        gy = src_transform.f + (r0 + y) * src_transform.e
        global_pts.append((gx, gy))
    return global_pts

def to_rgb_uint8(bxhxw: np.ndarray) -> np.ndarray:
    arr = bxhxw[:3] if bxhxw.shape[0] >= 3 else np.vstack([bxhxw] + [bxhxw[-1:]]*(3-bxhxw.shape[0]))
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [2, 98])
        scale = max(hi - lo, 1e-6)
        arr = np.clip((arr - lo)/scale, 0, 1)
        arr = (arr*255).astype(np.uint8)
    return arr

def enhance_local_contrast(rgb: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    except Exception:
        return rgb

def iter_tiles(src, tile: int, overlap: int) -> Iterator[Tuple[Window, np.ndarray, np.ndarray]]:
    W, H = src.width, src.height
    tw, th = min(tile, W), min(tile, H)
    y_step, x_step = max(1, th - overlap), max(1, tw - overlap)
    for top in range(0, H, y_step):
        for left in range(0, W, x_step):
            w = min(tw, W - left); h = min(th, H - top)
            if w <= 0 or h <= 0: continue
            win = Window(left, top, w, h)
            arr = src.read(out_dtype=np.float32, window=win, resampling=Resampling.bilinear)
            alpha = None
            if src.count >= 4:
                try: alpha = src.read(4, window=win, out_dtype=np.uint8)
                except Exception: pass
            yield win, arr, alpha

def crop_for_polygon(src, geom, pad_factor=0.3, min_pad_px=64):
    """
    Crop raster around a polygon with dynamic, centered padding.
    Returns: (rgb, poly_xy, (c0, r0))
    """
    import rasterio
    import numpy as np
    from shapely.geometry import mapping, MultiPolygon, Polygon

    # Use largest part if MultiPolygon
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)

    bounds = geom.bounds
    minx, miny, maxx, maxy = bounds
    row_min, col_min = src.index(minx, maxy)
    row_max, col_max = src.index(maxx, miny)

    width = col_max - col_min
    height = row_max - row_min
    pad = max(int(pad_factor * max(width, height)), min_pad_px)

    # --- Center the crop around the polygon ---
    cx = (col_min + col_max) // 2
    cy = (row_min + row_max) // 2
    x1 = max(0, cx - (width // 2 + pad))
    x2 = min(src.width, cx + (width // 2 + pad))
    y1 = max(0, cy - (height // 2 + pad))
    y2 = min(src.height, cy + (height // 2 + pad))

    window = ((y1, y2), (x1, x2))
    rgb = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))

    # Normalize safely
    ptp = np.ptp(rgb)
    rgb = np.clip((rgb - rgb.min()) / (ptp if ptp != 0 else 1e-6) * 255, 0, 255).astype(np.uint8)

    # Convert polygon to local coords
    poly_xy = [(int(src.index(x, y)[1] - x1), int(src.index(x, y)[0] - y1))
               for x, y in np.array(geom.exterior.coords)]

    return rgb, poly_xy, (x1, y1)



def sample_polygons(gdf: gpd.GeoDataFrame, n=40, seed=42):
    """Sample polygons roughly evenly spread across extent."""
    random.seed(seed)
    if len(gdf) <= n:
        return list(gdf.geometry)
    bounds = gdf.total_bounds
    xs = np.linspace(bounds[0], bounds[2], int(np.sqrt(n)))
    ys = np.linspace(bounds[1], bounds[3], int(np.sqrt(n)))
    samples = []
    for gx in xs:
        for gy in ys:
            subset = gdf.cx[gx:gx, gy:gy]
            if len(subset) > 0:
                samples.append(subset.sample(1, random_state=seed).geometry.iloc[0])
    if len(samples) < n:
        extras = gdf.sample(n - len(samples), random_state=seed)
        samples.extend(list(extras.geometry))
    return samples[:n]
