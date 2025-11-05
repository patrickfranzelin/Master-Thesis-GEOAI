from __future__ import annotations
import numpy as np, cv2
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from typing import Iterator, Tuple

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

def crop_for_polygon(src, poly, pad_factor: float = 0.25, min_pad_px: int = 64):
    """
    Crop image patch around a polygon with dynamic padding.

    Args:
        src: rasterio dataset
        poly: shapely Polygon (in CRS of src)
        pad_factor: fraction of polygon size to pad around bounding box (default: 25%)
        min_pad_px: minimum padding in pixels (default: 64)
    Returns:
        rgb (H×W×3 uint8), poly_xy (list of tuples)
    """
    inv = ~src.transform
    minx, miny, maxx, maxy = poly.bounds

    # Compute pixel-space bbox
    cmin, rmin = inv * (minx, miny)
    cmax, rmax = inv * (maxx, maxy)
    width_px = abs(cmax - cmin)
    height_px = abs(rmax - rmin)

    # Dynamic padding: fraction of object size, but not below min_pad_px
    pad_px = int(max(min_pad_px, pad_factor * max(width_px, height_px)))

    r0 = int(max(0, np.floor(min(rmin, rmax)) - pad_px))
    r1 = int(min(src.height, np.ceil(max(rmin, rmax)) + pad_px))
    c0 = int(max(0, np.floor(min(cmin, cmax)) - pad_px))
    c1 = int(min(src.width, np.ceil(max(cmin, cmax)) + pad_px))

    window = rasterio.windows.Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0))
    arr = src.read(window=window, out_dtype=np.uint8)
    rgb = np.moveaxis(arr[:3], 0, -1)

    # Local coordinates of polygon within crop
    xs = (np.array([p[0] for p in poly.exterior.coords]) - (src.transform.c + c0 * src.transform.a)) / src.transform.a
    ys = (np.array([p[1] for p in poly.exterior.coords]) - (src.transform.f + r0 * src.transform.e)) / src.transform.e
    poly_xy = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

    # Ensure RGB is valid (some edge tiles might be partial)
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("Invalid crop (polygon near image edge).")

    return rgb, poly_xy
