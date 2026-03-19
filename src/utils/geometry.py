import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shp_transform

def pixel_to_world(poly, transform_affine):
    return shp_transform(lambda x, y: transform_affine * (x, y), poly)
def polygon_to_sam_bbox(poly: Polygon, img_shape, scale=0.8, pad_frac=0.15, min_size=80):

    if poly is None or poly.is_empty:
        return None

    h_img, w_img = img_shape[:2]

    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny

    size = max(w, h)

    # Expand without shapely buffer (safer)
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    half = max(size * (1 + scale) / 2, min_size / 2)
    pad = half * pad_frac

    x1 = int(cx - half - pad)
    y1 = int(cy - half - pad)
    x2 = int(cx + half + pad)
    y2 = int(cy + half + pad)

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img - 1, x2)
    y2 = min(h_img - 1, y2)

    return [[x1, y1, x2, y2]]

def mask_to_polygon(mask):

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        return None

    mask = (mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) < 3:
        return None

    return Polygon(cnt.squeeze()).simplify(1.5)

def ensure_polygon_list(geom):

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    return []
