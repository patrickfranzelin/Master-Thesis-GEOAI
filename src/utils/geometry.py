from shapely.geometry import Polygon


from shapely.geometry import Polygon

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

