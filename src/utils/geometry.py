from shapely.geometry import Polygon


from shapely.geometry import Polygon


def polygon_to_sam_bbox(poly: Polygon, scale=0.8, pad_frac=0.15, min_size=200):
    """
    Adaptive SAM bbox from footprint.

    scale     : how much of footprint size to buffer (0.5–1.0 typical)
    pad_frac  : extra bbox padding relative to size
    min_size  : minimum bbox width/height in pixels
    """

    if poly is None or poly.is_empty:
        return None

    minx, miny, maxx, maxy = poly.bounds

    w = maxx - minx
    h = maxy - miny

    size = max(w, h)

    # grow polygon proportionally
    buffer_px = size * scale

    expanded = poly.buffer(buffer_px)

    if expanded.geom_type == "MultiPolygon":
        expanded = max(expanded.geoms, key=lambda g: g.area)

    minx, miny, maxx, maxy = expanded.bounds

    # enforce minimum bbox
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    half = max((maxx - minx) / 2, min_size / 2)

    pad = half * pad_frac

    return [[
        int(cx - half - pad),
        int(cy - half - pad),
        int(cx + half + pad),
        int(cy + half + pad),
    ]]

