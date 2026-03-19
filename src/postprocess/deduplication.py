from shapely.ops import unary_union
from src.utils.geometry import ensure_polygon_list


def deduplicate_polygons(polys):

    if not polys:
        return []

    merged = unary_union(polys)

    return ensure_polygon_list(merged)