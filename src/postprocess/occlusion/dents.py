import math
from shapely.geometry import Polygon
from shapely.ops import nearest_points


def fix_tree_dents(
    poly: Polygon,
    tree_union: Polygon,
    max_dent_area=120,
    proximity_dist=1.0,
):
    if poly.is_empty:
        return poly

    hull = poly.convex_hull
    dents = hull.difference(poly)

    if dents.is_empty:
        return poly

    dents = [dents] if dents.geom_type == "Polygon" else list(dents.geoms)

    cleaned = poly

    for dent in dents:

        # --- 1. size filter
        if dent.area > max_dent_area:
            continue

        # --- 2. proximity filter
        print("Dent distance:", dent.distance(tree_union))
        if dent.distance(tree_union) > proximity_dist:
            continue

        # --- 3. shape filter
        min_rect = dent.minimum_rotated_rectangle
        width = min_rect.bounds[2] - min_rect.bounds[0]
        height = min_rect.bounds[3] - min_rect.bounds[1]

        aspect_ratio = max(width, height) / max(min(width, height), 1e-6)

        if aspect_ratio < 1.5:
            continue

        # --- 4. direction check
        dent_center = dent.centroid
        tree_pt, _ = nearest_points(tree_union, dent_center)

        dx = tree_pt.x - dent_center.x
        dy = tree_pt.y - dent_center.y

        poly_center = poly.centroid
        px = poly_center.x - dent_center.x
        py = poly_center.y - dent_center.y

        if dx * px + dy * py > 0:
            continue

        #  THIS is where occlusion fix is ACTUALLY applied
        print("Tree occlusion fix applied! Dent area:", dent.area)

        cleaned = cleaned.union(dent)

    return cleaned