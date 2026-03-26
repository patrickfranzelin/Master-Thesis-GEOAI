# ==============================
# repair.py
# ==============================
from __future__ import annotations

import numpy as np
from shapely import MultiPolygon
from shapely.geometry import Polygon, Point

from .geometry import dominant_axes, soft_snap, fit_line, normalize


def repair_dent(poly: Polygon, dent_data, cfg) -> Polygon:

    # ✅ FIX 1: handle MultiPolygon
    if isinstance(poly, MultiPolygon):
        # keep largest part (most likely the building)
        poly = max(poly.geoms, key=lambda g: g.area)

    # ✅ FIX 2: ensure valid geometry
    if not poly.is_valid:
        poly = poly.buffer(0)
    dent = dent_data["dent"]
    dom = dent_data["dominant"]

    coords = list(poly.exterior.coords)
    n = len(coords)

    inside = [Point(c).distance(dom) < cfg.repair_buffer for c in coords]

    enter_idx = None
    exit_idx = None

    for i in range(n):
        if not inside[i] and inside[(i + 1) % n]:
            enter_idx = i
        if inside[i] and not inside[(i + 1) % n]:
            exit_idx = (i + 1) % n

    if enter_idx is None or exit_idx is None:
        return poly

    if abs(enter_idx - exit_idx) < 2:
        return poly

    axis1, axis2 = dominant_axes(poly)

    p_enter = np.array(coords[enter_idx])
    p_exit = np.array(coords[exit_idx])

    dir1 = normalize(p_enter - np.array(coords[enter_idx - 1]))
    dir2 = normalize(np.array(coords[(exit_idx + 1) % n]) - p_exit)

    dir1 = soft_snap(dir1, axis1, axis2, 0.3)
    dir2 = soft_snap(dir2, axis1, axis2, 0.3)

    dot = abs(np.dot(dir1, dir2))

    line1 = fit_line(p_enter, dir1)
    line2 = fit_line(p_exit, dir2)

    if dot > 0.9:
        # parallel → connect along wall
        wall_dir = normalize(dir1 + dir2)
        line = fit_line(p_enter, wall_dir)

        proj1 = line.interpolate(line.project(Point(p_enter)))
        proj2 = line.interpolate(line.project(Point(p_exit)))

        p1 = proj1.coords[0]
        p2 = proj2.coords[0]

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n

        new_coords.append(coords[enter_idx])
        new_coords.append(p1)
        new_coords.append(p2)

    else:
        inter = line1.intersection(line2)

        if inter.is_empty or inter.geom_type != "Point":
            return poly

        ip = inter.coords[0]

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n

        new_coords.append(coords[enter_idx])
        new_coords.append(ip)

    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords).buffer(0)
        if fixed.is_valid and fixed.area > poly.area * 0.6:
            return fixed
    except Exception:
        pass

    return poly
