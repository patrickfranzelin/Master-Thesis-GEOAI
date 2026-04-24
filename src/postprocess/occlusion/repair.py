# ==============================
# repair.py
# ==============================
from __future__ import annotations

import numpy as np
from shapely import MultiPolygon
from shapely.geometry import Polygon, Point
from shapely.geometry import LineString
from .geometry import dominant_axes, soft_snap, fit_line, normalize


def repair_dent(poly: Polygon, dent_data, cfg) -> Polygon:
    from shapely.geometry import Point
    import numpy as np

    # -----------------------------
    # 0. SAFETY
    # -----------------------------
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    if not poly.is_valid:
        poly = poly.buffer(0)

    dent = dent_data["dent"]
    dom = dent_data["dominant"]

    coords = list(poly.exterior.coords)
    n = len(coords)

    # -----------------------------
    # 1. FIND OCCLUDED SEGMENT
    # -----------------------------
    inside = [Point(c).distance(dom) < cfg.repair_buffer for c in coords]

    enter_idx, exit_idx = None, None

    for i in range(n):
        if not inside[i] and inside[(i + 1) % n]:
            enter_idx = i
        if inside[i] and not inside[(i + 1) % n]:
            exit_idx = (i + 1) % n

    if enter_idx is None or exit_idx is None:
        return poly

    if abs(enter_idx - exit_idx) < 2:
        return poly

    # -----------------------------
    # 2. EXTRACT EDGE DIRECTIONS
    # -----------------------------
    def edge_direction(i1, i2):
        v = np.array(coords[i2]) - np.array(coords[i1])
        return v / (np.linalg.norm(v) + 1e-9)

    dir1 = edge_direction(enter_idx - 1, enter_idx)
    dir2 = edge_direction(exit_idx, (exit_idx + 1) % n)

    # -----------------------------
    # 3. MULTI-DIRECTION SNAP (🔥 key improvement)
    # -----------------------------
    def get_edge_directions(poly):
        dirs = []
        c = list(poly.exterior.coords)
        for i in range(len(c) - 1):
            v = np.array(c[i+1]) - np.array(c[i])
            if np.linalg.norm(v) < 1e-6:
                continue
            dirs.append(v / np.linalg.norm(v))
        return dirs

    def cluster_dirs(dirs, tol=15):
        clusters = []
        for d in dirs:
            found = False
            for c in clusters:
                if abs(np.dot(d, c)) > np.cos(np.radians(tol)):
                    c[:] = (c + d) / np.linalg.norm(c + d)
                    found = True
                    break
            if not found:
                clusters.append(d.copy())
        return clusters

    def idx(i):
        return i % n

    dirs = [
        edge_direction(idx(enter_idx - 2), idx(enter_idx - 1)),
        edge_direction(idx(enter_idx - 1), idx(enter_idx)),
        edge_direction(idx(exit_idx), idx(exit_idx + 1)),
        edge_direction(idx(exit_idx + 1), idx(exit_idx + 2)),
    ]
    clusters = cluster_dirs(dirs)

    def snap_to_cluster(v):
        best_v, best_score = None, -1
        for c in clusters:
            score = abs(np.dot(v, c))
            if score > best_score:
                best_score = score
                # CRITICAL: Always use the direction that POINTS TOWARDS the dent
                candidate = c if np.dot(v, c) > 0 else -c
                # Verify it reduces dent distance
                if candidate is not None:  # Add distance check to dent center
                    best_v = candidate
        return best_v

    dir1 = snap_to_cluster(dir1)
    dir2 = snap_to_cluster(dir2)

    p_enter = np.array(coords[enter_idx])
    p_exit = np.array(coords[exit_idx])
    dent_center = (p_enter + p_exit) / 2

    # Flip dir1 so it points TOWARDS dent_center from enter
    if np.dot(dir1, dent_center - p_enter) < 0:
        dir1 = -dir1

    # Flip dir2 so it points TOWARDS dent_center from exit
    if np.dot(dir2, dent_center - p_exit) < 0:
        dir2 = -dir2

    # -----------------------------
    # 4. ROBUST LINE EXTENSION (🔥 critical)
    # -----------------------------
    def make_ray(p, d, scale=20):  # small!
        return LineString([p, p + d * scale])

    p_enter = np.array(coords[enter_idx])
    p_exit = np.array(coords[exit_idx])

    line1 = make_ray(p_enter, dir1)
    line2 = make_ray(p_exit, dir2)

    dot = abs(np.dot(dir1, dir2))

    # -----------------------------
    # 5. PARALLEL CASE (wall continuation)
    # -----------------------------
    if dot > 0.95:
        wall_dir = dir1.copy()

        # fallback if opposite directions
        if np.dot(dir1, dir2) < 0:
            wall_dir = dir1
        else:
            wall_dir = normalize(dir1 + dir2)

        wall = make_ray(p_enter, wall_dir)

        proj1 = wall.interpolate(wall.project(Point(p_enter)))
        proj2 = wall.interpolate(wall.project(Point(p_exit)))

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

    # -----------------------------
    # 6. CORNER CASE (intersection)
    # -----------------------------
    else:
        inter = line1.intersection(line2)

        if inter.is_empty or inter.geom_type != "Point":
            return poly

        ip = inter.coords[0]
        ip_arr = np.array(ip)

        # distance constraint (prevents spikes)
        dent_scale = np.linalg.norm(p_exit - p_enter)
        max_dist = dent_scale * 3.0  # tunable (2–4 works well)

        if np.linalg.norm(ip_arr - dent_center) > max_dist:
            return poly
        # ensure intersection is roughly between directions
        v1 = normalize(ip_arr - p_enter)
        v2 = normalize(ip_arr - p_exit)

        if np.dot(v1, dir1) < 0.3 or np.dot(v2, dir2) < 0.3:
            return poly

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n

        new_coords.append(coords[enter_idx])
        new_coords.append(ip)

    # -----------------------------
    # 7. VALIDATION (🔥 important)
    # -----------------------------
    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords).buffer(0)

        # stronger validation
        iou = fixed.intersection(poly).area / fixed.union(poly).area

        if fixed.is_valid and iou > 0.5:
            return fixed

    except Exception:
        pass

    return poly
