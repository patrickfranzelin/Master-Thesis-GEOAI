from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon

from .geometry import normalize


EPS = 1e-9


def _as_polygon(poly: Polygon) -> Polygon | None:
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def _runs(mask: list[bool]) -> list[list[int]]:
    if not mask or not any(mask):
        return []

    n = len(mask)
    seen = [False] * n
    runs = []

    for start in range(n):
        if seen[start] or not mask[start]:
            continue

        run = []
        i = start
        while mask[i] and not seen[i]:
            seen[i] = True
            run.append(i)
            i = (i + 1) % n

        runs.append(run)

    if len(runs) > 1 and mask[0] and mask[-1]:
        first = next(r for r in runs if 0 in r)
        last = next(r for r in runs if n - 1 in r)
        if first is not last:
            merged = last + first
            runs = [r for r in runs if r not in (first, last)]
            runs.append(merged)

    return runs


def _choose_occluded_run(coords, dent_line, repair_buffer):
    dent_mid = dent_line.interpolate(0.5, normalized=True)
    inside = [Point(c).distance(dent_line) <= repair_buffer for c in coords]
    candidates = _runs(inside)

    if not candidates:
        return None

    def score(run):
        length = 0.0
        dist = 0.0
        for idx in run:
            p = Point(coords[idx])
            dist += p.distance(dent_mid)
        for a, b in zip(run, run[1:]):
            length += Point(coords[a]).distance(Point(coords[b]))
        return (len(run), length, -dist / max(len(run), 1))

    return max(candidates, key=score)


def _building_axes(coords, excluded: set[int]) -> list[np.ndarray]:
    clusters: list[dict] = []
    n = len(coords)

    for i in range(n):
        if i in excluded or (i + 1) % n in excluded:
            continue

        p1 = np.array(coords[i], dtype=float)
        p2 = np.array(coords[(i + 1) % n], dtype=float)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 0.25:
            continue

        axis = normalize(vec)
        matched = None
        for cluster in clusters:
            if abs(np.dot(axis, cluster["axis"])) > math.cos(math.radians(18.0)):
                matched = cluster
                break

        if matched is None:
            clusters.append({"axis": axis, "weight": length})
            continue

        if np.dot(axis, matched["axis"]) < 0:
            axis = -axis
        new_weight = matched["weight"] + length
        matched["axis"] = normalize(
            matched["axis"] * matched["weight"] + axis * length
        )
        matched["weight"] = new_weight

    clusters.sort(key=lambda item: item["weight"], reverse=True)
    return [item["axis"] for item in clusters]


def _snap_to_axis(direction, axes, max_degrees=30.0):
    if not axes:
        return direction

    best = max(axes, key=lambda axis: abs(np.dot(direction, axis)))
    score = abs(np.dot(direction, best))
    if score < math.cos(math.radians(max_degrees)):
        return direction

    return best if np.dot(direction, best) >= 0 else -best


def _wall_direction(coords, anchor_idx, tangent_idx, target, axes):
    anchor = np.array(coords[anchor_idx], dtype=float)
    tangent = np.array(coords[tangent_idx], dtype=float) - anchor

    if np.linalg.norm(tangent) < EPS:
        direction = normalize(target - anchor)
    else:
        direction = normalize(tangent)

    direction = _snap_to_axis(direction, axes)
    if np.dot(direction, target - anchor) < 0:
        direction = -direction

    return direction


def _line_intersection(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None

    delta = p2 - p1
    t = (delta[0] * d2[1] - delta[1] * d2[0]) / cross
    u = (delta[0] * d1[1] - delta[1] * d1[0]) / cross
    return p1 + t * d1, t, u


def _parallel_patch_points(p_enter, dir1, p_exit, dir2):
    wall_dir = normalize(dir1 + dir2)
    if np.linalg.norm(wall_dir) < 1e-6:
        wall_dir = dir1

    bridge = p_exit - p_enter
    normal = np.array([-wall_dir[1], wall_dir[0]])
    offset = np.dot(bridge, normal)

    p1 = p_enter + normal * offset * 0.5
    p2 = p_exit - normal * offset * 0.5

    if np.linalg.norm(p2 - p1) < EPS:
        return [tuple(p_exit)]
    return [tuple(p1), tuple(p2)]


def _clean_ring(coords):
    cleaned = []
    for coord in coords:
        if not cleaned or Point(cleaned[-1]).distance(Point(coord)) > 1e-6:
            cleaned.append(coord)

    if len(cleaned) > 1 and Point(cleaned[0]).distance(Point(cleaned[-1])) <= 1e-6:
        cleaned.pop()

    return cleaned


def _has_spike(coords, min_angle_degrees=12.0, min_edge_length=0.03):
    if len(coords) < 4:
        return False

    min_sin = math.sin(math.radians(min_angle_degrees))
    n = len(coords)
    arr = [np.array(c, dtype=float) for c in coords]

    for i in range(n):
        prev_p = arr[(i - 1) % n]
        p = arr[i]
        next_p = arr[(i + 1) % n]
        v1 = prev_p - p
        v2 = next_p - p
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)

        if l1 < min_edge_length or l2 < min_edge_length:
            return True

        sin_angle = abs(v1[0] * v2[1] - v1[1] * v2[0]) / max(l1 * l2, EPS)
        if sin_angle < min_sin and np.dot(v1, v2) > 0:
            return True

    return False


def _validate_repair(original: Polygon, fixed, dent_line, patch_points) -> Polygon | None:
    fixed = _as_polygon(fixed)
    if fixed is None:
        return None

    if fixed.area <= 0:
        return None

    original_area = max(original.area, EPS)
    area_ratio = fixed.area / original_area
    if area_ratio < 0.65 or area_ratio > 1.45:
        return None

    union = fixed.union(original)
    if union.area <= EPS:
        return None

    iou = fixed.intersection(original).area / union.area
    if iou < 0.55:
        return None

    dent_center = np.array(dent_line.interpolate(0.5, normalized=True).coords[0])
    gap_scale = max(dent_line.length, 1.0)
    for point in patch_points:
        if np.linalg.norm(np.array(point) - dent_center) > gap_scale * 4.0:
            return None

    if _has_spike(list(fixed.exterior.coords[:-1])):
        return None

    return fixed


def repair_dent(poly: Polygon, dent_data, cfg) -> Polygon:
    poly = _as_polygon(poly)
    if poly is None:
        return poly

    dent = dent_data["dent"]
    dent_line = dent_data.get("dominant") or dent
    if dent_line is None or dent_line.is_empty:
        return poly

    coords = _clean_ring(list(poly.exterior.coords[:-1]))
    n = len(coords)
    if n < 4:
        return poly

    run = _choose_occluded_run(coords, dent_line, cfg.repair_buffer)
    if not run:
        return poly

    enter_idx = (run[0] - 1) % n
    exit_idx = (run[-1] + 1) % n
    if enter_idx == exit_idx or enter_idx in run or exit_idx in run:
        return poly

    p_enter = np.array(coords[enter_idx], dtype=float)
    p_exit = np.array(coords[exit_idx], dtype=float)
    dent_center = np.array(dent_line.interpolate(0.5, normalized=True).coords[0])

    excluded = set(run)
    axes = _building_axes(coords, excluded)

    dir1 = _wall_direction(coords, enter_idx, run[0], dent_center, axes)
    dir2 = _wall_direction(coords, exit_idx, run[-1], dent_center, axes)

    same_axis = abs(np.dot(dir1, dir2)) > 0.965
    if same_axis:
        patch_points = _parallel_patch_points(p_enter, dir1, p_exit, dir2)
    else:
        intersection = _line_intersection(p_enter, dir1, p_exit, dir2)
        if intersection is None:
            return poly

        ip, t, u = intersection
        if t < -0.05 or u < -0.05:
            return poly
        patch_points = [tuple(ip)]

    new_coords = []
    i = exit_idx
    while True:
        new_coords.append(coords[i])
        if i == enter_idx:
            break
        i = (i + 1) % n

    new_coords.extend(patch_points)
    new_coords = _clean_ring(new_coords)

    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords)
        if not fixed.is_valid:
            fixed = fixed.buffer(0)
    except Exception:
        return poly

    validated = _validate_repair(poly, fixed, dent_line, patch_points)
    return validated if validated is not None else poly
