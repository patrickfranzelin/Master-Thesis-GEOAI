from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


# ============================================================
# CONFIG
# ============================================================

@dataclass
class OcclusionConfig:
    tree_outer_buffer: float = 0.3
    tree_inner_buffer: float = 0.3
    min_edge_length: float = 1.0
    min_overlap_ratio: float = 0.3
    min_curvature: float = 0.05
    repair_buffer: float = 0.3


# ============================================================
# BASIC HELPERS
# ============================================================

def _to_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return list(geom.geoms)


def _geom_parts(geom):
    if geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if not g.is_empty]
    return [geom]


def _to_lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if g.geom_type == "LineString"]
    return []


# ============================================================
# CURVATURE + DOMINANT EDGE
# ============================================================

def _curvature(edge) -> float:
    lines = _to_lines(edge)
    if not lines:
        return 0.0

    seg = max(lines, key=lambda g: g.length)
    coords = list(seg.coords)

    if len(coords) < 3:
        return 0.0

    chord = LineString([coords[0], coords[-1]])
    chord_len = max(chord.length, 1e-9)

    max_dev = max(Point(c).distance(chord) for c in coords)
    return max_dev / chord_len


def _dominant(edge) -> Optional[LineString]:
    lines = _to_lines(edge)
    if not lines:
        return None
    return max(lines, key=lambda g: g.length)


# ============================================================
# GEOMETRY / DIRECTION
# ============================================================

def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _fit_local_direction(coords, idx, window=3):
    n = len(coords)
    pts = [coords[(idx + i) % n] for i in range(-window, window + 1)]
    pts = np.array(pts)

    mean = pts.mean(axis=0)
    centered = pts - mean

    _, _, vh = np.linalg.svd(centered)
    return _normalize(vh[0])


def _fit_line(anchor, direction, scale=1000):
    return LineString([
        (anchor[0] - direction[0] * scale, anchor[1] - direction[1] * scale),
        (anchor[0] + direction[0] * scale, anchor[1] + direction[1] * scale),
    ])


def _dominant_axes(poly: Polygon):
    coords = np.array(poly.exterior.coords[:-1])
    edges = np.diff(np.vstack([coords, coords[0]]), axis=0)

    lengths = np.linalg.norm(edges, axis=1)
    directions = edges / (lengths[:, None] + 1e-9)

    mask = lengths > np.percentile(lengths, 60)
    directions = directions[mask]

    if len(directions) < 2:
        return np.array([1, 0]), np.array([0, 1])

    directions = np.where(directions[:, 0:1] < 0, -directions, directions)

    angles = np.arctan2(directions[:, 1], directions[:, 0])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=5).fit(angles.reshape(-1, 1))

    a = kmeans.cluster_centers_.flatten()

    axis1 = np.array([np.cos(a[0]), np.sin(a[0])])
    axis2 = np.array([-axis1[1], axis1[0]])

    return _normalize(axis1), _normalize(axis2)


def _soft_snap(direction, axis1, axis2, strength=0.3):
    direction = _normalize(direction)

    d1 = abs(np.dot(direction, axis1))
    d2 = abs(np.dot(direction, axis2))
    target = axis1 if d1 >= d2 else axis2

    if np.dot(direction, target) < 0:
        target = -target

    blended = (1 - strength) * direction + strength * target
    return _normalize(blended)


# ============================================================
# SNAP CLOSE (IMPORTANT)
# ============================================================

def _snap_if_close(p1, p2, tol):
    if np.linalg.norm(p1 - p2) < tol:
        return (p1 + p2) / 2
    return None


# ============================================================
# REPAIR (CORE)
# ============================================================

def _repair_polygon(poly: Polygon, dom: LineString, repair_buffer: float) -> Polygon:
    coords = list(poly.exterior.coords)
    n = len(coords)

    inside = [Point(c).distance(dom) < repair_buffer for c in coords]

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

    # --------------------------------------------------------
    # DIRECTIONS
    # --------------------------------------------------------
    axis1, axis2 = _dominant_axes(poly)

    p_enter = np.array(coords[enter_idx])
    p_exit  = np.array(coords[exit_idx])

    dir1 = _fit_local_direction(coords, enter_idx)
    dir2 = _fit_local_direction(coords, exit_idx)

    dir1 = _soft_snap(dir1, axis1, axis2, 0.3)
    dir2 = _soft_snap(dir2, axis1, axis2, 0.3)

    # avoid parallel collapse
    if abs(np.dot(dir1, dir2)) > 0.98:
        dir2 = np.array([-dir1[1], dir1[0]])

    # --------------------------------------------------------
    # EXTEND BOTH LINES (KEEP THIS!)
    # --------------------------------------------------------
    line1 = _fit_line(p_enter, dir1)
    line2 = _fit_line(p_exit,  dir2)

    inter = line1.intersection(line2)

    # --------------------------------------------------------
    # SNAP / FALLBACK
    # --------------------------------------------------------
    if inter.is_empty or inter.geom_type != "Point":

        snap = _snap_if_close(p_enter, p_exit, repair_buffer)

        if snap is not None:
            inter_pt = snap
        else:
            proj1 = dom.interpolate(dom.project(Point(p_enter)))
            proj2 = dom.interpolate(dom.project(Point(p_exit)))

            inter_pt = np.array([
                (proj1.x + proj2.x) / 2,
                (proj1.y + proj2.y) / 2
            ])
    else:
        inter_pt = np.array(inter.coords[0])

    # --------------------------------------------------------
    # BUILD POLYGON
    # --------------------------------------------------------
    new_coords = []
    i = exit_idx

    while i != enter_idx:
        new_coords.append(coords[i])
        i = (i + 1) % n

    new_coords.append(coords[enter_idx])
    new_coords.append(tuple(inter_pt))

    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords).buffer(0)

        if fixed.is_valid and fixed.area > poly.area * 0.6:
            return fixed

    except Exception:
        pass

    return poly


# ============================================================
# MAIN
# ============================================================

def detect_tree_occlusions(building_geom, tree_geom, cfg: Optional[OcclusionConfig] = None):

    cfg = cfg or OcclusionConfig()

    buildings = _to_polygons(building_geom)
    trees = _to_polygons(tree_geom)

    if not buildings or not trees:
        return building_geom, []

    tree_union = unary_union(trees)
    tree_boundary = tree_union.boundary

    tree_outer = tree_union.buffer(cfg.tree_outer_buffer)
    tree_inner = tree_union.buffer(-cfg.tree_inner_buffer)
    tree_ring = tree_outer.difference(tree_inner)

    repaired = []
    detections = []

    for poly in buildings:

        edge = poly.boundary.intersection(tree_ring)

        if edge.is_empty or edge.length < cfg.min_edge_length:
            repaired.append(poly)
            continue

        overlap = edge.buffer(0.2).intersection(tree_boundary)
        overlap_ratio = overlap.length / edge.length if edge.length > 0 else 0

        if overlap_ratio < cfg.min_overlap_ratio:
            repaired.append(poly)
            continue

        curv = _curvature(edge)

        if curv < cfg.min_curvature:
            repaired.append(poly)
            continue

        dom = _dominant(edge)
        if dom is None:
            repaired.append(poly)
            continue

        fixed = _repair_polygon(poly, dom, cfg.repair_buffer)
        repaired.append(fixed)

        detections.append({
            "edge_length": edge.length,
            "curvature": curv,
            "overlap_ratio": overlap_ratio,
        })

    result = unary_union(repaired).buffer(0)

    if result.geom_type == "MultiPolygon":
        result = max(result.geoms, key=lambda g: g.area)

    return result, detections