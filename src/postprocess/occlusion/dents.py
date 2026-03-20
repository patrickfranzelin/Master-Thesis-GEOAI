from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize


@dataclass
class OcclusionConfig:
    tree_outer_buffer: float = 0.3
    tree_inner_buffer: float = 0.3
    min_edge_length: float = 1.0
    min_overlap_ratio: float = 0.3
    min_curvature: float = 0.05
    repair_buffer: float = 0.3


def _to_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return list(geom.geoms)


def _geom_parts(geom) -> List:
    if geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if not g.is_empty]
    return [geom]


def _curvature(edge) -> float:
    parts = _geom_parts(edge)
    if not parts:
        return 0.0
    seg = max(parts, key=lambda g: g.length)
    coords = list(seg.coords)
    if len(coords) < 3:
        return 0.0
    chord = LineString([coords[0], coords[-1]])
    chord_len = max(chord.length, 1e-9)
    max_dev = max(Point(c).distance(chord) for c in coords)
    return max_dev / chord_len


def _dominant(edge) -> Optional[LineString]:
    parts = [g for g in _geom_parts(edge) if g.geom_type == "LineString"]
    if not parts:
        return None
    return max(parts, key=lambda g: g.length)


def _fit_line(p1, p2, scale=1000):
    """Create long line from two points (for intersection)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return LineString([
        (p1[0] - dx * scale, p1[1] - dy * scale),
        (p1[0] + dx * scale, p1[1] + dy * scale),
    ])


import numpy as np

def _dominant_axes(poly: Polygon):
    """
    Robust dominant axes using only long, straight edges.
    """
    coords = np.array(poly.exterior.coords[:-1])
    edges = np.diff(np.vstack([coords, coords[0]]), axis=0)

    lengths = np.linalg.norm(edges, axis=1)
    directions = edges / (lengths[:, None] + 1e-9)

    # ---- FILTER: only long edges ----
    length_thresh = np.percentile(lengths, 60)   # keep top 40%
    mask = lengths > length_thresh

    directions = directions[mask]
    lengths = lengths[mask]

    if len(directions) < 2:
        # fallback
        return np.array([1, 0]), np.array([0, 1])

    # ---- normalize orientation (0–180°) ----
    directions = np.where(directions[:, 0:1] < 0, -directions, directions)

    # ---- cluster into 2 main directions ----
    # project angles
    angles = np.arctan2(directions[:, 1], directions[:, 0])

    # k-means (k=2) on angle
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=5).fit(angles.reshape(-1, 1))

    centers = kmeans.cluster_centers_.flatten()

    axis1 = np.array([np.cos(centers[0]), np.sin(centers[0])])
    axis2 = np.array([np.cos(centers[1]), np.sin(centers[1])])

    # enforce orthogonality
    axis2 = np.array([-axis1[1], axis1[0]])

    return axis1 / np.linalg.norm(axis1), axis2 / np.linalg.norm(axis2)


def _snap_to_axes(direction: np.ndarray, axis1: np.ndarray, axis2: np.ndarray) -> np.ndarray:
    """Snap a direction vector to whichever dominant axis it is closest to."""
    d1 = abs(np.dot(direction, axis1))
    d2 = abs(np.dot(direction, axis2))
    snapped = axis1 if d1 >= d2 else axis2
    # preserve original sign
    if np.dot(direction, snapped) < 0:
        snapped = -snapped
    return snapped


def _fit_line_from_wall(anchor: np.ndarray, direction: np.ndarray, scale: float = 1000) -> LineString:
    return LineString([
        (anchor[0] - direction[0] * scale, anchor[1] - direction[1] * scale),
        (anchor[0] + direction[0] * scale, anchor[1] + direction[1] * scale),
    ])
def _soft_snap(direction, axis1, axis2, strength=0.3):
    """
    Blend original direction with nearest dominant axis.
    strength = 0 → no snapping
    strength = 1 → full snapping
    """
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    d1 = abs(np.dot(direction, axis1))
    d2 = abs(np.dot(direction, axis2))
    target = axis1 if d1 >= d2 else axis2

    if np.dot(direction, target) < 0:
        target = -target

    blended = (1 - strength) * direction + strength * target
    return blended / (np.linalg.norm(blended) + 1e-9)

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

    # -------------------------------------------------
    # dominant axes (robust global orientation)
    # -------------------------------------------------
    axis1, axis2 = _dominant_axes(poly)

    # -------------------------------------------------
    # local geometry (IMPORTANT)
    # -------------------------------------------------
    prev_idx = (enter_idx - 1) % n
    next_idx = (exit_idx + 1) % n

    p_prev  = np.array(coords[prev_idx])
    p_enter = np.array(coords[enter_idx])
    p_exit  = np.array(coords[exit_idx])
    p_next  = np.array(coords[next_idx])

    raw_dir1 = p_enter - p_prev
    raw_dir2 = p_next  - p_exit

    # normalize
    raw_dir1 = raw_dir1 / (np.linalg.norm(raw_dir1) + 1e-9)
    raw_dir2 = raw_dir2 / (np.linalg.norm(raw_dir2) + 1e-9)

    # -------------------------------------------------
    # SNAP but KEEP geometry logic
    # -------------------------------------------------
    dir1 = _soft_snap(raw_dir1, axis1, axis2, strength=0.25)
    dir2 = _soft_snap(raw_dir2, axis1, axis2, strength=0.25)

    # preserve correct orientation
    if np.dot(raw_dir2, dir2) < 0:
        dir2 = -dir2
    # -------------------------------------------------
    # ENSURE DIRECTIONS ARE NOT PARALLEL
    # -------------------------------------------------
    if abs(np.dot(dir1, dir2)) > 0.9:
        # force orthogonal direction for dir2
        dir2 = np.array([-dir1[1], dir1[0]])

        # keep orientation consistent with original
        if np.dot(raw_dir2, dir2) < 0:
            dir2 = -dir2
    # -------------------------------------------------
    # EXTEND WALLS (this is what you lost)
    # -------------------------------------------------
    line1 = _fit_line_from_wall(p_enter, dir1)
    line2 = _fit_line_from_wall(p_exit,  dir2)

    intersection = line1.intersection(line2)

    # -------------------------------------------------
    # BUILD NEW POLYGON
    # -------------------------------------------------
    if intersection.is_empty or intersection.geom_type != "Point":
        # fallback → project onto dominant edge
        proj_enter = dom.interpolate(dom.project(Point(coords[enter_idx])))
        proj_exit  = dom.interpolate(dom.project(Point(coords[exit_idx])))

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n

        new_coords.append(coords[enter_idx])
        new_coords.append((proj_enter.x, proj_enter.y))
        new_coords.append((proj_exit.x, proj_exit.y))

    else:
        ip = (intersection.x, intersection.y)

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n

        new_coords.append(coords[enter_idx])
        new_coords.append(ip)

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords).buffer(0)
        if fixed.is_valid and not fixed.is_empty and fixed.area > poly.area * 0.6:
            return fixed
    except Exception:
        pass

    return poly

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

        # =====================================================
        # 🔧 REPAIR
        # =====================================================

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
