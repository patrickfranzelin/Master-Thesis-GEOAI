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


def _repair_polygon(poly: Polygon, dom: LineString, repair_buffer: float) -> Polygon:

    coords = list(poly.exterior.coords)
    n = len(coords)

    #  detect occluded segment (distance-based)
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

    # avoid tiny removals (likely corner noise)
    if abs(enter_idx - exit_idx) < 2:
        return poly

    # -------------------------------------------------
    # STEP 1: get neighboring edges
    # -------------------------------------------------

    prev_idx = (enter_idx - 1) % n
    next_idx = (exit_idx + 1) % n

    p_prev = coords[prev_idx]
    p_enter = coords[enter_idx]

    p_exit = coords[exit_idx]
    p_next = coords[next_idx]

    # -------------------------------------------------
    #  STEP 2: build lines (edge directions)
    # -------------------------------------------------

    line1 = _fit_line(p_prev, p_enter)
    line2 = _fit_line(p_exit, p_next)

    intersection = line1.intersection(line2)

    # -------------------------------------------------
    # fallback if lines don't intersect (parallel case)
    # -------------------------------------------------

    if intersection.is_empty or intersection.geom_type != "Point":
        # fallback → projection (your previous method)
        proj_enter = dom.interpolate(dom.project(Point(p_enter)))
        proj_exit = dom.interpolate(dom.project(Point(p_exit)))

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n
        new_coords.append(coords[enter_idx])

        new_coords.append((proj_enter.x, proj_enter.y))
        new_coords.append((proj_exit.x, proj_exit.y))

    else:
        # ------------------------------------------------
        #  TRUE CORNER RECONSTRUCTION
        # -------------------------------------------------

        ip = (intersection.x, intersection.y)

        new_coords = []
        i = exit_idx
        while i != enter_idx:
            new_coords.append(coords[i])
            i = (i + 1) % n
        new_coords.append(coords[enter_idx])

        #  insert reconstructed corner
        new_coords.append(ip)

    # -------------------------------------------------
    # validation
    # -------------------------------------------------

    if len(new_coords) < 3:
        return poly

    try:
        fixed = Polygon(new_coords).buffer(0)

        if (
            fixed.is_valid
            and not fixed.is_empty
            and fixed.area > poly.area * 0.6
        ):
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
