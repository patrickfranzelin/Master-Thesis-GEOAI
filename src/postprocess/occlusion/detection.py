# ==============================
# detection.py
# ==============================
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .geometry import to_lines
from .dent_split import split_dents


@dataclass
class OcclusionConfig:
    tree_outer_buffer: float = 0.3
    tree_inner_buffer: float = 0.3
    min_edge_length: float = 1.0
    min_overlap_ratio: float = 0.3
    min_curvature: float = 0.05
    repair_buffer: float = 0.3


# ------------------------------
# helpers
# ------------------------------

def curvature(edge) -> float:
    lines = to_lines(edge)
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


def dominant(edge) -> LineString | None:
    lines = to_lines(edge)
    if not lines:
        return None
    return max(lines, key=lambda g: g.length)


# ------------------------------
# MAIN DETECTION
# ------------------------------

def detect_dents(poly: Polygon, tree_union, cfg: OcclusionConfig):

    tree_boundary = tree_union.boundary

    tree_outer = tree_union.buffer(cfg.tree_outer_buffer)
    tree_inner = tree_union.buffer(-cfg.tree_inner_buffer)
    tree_ring = tree_outer.difference(tree_inner)

    edge = poly.boundary.intersection(tree_ring)

    dents = split_dents(edge, cfg)

    results = []

    for dent in dents:

        if dent.length < cfg.min_edge_length:
            continue

        overlap = dent.buffer(0.2).intersection(tree_boundary)
        overlap_ratio = overlap.length / dent.length if dent.length > 0 else 0

        if overlap_ratio < cfg.min_overlap_ratio:
            continue

        curv = curvature(dent)
        if curv < cfg.min_curvature:
            continue

        dom = dominant(dent)
        if dom is None:
            continue

        results.append({
            "dent": dent,
            "dominant": dom,
            "curvature": curv,
            "overlap_ratio": overlap_ratio,
        })

    return results