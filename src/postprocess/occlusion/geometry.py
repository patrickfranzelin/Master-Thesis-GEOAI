# ==============================
# geometry.py
# ==============================
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

EPS = 1e-9


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)


def to_lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if g.geom_type == "LineString"]
    return []


def dominant_axes(poly: Polygon):
    coords = np.array(poly.exterior.coords[:-1])
    centered = coords - coords.mean(axis=0)

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    axis1 = eigvecs[:, np.argmax(eigvals)]
    axis2 = np.array([-axis1[1], axis1[0]])

    return normalize(axis1), normalize(axis2)


def soft_snap(direction, axis1, axis2, strength=0.3):
    direction = normalize(direction)

    d1 = abs(np.dot(direction, axis1))
    d2 = abs(np.dot(direction, axis2))

    target = axis1 if d1 >= d2 else axis2

    if np.dot(direction, target) < 0:
        target = -target

    blended = (1 - strength) * direction + strength * target
    return normalize(blended)


def fit_line(anchor, direction, scale=1000):
    return LineString([
        (anchor[0] - direction[0] * scale, anchor[1] - direction[1] * scale),
        (anchor[0] + direction[0] * scale, anchor[1] + direction[1] * scale),
    ])





