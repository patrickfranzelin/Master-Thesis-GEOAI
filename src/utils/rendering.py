import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Tuple


# --------------------------------------------------
# Grid overlay
# --------------------------------------------------

def add_grid_overlay(img: np.ndarray, step: int = 50) -> np.ndarray:
    """
    Draw white debug grid with cyan coordinate labels.
    """
    overlay = img.copy()
    h, w = img.shape[:2]

    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(
            overlay, str(x), (x + 3, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(
            overlay, str(y), (5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return overlay


# --------------------------------------------------
# Center star
# --------------------------------------------------

def add_center_star(
    img: np.ndarray,
    size: int = 25,
    color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Draw red star at true image center.
    """

    overlay = img.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    rad = size / 2

    for i in range(0, 10, 2):
        x1 = int(cx + rad * np.cos(angles[i]))
        y1 = int(cy + rad * np.sin(angles[i]))
        x2 = int(cx + (rad / 2) * np.cos(angles[i + 1]))
        y2 = int(cy + (rad / 2) * np.sin(angles[i + 1]))

        cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

    cv2.circle(overlay, (cx, cy), 6, color, -1)

    cv2.putText(
        overlay,
        "HOUSE",
        (cx - 30, cy - size - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    return overlay


# --------------------------------------------------
# Polygon overlay
# --------------------------------------------------

def add_polygon_overlay(
    img: np.ndarray,
    polygon,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw shapely Polygon or MultiPolygon on image.
    """

    overlay = img.copy()

    def _draw(poly: Polygon):
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, color, thickness)

    if isinstance(polygon, Polygon):
        _draw(polygon)

    elif isinstance(polygon, MultiPolygon):
        for p in polygon.geoms:
            _draw(p)

    return overlay


# --------------------------------------------------
# Inside / Outside points
# --------------------------------------------------

def draw_points(
    img: np.ndarray,
    inside_pts: List[List[int]],
    outside_pts: List[List[int]],
) -> np.ndarray:
    """
    Draw QA points:
      Green = inside
      Red = outside
    """

    overlay = img.copy()
    h, w = overlay.shape[:2]

    for x, y in inside_pts:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), 10, (0, 255, 0), -1)

    for x, y in outside_pts:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)

    return overlay
