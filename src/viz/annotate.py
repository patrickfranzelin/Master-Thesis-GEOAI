import cv2, numpy as np
from typing import List, Tuple

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (255, 255, 255)
THICKNESS = 1

def overlay_numbered_grid(img: np.ndarray, grid_size=50, color=(180, 180, 180)):
    """Draw a numbered grid for coordinate reference."""
    vis = img.copy()
    h, w = vis.shape[:2]
    for x in range(0, w, grid_size):
        cv2.line(vis, (x, 0), (x, h), color, 1)
        cv2.putText(vis, str(x), (x + 2, 12), FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)
    for y in range(0, h, grid_size):
        cv2.line(vis, (0, y), (w, y), color, 1)
        cv2.putText(vis, str(y), (2, y + 12), FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)
    return vis

def is_black_or_empty(img: np.ndarray, threshold=5):
    """Return True if image is mostly black or empty."""
    return img.mean() < threshold or np.std(img) < 1

def plot_mllm_points(img, inside, outside, poly_xy):
    """Overlay polygon and MLLM points on RGB image."""
    vis = img.copy()
    if poly_xy:
        cv2.polylines(vis, [np.array(poly_xy, np.int32)], True, (0, 255, 255), 2)
    for (x, y) in inside:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    for (x, y) in outside:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
    return vis