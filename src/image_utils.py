import cv2
import numpy as np
from typing import Tuple

def add_grid_overlay(img, step=50):
    """Add prominent white grid with cyan labels (thinner lines)."""
    h, w, _ = img.shape
    overlay = img.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)  # Changed: 2 → 1 (thinner)
        cv2.putText(overlay, str(x), (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)  # Changed: 2 → 1 (thinner)
        cv2.putText(overlay, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return overlay


def add_center_star(img, size=25, color=(0, 0, 255)):
    """Red star at true image center."""
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
        2
    )
    return overlay

