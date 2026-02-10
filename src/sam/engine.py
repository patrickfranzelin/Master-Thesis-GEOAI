# src/sam/engine.py

from ultralytics import SAM
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "sam3_weights" / "sam3.pt"

_MODEL = SAM(str(MODEL_PATH))


def segment_with_points(
    image_path: Path,
    inside_pts,
    outside_pts=None,
    bbox=None,
    morph_kernel=7,
):
    if not inside_pts:
        return None, None

    outside_pts = outside_pts or []
    img = cv2.imread(str(image_path))

    points = [inside_pts + outside_pts]
    labels = [[1] * len(inside_pts) + [0] * len(outside_pts)]

    result = _MODEL.predict(
        source=img,
        points=points,
        labels=labels,
        bboxes=bbox,
        verbose=False,
    )

    if not result or result[0].masks is None:
        return None, None

    mask = (result[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)

    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 3:
        return mask, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)
    return mask, poly
