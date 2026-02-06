from ultralytics import SAM
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "sam3_weights" / "sam3.pt"

MODEL = SAM(str(MODEL_PATH))


# --------------------------------------------------
# Public API
# --------------------------------------------------

def run_sam(
    image_path: Path,
    inside_pts,
    outside_pts,
    bbox=None,
    morph_kernel=7,
):

    if len(inside_pts) == 0:
        return None, None

    img = cv2.imread(str(image_path))

    # ---------------------------------------------
    # Combine points + labels
    # ---------------------------------------------

    points = [inside_pts + outside_pts]
    labels = [[1] * len(inside_pts) + [0] * len(outside_pts)]

    # ---------------------------------------------
    # Run SAM
    # ---------------------------------------------

    result = MODEL.predict(
        source=img,
        points=points,
        labels=labels,
        bboxes=bbox,
        verbose=False,
    )

    if result[0].masks is None or len(result[0].masks.data) == 0:
        return None, None

    mask = result[0].masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    # ---------------------------------------------
    # Morph cleanup
    # ---------------------------------------------

    k = np.ones((morph_kernel, morph_kernel), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ---------------------------------------------
    # Mask → polygon
    # ---------------------------------------------

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None

    cnt = max(contours, key=cv2.contourArea)

    if cnt.shape[0] < 3:
        return mask, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)

    return mask, poly




