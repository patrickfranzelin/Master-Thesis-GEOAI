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


def run_sam_multi_building(
    image_path: Path,
    buildings_data,
    negative_pts,
    morph_kernel=7,
):
    """
    Run SAM on multiple buildings in discovery mode.
    
    Args:
        image_path: Path to image
        buildings_data: List of dicts with 'inside_points' for each building
        negative_pts: Shared negative points for all buildings
        morph_kernel: Morphological operation kernel size
        
    Returns:
        List of (mask, polygon) tuples for each detected building
    """
    
    if len(buildings_data) == 0:
        return []
    
    img = cv2.imread(str(image_path))
    results = []
    
    for building in buildings_data:
        inside = building.get("inside_points", [])
        
        if len(inside) == 0:
            # Append (None, None) to maintain index alignment with buildings_data
            results.append((None, None))
            continue
        
        # Combine this building's inside points with shared negative points
        # Both inside and negative_pts are lists of [x,y] coordinate pairs
        points = [inside + negative_pts]  # SAM expects [[point1, point2, ...]]
        labels = [[1] * len(inside) + [0] * len(negative_pts)]
        
        # Run SAM for this building
        result = MODEL.predict(
            source=img,
            points=points,
            labels=labels,
            verbose=False,
        )
        
        if result[0].masks is None or len(result[0].masks.data) == 0:
            results.append((None, None))
            continue
        
        mask = result[0].masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        
        # Morph cleanup
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Mask → polygon
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            results.append((mask, None))
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        
        if cnt.shape[0] < 3:
            results.append((mask, None))
            continue
        
        poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)
        results.append((mask, poly))
    
    return results


