from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "sam2_weights" / "sam_building_decoder_finetuned.pth"

DEVICE = "cuda"

# Load fine-tuned SAM2
sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
sam.to(DEVICE)
sam.eval()

predictor = SamPredictor(sam)


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

    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    # Prepare prompts
    points = np.array(inside_pts + outside_pts)
    labels = np.array(
        [1] * len(inside_pts) + [0] * len(outside_pts)
    )

    if bbox is not None:
        box = np.array(bbox[0])
    else:
        box = None

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=False,
    )

    mask = (masks[0].astype(np.uint8) * 255)

    # Morph cleanup
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None

    cnt = max(contours, key=cv2.contourArea)

    if cnt.shape[0] < 3:
        return mask, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)

    return mask, poly
