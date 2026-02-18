# src/sam/model_.py

import torch
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base SAM checkpoint
BASE_CHECKPOINT = PROJECT_ROOT / "models" / "sam3_weights" / "sam_vit_b_01ec64.pth"

# Fine-tuned decoder weights
DECODER_CHECKPOINT = PROJECT_ROOT / "models" / "sam2_weights" / "sam_building_decoder_finetuned.pth"

# --------------------------------------------------
# Load Base Model
# --------------------------------------------------

sam = sam_model_registry["vit_b"](checkpoint=str(BASE_CHECKPOINT))
sam.to(DEVICE)
sam.eval()

# Load fine-tuned decoder
decoder_weights = torch.load(DECODER_CHECKPOINT, map_location=DEVICE)
sam.mask_decoder.load_state_dict(decoder_weights)

predictor = SamPredictor(sam)

print("✅ SAM2 fine-tuned decoder loaded")


# --------------------------------------------------
# Public API
# --------------------------------------------------

def segment_with_points(
    image_path: Path,
    inside_pts,
    outside_pts=None,
    bbox=None,
    morph_kernel=7,
    debug=False,
    multimask_output=False,
    mode=None,
):

    if not inside_pts:
        return None, None

    outside_pts = outside_pts or []

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    points = np.array(inside_pts + outside_pts)
    labels = np.array([1] * len(inside_pts) + [0] * len(outside_pts))
    box = np.array(bbox) if bbox is not None else None

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=multimask_output,
    )

    if masks is None or len(masks) == 0:
        return None, None

    all_masks = []
    all_polys = []

    for mask_raw in masks:

        mask = (mask_raw * 255).astype(np.uint8)
#---------------------------------------------------------------------------------------IMPORTANT
        k_close = np.ones((7, 7), np.uint8)  # was 15
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        mask = cv2.medianBlur(mask, 5)  # was 9

        k_open = np.ones((3, 3), np.uint8)  # was 5
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 1000:
                continue

            if cnt.shape[0] < 3:
                continue

            poly = Polygon(cnt.squeeze()).simplify(
                2.0,
                preserve_topology=True
            )

            all_masks.append(mask)
            all_polys.append(poly)

    if not all_polys:
        return None, None

    return all_masks, all_polys


