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
        multimask_output=False,
    )

    if masks is None or len(masks) == 0:
        return None, None

    mask = (masks[0] * 255).astype(np.uint8)

    # Morph cleanup
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Mask → polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None

    cnt = max(contours, key=cv2.contourArea)

    if cnt.shape[0] < 3:
        return mask, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)

    return mask, poly
