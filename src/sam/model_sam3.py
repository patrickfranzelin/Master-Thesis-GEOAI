# src/sam/model_sam3.py

from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor

import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path


# ------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "sam3_weights" / "sam3.pt"

# Visual prompt mode (SAM2-compatible)
_MODEL_VISUAL = SAM(str(MODEL_PATH))

# Semantic concept mode (text / exemplar prompts)
_SEMANTIC_OVERRIDES = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str(MODEL_PATH),
    half=True,
    verbose=False,
)
_MODEL_SEMANTIC = SAM3SemanticPredictor(overrides=_SEMANTIC_OVERRIDES)


# ------------------------------------------------------------------
# Utility: mask -> shapely polygon
# ------------------------------------------------------------------

def _mask_to_polygon(mask, morph_kernel=15):
    mask = (mask * 255).astype(np.uint8)

    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    if cnt.shape[0] < 3:
        return None, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)

    return mask, poly


# ------------------------------------------------------------------
# 1️⃣ Visual Prompt Segmentation (Point / Box)
# ------------------------------------------------------------------

def segment_with_points(
    image_path: Path,
    inside_pts,
    outside_pts=None,
    bbox=None,
    morph_kernel=8,
):
    """
    SAM3 in visual prompt mode (SAM2 compatible).
    Deterministic refinement with inside/outside logic.
    """

    if not inside_pts:
        return None, None

    outside_pts = outside_pts or []
    img = cv2.imread(str(image_path))

    points = [inside_pts + outside_pts]
    labels = [[1] * len(inside_pts) + [0] * len(outside_pts)]

    results = _MODEL_VISUAL.predict(
        source=img,
        points=points,
        labels=labels,
        bboxes=bbox,
        verbose=False,
    )

    if not results or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    all_masks, all_polys = [], []

    for m in masks:
        mask, poly = _mask_to_polygon(m, morph_kernel)

        if poly is None:
            continue

        all_masks.append(mask)
        all_polys.append(poly)

    if not all_polys:
        return None, None

    return all_masks, all_polys


# ------------------------------------------------------------------
# 2️⃣ Text-Based Concept Segmentation (PCS Mode)
# ------------------------------------------------------------------

def segment_with_text(
    image_path: Path,
    text_prompts,
    morph_kernel=15,
):
    """
    SAM3 Promptable Concept Segmentation.
    Returns ALL instances matching text concept.
    """

    img = cv2.imread(str(image_path))
    _MODEL_SEMANTIC.set_image(img)

    results = _MODEL_SEMANTIC(text=text_prompts)

    if not results or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    all_masks, all_polys = [], []

    for m in masks:
        mask, poly = _mask_to_polygon(m, morph_kernel)

        if poly is None:
            continue

        all_masks.append(mask)
        all_polys.append(poly)

    if not all_polys:
        return None, None

    return all_masks, all_polys


# ------------------------------------------------------------------
# 3️⃣ Exemplar-Based Concept Segmentation
# ------------------------------------------------------------------

def segment_with_exemplar(
    image_path: Path,
    bboxes,
    morph_kernel=8,
):
    """
    Concept segmentation via exemplar bounding boxes.
    Finds similar objects across image.
    """

    img = cv2.imread(str(image_path))
    _MODEL_SEMANTIC.set_image(img)

    results = _MODEL_SEMANTIC(bboxes=bboxes)

    if not results or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    all_masks, all_polys = [], []

    for m in masks:
        mask, poly = _mask_to_polygon(m, morph_kernel)

        if poly is None:
            continue

        all_masks.append(mask)
        all_polys.append(poly)

    if not all_polys:
        return None, None

    return all_masks, all_polys