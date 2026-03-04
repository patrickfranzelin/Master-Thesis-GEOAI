import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from shapely.geometry import Polygon

import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = Path(sam3.__file__).resolve().parent.parent
BPE_PATH = SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# Singleton model + processor (loaded once)
_MODEL = None
_PROCESSOR = None

def _get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        _MODEL = build_sam3_image_model(bpe_path=str(BPE_PATH), enable_inst_interactivity=True)
        _PROCESSOR = Sam3Processor(_MODEL)
    return _MODEL, _PROCESSOR


def segment_with_points(
    image_path: Path,
    inside_pts,
    outside_pts=None,
    bbox=None,
    morph_kernel=8,
    debug=False,
    prev_logits=None,          # ← NEW: pass logits from previous iteration
):
    """
    Native SAM3 API wrapper. Returns (masks, polys, logits) so the caller
    can pass logits into the next iteration for the feedback loop.
    """
    if not inside_pts:
        return None, None, None

    outside_pts = outside_pts or []
    model, processor = _get_model()

    # Load image
    pil_image = Image.open(str(image_path)).convert("RGB")
    inference_state = processor.set_image(pil_image)

    # Build point arrays  [x, y] format
    all_pts = inside_pts + outside_pts
    all_labels = [1] * len(inside_pts) + [0] * len(outside_pts)

    input_point = np.array(all_pts)         # shape (N, 2)
    input_label = np.array(all_labels)       # shape (N,)

    # Box prompt  (xyxy, shape (4,))
    input_box = np.array(bbox[0]) if bbox is not None else None

    # --- Call native predict_inst ---
    # First call: multimask_output=True to get the 3 candidates + scores
    # Subsequent calls: pass prev_logits as mask_input + multimask_output=False
    if prev_logits is None:
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,          # get 3 masks, pick best
        )
        # Pick the best mask for the return value
        best_idx = int(np.argmax(scores))
        best_mask_tensor = masks[best_idx]
        best_logits = logits[best_idx]
    else:
        # Feedback loop: pass previous best logits as mask_input
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            mask_input=prev_logits[None],   # shape (1, H_low, W_low)
            multimask_output=False,
        )
        best_mask_tensor = masks[0]
        best_logits = logits[0]

    # Convert to uint8 numpy mask
    img_np = np.array(pil_image)
    if isinstance(best_mask_tensor, torch.Tensor):
        mask = best_mask_tensor.detach().cpu().numpy()
    else:
        mask = best_mask_tensor

    mask = (mask.astype(np.uint8)) * 255
    # Morphological cleanup
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 3:
        return None, None, None

    poly = Polygon(cnt.squeeze()).simplify(2.0, preserve_topology=True)

    if debug:
        overlay = img_np.copy()
        pts = np.array(poly.exterior.coords).astype("int32")
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(f"debug_mask.png", overlay)

    return [mask], [poly], best_logits   # ← now returns logits too
