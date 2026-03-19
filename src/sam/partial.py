# src/sam/detect_all_sam3.py
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

from src.sam.model_sam3 import _MODEL_SEMANTIC, _mask_to_polygon

# ------------------------------------------------------------------
# House concept prompt — swap to ["roof"] or ["building rooftop"]
# for aerial/nadir imagery if needed
# ------------------------------------------------------------------
HOUSE_PROMPTS = ["roof"]


def run_sam_detect_all(
    img: np.ndarray,
    out_dir: Path,
    bid: int,
    text_prompts=None,
    morph_kernel: int = 15,
    min_poly_area: int = 1500,
) -> list:
    """
    Drop-in replacement for the SAM1-based run_sam_detect_all.
    Uses SAM3 text-concept segmentation instead of automatic mask generation.

    Args:
        img:           BGR uint8 image array (same as before)
        out_dir:       directory to write debug masks + overlay
        bid:           building ID for filename formatting
        text_prompts:  SAM3 concept strings, defaults to ["house"]
        morph_kernel:  morphological clean-up kernel size
        min_poly_area: minimum contour area to keep (pixels²)

    Returns:
        List of shapely Polygons for detected house/roof regions
    """

    if text_prompts is None:
        text_prompts = HOUSE_PROMPTS

    # SAM3 expects RGB — convert from BGR input (same as SAM1 path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)

    # --- Run SAM3 semantic segmentation ---
    _MODEL_SEMANTIC.set_image(image_rgb)
    results = _MODEL_SEMANTIC(text=text_prompts)

    roof_polys = []
    overlay = img.copy()

    if not results or results[0].masks is None:
        print(f"SAM3: no masks returned for bid={bid}")
        cv2.imwrite(str(out_dir / f"bld_{bid:07d}_detect_all_overlay.png"), overlay)
        return roof_polys

    raw_masks = results[0].masks.data.cpu().numpy()  # (N, H, W) float

    for i, m in enumerate(raw_masks):

        # _mask_to_polygon handles morph cleanup + contour extraction
        clean_mask, poly = _mask_to_polygon(m, morph_kernel)

        if clean_mask is None or poly is None:
            continue

        # Save every raw mask for debugging (mirrors SAM1 behavior)
        #cv2.imwrite(
           # str(out_dir / f"bld_{bid:07d}_mask_{i:03d}.png"),
           # clean_mask,
        #)

        # Area filter on the polygon itself
        if poly.area < min_poly_area:
            continue

        # Validity guard
        if not poly.is_valid or len(poly.exterior.coords) < 3:
            continue

        roof_polys.append(poly)

        # Draw on overlay
        pts = np.array(poly.exterior.coords).astype("int32")
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

    cv2.imwrite(
        str(out_dir / f"bld_{bid:07d}_detect_all_overlay.png"),
        overlay,
    )
    print(f"SAM3 '{text_prompts}' → {len(raw_masks)} masks, {len(roof_polys)} kept for bid={bid}")

    return roof_polys
