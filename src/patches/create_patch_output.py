import cv2
from pathlib import Path
from src.utils.rendering import add_polygon_overlay, add_center_star, add_grid_overlay


def create_patch_outputs(img, poly_px, out_dirs, bid):

    raw_path = out_dirs["raw"] / f"bld_{bid:07d}_raw.png"
    clean_path = out_dirs["clean"] / f"bld_{bid:07d}_clean.png"
    debug_path = out_dirs["debug"] / f"bld_{bid:07d}_debug.png"

    cv2.imwrite(str(raw_path), img)

    # clean = image + polygon
    clean = add_polygon_overlay(img.copy(), poly_px)
    cv2.imwrite(str(clean_path), clean)

    # debug = ONLY center + grid
    debug = img.copy()
    debug = add_center_star(debug)
    #debug = add_grid_overlay(debug)
    cv2.imwrite(str(debug_path), debug)

    return raw_path, clean_path, debug_path

def create_comparison_patch(
    img,
    start_poly_px,
    refined_poly_px,
    out_dirs: dict,
    bid: int,
) -> Path:
    """
    Render a side-by-side comparison patch:
      LEFT  = img + start_poly_px in RED   (labelled ORIGINAL)
      RIGHT = img + refined_poly_px in GREEN (labelled REFINED)

    Uses add_polygon_overlay for consistent rendering with the rest of the pipeline.
    Saves to out_dirs["comparison"] and returns the path.
    """
    import numpy as np

    DIVIDER_WIDTH = 6
    FONT          = cv2.FONT_HERSHEY_SIMPLEX

    # --- Left panel: original polygon in RED ---
    left = add_polygon_overlay(img.copy(), start_poly_px, color=(0, 0, 255))
    cv2.putText(left, "ORIGINAL", (10, 28), FONT, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Right panel: refined polygon in GREEN ---
    right = add_polygon_overlay(img.copy(), refined_poly_px, color=(0, 200, 0))
    cv2.putText(right, "REFINED", (10, 28), FONT, 0.85, (0, 200, 0), 2, cv2.LINE_AA)

    # --- Stitch together with a light-gray divider ---
    h = left.shape[0]
    divider = np.ones((h, DIVIDER_WIDTH, 3), dtype=np.uint8) * 220
    comparison = np.hstack([left, divider, right])

    comp_path = out_dirs["comparison"] / f"bld_{bid:07d}_comparison.png"
    cv2.imwrite(str(comp_path), comparison)
    return comp_path