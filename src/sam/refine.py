import cv2
import numpy as np
from src.sam.model import segment_with_points
from src.utils.geometry import polygon_to_sam_bbox


def run_sam_stage(img, raw_path, poly_px, inside, outside, out_dir, bid, max_iters=3, mode="standard"):
    """
    Run SAM refinement stage.
    
    Args:
        img: Input image
        raw_path: Path to raw image
        poly_px: Polygon in pixel coordinates
        inside: List of positive points
        outside: List of negative points
        out_dir: Output directory
        bid: Building ID
        max_iters: Maximum SAM iterations
        mode: "standard" for full houses, "escalated" for partial houses
    """

    is_escalated = (mode == "escalated")
    
    if is_escalated:
        # For partial houses, reset points and use larger bbox
        print(f"  SAM mode: escalated (partial house) - using larger bbox, resetting MLQA points")
        inside = []
        outside = []
        bbox_scale = 0.8
    else:
        # For full houses, use standard bbox and MLQA points
        print(f"  SAM mode: standard (full house) - using MLQA points")
        bbox_scale = 0.2

    if len(inside) == 0 and not is_escalated:
        print("SAM skipped (no inside points)")
        return None
        
    # initial bbox from footprint
    bbox_init = polygon_to_sam_bbox(poly_px, scale=bbox_scale)
    bbox = bbox_init.copy() if bbox_init else None

    # always add bbox center as positive anchor
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        inside = inside + [[cx, cy]]


    mask = None
    poly = None

    for i in range(max_iters):

        print(f"SAM iteration {i+1}")

        mask, poly = segment_with_points(
            image_path=raw_path,
            inside_pts=inside,
            outside_pts=outside,
            bbox=bbox,
        )

        if mask is None:
            print("SAM failed")
            break

        # -----------------------------------
        # tighten bbox from mask
        # -----------------------------------

        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            bbox = [[
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]]

            # add new center point
            cx = int((bbox[0][0] + bbox[0][2]) / 2)
            cy = int((bbox[0][1] + bbox[0][3]) / 2)
            inside.append([cx, cy])

    if mask is None:
        return None

    print("SAM converged")

    # ---------------------------------------------
    # Debug visualization
    # ---------------------------------------------

    sam_input = img.copy()

    for x, y in inside:
        cv2.circle(sam_input, (int(x), int(y)), 5, (0, 255, 0), -1)

    for x, y in outside:
        cv2.circle(sam_input, (int(x), int(y)), 5, (0, 0, 255), -1)

    # original bbox (BLUE)
    if bbox_init is not None:
        x1, y1, x2, y2 = bbox_init[0]
        cv2.rectangle(sam_input, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # final bbox (YELLOW)
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(sam_input, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(str(out_dir / f"bld_{bid:07d}_sam_input.png"), sam_input)
    cv2.imwrite(str(out_dir / f"bld_{bid:07d}_mask.png"), mask)

    if poly is not None:
        overlay = img.copy()
        pts = np.array(poly.exterior.coords).astype("int32")
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"bld_{bid:07d}_sam.png"), overlay)

    return poly




