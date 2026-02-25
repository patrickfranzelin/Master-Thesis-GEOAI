import cv2
import numpy as np
from src.sam.model import segment_with_points
from src.utils.geometry import polygon_to_sam_bbox


def touches_border(poly, img_shape, margin=3):
    """Return True if any vertex of *poly* lies within *margin* pixels of the image border."""
    h, w = img_shape[:2]
    for x, y in poly.exterior.coords:
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
            return True
    return False


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
    original_inside = inside.copy()
    original_outside = outside.copy()
    bbox_scale = 0.25

    # initial bbox from footprint
    bbox_init = polygon_to_sam_bbox(
        poly_px,
        img.shape,
        scale=bbox_scale
    )
    bbox = bbox_init.copy() if bbox_init else None

    # always add bbox center as positive anchor
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        inside = inside + [[cx, cy]]

    mask = None
    poly = None

    for iter_idx in range(max_iters):

        print(f"SAM iteration {iter_idx + 1}")

        masks, polys = segment_with_points(
            image_path=raw_path,
            inside_pts=inside,
            outside_pts=outside,
            bbox=bbox,
            morph_kernel=8,
            debug=False,
        )

        if not polys:
            print("SAM failed")
            break

        print(f"Found {len(polys)} polygons")

        # -------------------------------------------------
        # 🔎 DEBUG: plot all polygons
        # -------------------------------------------------
        # -------------------------------------------
        # Select polygon closest to original anchor
        # -------------------------------------------

        anchor = poly_px.centroid

        def score(p):
            return p.centroid.distance(anchor)

        best_idx = min(range(len(polys)), key=lambda i: score(polys[i]))

        mask = masks[best_idx]
        poly = polys[best_idx]

        # -------------------------------------------
        # DEBUG: visualize ONLY selected polygon
        # -------------------------------------------

        overlay_selected = img.copy()
        pts = np.array(poly.exterior.coords).astype("int32")

        # Thick bright green outline
        cv2.polylines(overlay_selected, [pts], True, (0, 255, 0), 4)

        # Optional: fill transparent highlight
        mask_color = overlay_selected.copy()
        cv2.fillPoly(mask_color, [pts], (0, 255, 0))
        overlay_selected = cv2.addWeighted(mask_color, 0.3, overlay_selected, 0.7, 0)

        cv2.imwrite(
            str(out_dir / f"bld_{bid:07d}_selected_iter{iter_idx + 1}.png"),
            overlay_selected

        )

        if not polys:
            break

        # -------------------------------------------
        # Select polygon closest to original anchor
        # -------------------------------------------

        anchor = poly_px.centroid

        def score(p):
            return p.centroid.distance(anchor)

        best_idx = min(range(len(polys)), key=lambda i: score(polys[i]))

        mask = masks[best_idx]
        poly = polys[best_idx]

        # -------------------------------------------------
        # tighten bbox from selected mask
        # -------------------------------------------------
        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            bbox = [[
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]]

            cx = int((bbox[0][0] + bbox[0][2]) / 2)
            cy = int((bbox[0][1] + bbox[0][3]) / 2)
            inside.append([cx, cy])

    if mask is None:
        return None

    # Check if the final selected polygon touches the image border
    if touches_border(poly, img.shape):
        print("⚠ Selected polygon touches border → expand patch")
        return "EXPAND_PATCH"

    print("SAM converged")

    # ---------------------------------------------
    # Debug visualization
    # ---------------------------------------------

    sam_input = img.copy()

    # Draw ONLY original MLLM inside points (green)
    for x, y in original_inside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 255, 0), -1)

    # Draw ONLY original MLLM outside points (red)
    for x, y in original_outside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 0, 255), -1)

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