import cv2
import numpy as np
from src.sam.model_samlocal import segment_with_points
from src.utils.geometry import polygon_to_sam_bbox


def touches_border(poly, img_shape, margin=3):
    """Return True if any vertex of *poly* lies within *margin* pixels of the image border."""
    h, w = img_shape[:2]
    for x, y in poly.exterior.coords:
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
            return True
    return False


def run_sam_stage(img, raw_path, poly_px, inside, outside, out_dir, bid, max_iters=3, mode="standard", init_polygon=None ):
    original_inside = inside.copy()
    original_outside = outside.copy()
    bbox_scale = 0.25

    bbox_init = polygon_to_sam_bbox(poly_px, img.shape, scale=bbox_scale)
    bbox = bbox_init.copy() if bbox_init else None
    prev_logits = None

    if init_polygon is not None:
        # boundary points
        for x, y in init_polygon.exterior.coords[::8]:
            inside.append([int(x), int(y)])

        # centroid
        c = init_polygon.centroid
        inside.append([int(c.x), int(c.y)])

    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        inside = inside + [[cx, cy]]

    mask = None
    poly = None
    #prev_logits = None  # ← tracks logits across iterations

    for iter_idx in range(max_iters):
        print(f"SAM iteration {iter_idx + 1}")

        # ← now receives logits too, and passes prev_logits in
        masks, polys, logits = segment_with_points(
            image_path=raw_path,
            inside_pts=inside,
            outside_pts=outside,
            bbox=bbox,
            morph_kernel=12,
            debug=False,
            prev_logits=prev_logits,   # ← FEEDBACK LOOP
        )

        if not polys:
            print("SAM failed")
            break

        anchor = poly_px.centroid
        best_idx = min(range(len(polys)), key=lambda i: polys[i].centroid.distance(anchor))

        mask = masks[best_idx]
        poly = polys[best_idx]
        prev_logits = logits          # ← store for next iteration

        # Debug: visualize selected polygon
        overlay_selected = img.copy()
        pts = np.array(poly.exterior.coords).astype("int32")
        cv2.polylines(overlay_selected, [pts], True, (0, 255, 0), 4)
        mask_color = overlay_selected.copy()
        cv2.fillPoly(mask_color, [pts], (0, 255, 0))
        overlay_selected = cv2.addWeighted(mask_color, 0.3, overlay_selected, 0.7, 0)
        cv2.imwrite(str(out_dir / f"bld_{bid:07d}_selected_iter{iter_idx + 1}.png"), overlay_selected)

        # Tighten bbox from current mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            bbox = [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]
            #prev_logits = None  # ← reset: bbox changed, fresh start next iter

    if mask is None:
        return None

    if touches_border(poly, img.shape):
        print("Selected polygon touches border → expand patch")
        return "EXPAND_PATCH"

    print("SAM converged")

    # --- Debug visualization (unchanged) ---
    sam_input = img.copy()
    # MLLM inside → green
    for x, y in original_inside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 255, 0), -1)

    # polygon inside → cyan
    for x, y in inside[len(original_inside):]:
        cv2.circle(sam_input, (int(x), int(y)), 6, (255, 255, 0), -1)

    # outside → red
    for x, y in outside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 0, 255), -1)
    if bbox_init is not None:
        x1, y1, x2, y2 = bbox_init[0]
        cv2.rectangle(sam_input, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(sam_input, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"bld_{bid:07d}_sam_input.png"), sam_input)

    if poly is not None:
        overlay = img.copy()
        pts = np.array(poly.exterior.coords).astype("int32")
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"bld_{bid:07d}_sam.png"), overlay)

    return poly
