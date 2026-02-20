import cv2
import numpy as np

from src.patches.extractor import extract_patch
from src.sam.model_ import segment_with_points
from src.utils.geometry import polygon_to_sam_bbox

def _touches_border(mask, margin=2):
    h, w = mask.shape[:2]

    # check 4 borders
    if np.any(mask[:margin, :] > 0):
        return True
    if np.any(mask[h - margin:h, :] > 0):
        return True
    if np.any(mask[:, :margin] > 0):
        return True
    if np.any(mask[:, w - margin:w] > 0):
        return True

    return False

def run_sam_stage(
    img,
    raw_path,
    poly_px,
    inside,
    outside,
    out_dir,
    bid,
    geom=None,
    crs=None,
    tiff_path=None,
    context=2.0,
    max_context=8.0,
    max_iters=3,
):
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
    bbox_scale = 0.35

    while True:

        # initial bbox from footprint
        bbox_init = polygon_to_sam_bbox(poly_px, scale=bbox_scale)
        bbox = bbox_init.copy() if bbox_init else None

        inside = original_inside.copy()
        outside = original_outside.copy()

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
                morph_kernel=5,
                debug=False,
            )

            if not polys:
                print("SAM failed")
                break

            print(f"Found {len(polys)} polygons")

            # TEMP: choose first mask (your original logic)
            if len(masks) > 0 and len(polys) > 0:
                mask = masks[0]
                poly = polys[0]
            else:
                break

            # BORDER CHECK (moved here so mask exists)
            if _touches_border(mask):

                if geom is None:
                    print("Border touched but no geom provided.")
                    return None

                if context >= max_context:
                    print("Reached max context. Stopping resize.")
                    break

                print(f"⚠️ Border hit → increasing context {context} → {context * 1.5}")

                context *= 1.5

                # Re-extract bigger patch
                img, poly_px = extract_patch(
                    geom,
                    crs,
                    tiff_path,
                    context=context,
                )

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                raw_path = out_dir / f"bld_{bid:07d}_resized.png"
                cv2.imwrite(str(raw_path), img)

                continue  # restart outer while loop

            # -------------------------------------------------
            # DEBUG: plot all polygons
            # -------------------------------------------------
            overlay_all = img.copy()

            for idx, poly_candidate in enumerate(polys):
                pts = np.array(poly_candidate.exterior.coords).astype("int32")
                color = (0, 255 - idx * 80, idx * 80)
                cv2.polylines(overlay_all, [pts], True, color, 2)

            cv2.imwrite(
                str(out_dir / f"bld_{bid:07d}_all_masks_iter{iter_idx + 1}.png"),
                overlay_all
            )

            # tighten bbox
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

        break  # exit while if no border retry

    if mask is None:
        return None

    print("SAM converged")

    sam_input = img.copy()

    for x, y in original_inside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 255, 0), -1)

    for x, y in original_outside:
        cv2.circle(sam_input, (int(x), int(y)), 6, (0, 0, 255), -1)

    if bbox_init is not None:
        x1, y1, x2, y2 = bbox_init[0]
        cv2.rectangle(sam_input, (x1, y1), (x2, y2), (255, 0, 0), 2)

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




