from segment_anything import SamAutomaticMaskGenerator
import cv2
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path

# use the same loaded SAM model from model_.py
from src.sam.model_ import sam


mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.85,
    min_mask_region_area=400,
)

def run_sam_detect_all(
    img,
    out_dir,
    bid,
):

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)

    roof_polys = []

    overlay = img.copy()

    for i, m in enumerate(masks):

        mask = (m["segmentation"].astype(np.uint8) * 255)

        # Save every raw mask (for debugging)
        cv2.imwrite(
            str(out_dir / f"bld_{bid:07d}_mask_{i:03d}.png"),
            mask,
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 1500:
                continue

            if cnt.shape[0] < 3:
                continue

            poly = Polygon(cnt.squeeze()).simplify(
                2.0,
                preserve_topology=True
            )

            roof_polys.append(poly)

            # draw immediately
            pts = np.array(poly.exterior.coords).astype("int32")
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

    # Save overlay of selected masks
    cv2.imwrite(
        str(out_dir / f"bld_{bid:07d}_detect_all_overlay.png"),
        overlay,
    )
    print("Total masks from SAM:", len(masks))

    return roof_polys

