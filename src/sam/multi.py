import cv2
import numpy as np
from src.sam.model import segment_with_points


def run_sam_multi_building(
    image_path,
    buildings_data,
    negative_pts=None,
    morph_kernel=7,
    bbox_scale=0.6,          # fraction of image size
    refine_once=True,
):
    """
    Stable multi-building SAM segmentation.

    For each building:
        - Compute centroid from inside points
        - Create adaptive bbox
        - Run SAM constrained to bbox
        - Optionally tighten bbox once

    Returns:
        List[(mask, polygon)]
    """

    results = []

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    def clamp(pt):
        x, y = pt
        return [
            int(min(max(0, x), w - 1)),
            int(min(max(0, y), h - 1)),
        ]

    negative_pts = negative_pts or []
    negative_pts = [clamp(p) for p in negative_pts]

    for building in buildings_data:

        inside = building.get("inside_points", [])
        inside = [clamp(p) for p in inside]

        if not inside:
            results.append((None, None))
            continue

        # -------------------------------------------------
        # 1️⃣ Compute centroid of inside points
        # -------------------------------------------------
        cx = int(sum(p[0] for p in inside) / len(inside))
        cy = int(sum(p[1] for p in inside) / len(inside))

        # -------------------------------------------------
        # 2️⃣ Adaptive bounding box
        # -------------------------------------------------
        side = int(bbox_scale * min(h, w))

        x1 = max(cx - side // 2, 0)
        y1 = max(cy - side // 2, 0)
        x2 = min(cx + side // 2, w - 1)
        y2 = min(cy + side // 2, h - 1)

        bbox = [[x1, y1, x2, y2]]

        # -------------------------------------------------
        # 3️⃣ First SAM pass
        # -------------------------------------------------
        mask, poly = segment_with_points(
            image_path=image_path,
            inside_pts=inside,
            outside_pts=negative_pts,
            bbox=bbox,
            morph_kernel=morph_kernel,
        )

        if mask is None or poly is None:
            results.append((mask, poly))
            continue

        # -------------------------------------------------
        # 4️⃣ Optional refinement: tighten bbox from mask
        # -------------------------------------------------
        if refine_once:
            ys, xs = np.where(mask > 0)

            if len(xs) > 0:
                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max())
                y2 = int(ys.max())

                bbox_refined = [[x1, y1, x2, y2]]

                mask, poly = segment_with_points(
                    image_path=image_path,
                    inside_pts=inside,
                    outside_pts=negative_pts,
                    bbox=bbox_refined,
                    morph_kernel=morph_kernel,
                )

        results.append((mask, poly))

    return results
