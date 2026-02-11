import cv2
from src.sam.model import segment_with_points


def run_sam_multi_building(
    image_path,
    buildings_data,
    negative_pts,
    morph_kernel=7,
):
    results = []

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    def clamp(pt):
        x, y = pt
        return [min(max(0, x), w - 1), min(max(0, y), h - 1)]

    # Clamp negative points once
    negative_pts = [clamp(p) for p in negative_pts]

    for building in buildings_data:
        inside = building.get("inside_points", [])

        # Clamp inside points per building
        inside = [clamp(p) for p in inside]

        if not inside:
            results.append((None, None))
            continue

        mask, poly = segment_with_points(
            image_path=image_path,
            inside_pts=inside,
            outside_pts=negative_pts,
            morph_kernel=morph_kernel,
        )

        results.append((mask, poly))

    return results
