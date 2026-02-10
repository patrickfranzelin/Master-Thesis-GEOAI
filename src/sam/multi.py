# src/sam/multi.py

from pathlib import Path
from src.sam.engine import segment_with_points


def segment_multiple_buildings(
    image_path: Path,
    buildings_data,
    negative_pts,
    morph_kernel=7,
):
    results = []

    for building in buildings_data:
        inside = building.get("inside_points", [])
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
