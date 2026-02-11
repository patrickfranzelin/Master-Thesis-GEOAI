from src.sam.model import segment_with_points


def run_sam_multi_building(
    image_path,
    buildings_data,
    negative_pts,
    morph_kernel=7,
):
    results = []

    for building in buildings_data:
        inside = building.get("inside_points", [])

        mask, poly = segment_with_points(
            image_path=image_path,
            inside_pts=inside,
            outside_pts=negative_pts,
            morph_kernel=morph_kernel,
        )

        results.append((mask, poly))

    return results
