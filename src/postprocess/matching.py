def filter_new_buildings(candidates, existing_polys, iou_threshold=0.3):

    new = []

    for cand in candidates:

        is_match = False

        for existing in existing_polys:

            if not cand.intersects(existing):
                continue

            intersection = cand.intersection(existing).area
            union = cand.union(existing).area

            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                is_match = True
                break

        if not is_match:
            new.append(cand)

    return new