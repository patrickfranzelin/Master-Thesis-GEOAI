import numpy as np
from shapely import LineString

from postprocess.occlusion.geometry import to_lines, normalize


def split_dents(edge, cfg):
    """
    Split edge into smaller 'dent candidates'
    based on direction + curvature changes
    """

    lines = to_lines(edge)

    dents = []

    for line in lines:
        coords = list(line.coords)

        if len(coords) < 3:
            dents.append(line)
            continue

        current = [coords[0]]

        for i in range(1, len(coords) - 1):
            p_prev = np.array(coords[i-1])
            p_curr = np.array(coords[i])
            p_next = np.array(coords[i+1])

            v1 = normalize(p_curr - p_prev)
            v2 = normalize(p_next - p_curr)

            angle = abs(np.dot(v1, v2))

            # split if direction changes strongly
            if angle < 0.8:   # threshold tunable
                current.append(coords[i])
                dents.append(LineString(current))
                current = [coords[i]]
            else:
                current.append(coords[i])

        current.append(coords[-1])
        dents.append(LineString(current))

    return [d for d in dents if d.length > cfg.min_edge_length]