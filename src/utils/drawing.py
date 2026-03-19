import numpy as np
import torch
import cv2


def draw_masks_on_tile(tile, masks):

    vis = tile.copy()

    for m in masks:

        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()

        m = np.squeeze(m)

        if m.ndim != 2:
            continue

        mask = (m > 0.5).astype(np.uint8)

        color = np.zeros_like(vis)
        color[:,:,0] = mask * 255

        vis = cv2.addWeighted(vis, 1.0, color, 0.4, 0)

    return vis

def draw_polygons(image, polys):

    out = image.copy()

    for poly in polys:

        if poly is None or poly.is_empty:
            continue

        pts = np.array(poly.exterior.coords).astype("int32")

        cv2.polylines(out, [pts], True, (255, 255, 255), 5)
        cv2.polylines(out, [pts], True, (255, 0, 0), 2)

    return out