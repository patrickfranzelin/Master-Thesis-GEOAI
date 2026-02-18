import cv2
from pathlib import Path
from src.utils.rendering import add_polygon_overlay, add_center_star, add_grid_overlay


def create_patch_outputs(img, poly_px, out_dirs, bid):

    raw_path = out_dirs["raw"] / f"bld_{bid:07d}_raw.png"
    clean_path = out_dirs["clean"] / f"bld_{bid:07d}_clean.png"
    debug_path = out_dirs["debug"] / f"bld_{bid:07d}_debug.png"

    cv2.imwrite(str(raw_path), img)

    # clean = image + polygon
    clean = add_polygon_overlay(img.copy(), poly_px)
    cv2.imwrite(str(clean_path), clean)

    # debug = ONLY center + grid
    debug = img.copy()
    debug = add_center_star(debug)
    #debug = add_grid_overlay(debug)
    cv2.imwrite(str(debug_path), debug)

    return raw_path, clean_path, debug_path

