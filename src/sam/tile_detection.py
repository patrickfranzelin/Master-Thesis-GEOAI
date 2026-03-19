import cv2
from PIL import Image
from shapely.geometry import Polygon
import time
from src.sam.model_samlocal import _get_model
from src.utils.geometry import mask_to_polygon
from src.utils.drawing import draw_masks_on_tile
from pathlib import Path


def detect_polygons(image, prompt, image_patch_dir, tile_size, overlap):
    """
    Detect polygons in tiled image using text prompt.

    Args:
        image: (H,W,C) numpy RGB image
        prompt: text prompt for SAM (e.g. "building roof")
        image_patch_dir: Path or None for saving visualized tiles
        tile_size: pixels per tile
        overlap: tile overlap pixels

    Returns:
        List of Shapely Polygons in image coordinates
    """
    model, processor = _get_model()

    H, W, _ = image.shape
    step = tile_size - overlap

    print(f"[DETECT] Image: {W}x{H}, tiles: {tile_size}px, overlap: {overlap}px, step: {step}px")

    polys = []
    patch_idx = 0
    total_tiles = 0
    valid_tiles = 0

    for y in range(0, H, step):
        for x in range(0, W, step):
            tile_start = time.time()
            total_tiles += 1
            tile = image[y:y + tile_size, x:x + tile_size]

            if tile.shape[0] < 256 or tile.shape[1] < 256:
                print(f"[DETECT] Skipping tiny tile at ({x},{y}): {tile.shape[:2]}")
                continue

            patch_idx += 1
            valid_tiles += 1

            h, w = tile.shape[:2]
            print(f"[DETECT] Tile #{patch_idx} ({w}x{h}) at ({x},{y})")

            # SAM processing
            pil = Image.fromarray(tile)
            state = processor.set_image(pil)

            output = processor.set_text_prompt(
                state=state,
                prompt=prompt
            )

            masks = output["masks"]  # Always extract masks
            print(f"[DETECT] Tile #{patch_idx}: {len(masks)} masks")

            # Visualize (optional)
            if image_patch_dir is not None:
                vis = draw_masks_on_tile(tile, masks)
                patch_path = Path(image_patch_dir) / f"tile_{patch_idx}.png"
                cv2.imwrite(str(patch_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                print(f"[DETECT] Saved: {patch_path.name}")

            # Extract polygons (always)
            tile_polys = 0
            for i, m in enumerate(masks):
                poly = mask_to_polygon(m)
                if poly is None:
                    print(f"[DETECT] Mask {i + 1}/{len(masks)} -> invalid polygon")
                    continue

                # Shift to image coordinates
                shifted = [(px + x, py + y) for px, py in poly.exterior.coords]
                polys.append(Polygon(shifted))
                tile_polys += 1

            tile_time = time.time() - tile_start
            print(f"[DETECT] Tile #{patch_idx} time: {tile_time:.2f} sec")
            print(f"[DETECT] Tile #{patch_idx}: {tile_polys} valid polygons")

    print(f"[DETECT] Summary:")
    print(f"  Total tiles attempted: {total_tiles}")
    print(f"  Valid tiles processed: {valid_tiles}")
    print(f"  Total polygons found: {len(polys)}")

    return polys
