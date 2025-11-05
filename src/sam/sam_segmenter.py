from __future__ import annotations
import numpy as np, cv2
from typing import List, Dict, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union
import rasterio
from . import sam_segmenter  # for relative import sanity in some editors
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ..geo.tiler import to_rgb_uint8, enhance_local_contrast, iter_tiles
from ..geo.projection import to_meters, back_from_meters
from pyproj import CRS
from tqdm import tqdm

def polygon_from_contour(contour, transform) -> Polygon | None:
    coords = contour.squeeze(1)
    if len(coords) < 3: return None
    X = transform.c + coords[:,0]*transform.a + coords[:,1]*transform.b
    Y = transform.f + coords[:,0]*transform.d + coords[:,1]*transform.e
    poly = Polygon(np.column_stack([X, Y]))
    if not poly.is_valid: poly = poly.buffer(0)
    return poly if poly.is_valid and not poly.is_empty else None

class SamSegmenter:
    def __init__(self, ckpt: str, generator_kwargs: dict, tile_size: int, overlap: int, upscale: float):
        self.ckpt = ckpt
        self.gen_kwargs = generator_kwargs
        self.tile_size = tile_size
        self.overlap = overlap
        self.upscale = upscale
        self.sam = None
        self.generator = None

    def load(self, device: str = "auto"):
        import torch
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.sam = sam_model_registry["vit_b"](checkpoint=self.ckpt).to(dev)
        self.generator = SamAutomaticMaskGenerator(self.sam, **self.gen_kwargs)
        self.device = dev

    def segment_building_candidates(self, tif_path: str) -> tuple[list[Polygon], dict]:
        geoms: list[Polygon] = []
        attrs: list[Dict] = []
        with rasterio.Env():
            with rasterio.open(tif_path) as src:
                crs = src.crs
                px_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0 or 0.1

                for _, arr, alpha in tqdm(iter_tiles(src, self.tile_size, self.overlap), desc="Tiles"):
                    rgb = to_rgb_uint8(arr)
                    if alpha is not None:
                        mask = alpha == 0
                        if mask.any(): rgb[mask] = 0

                    rgb = enhance_local_contrast(rgb)
                    rgb_for_sam = rgb
                    if self.upscale and self.upscale != 1.0:
                        new_w = int(rgb.shape[1]*self.upscale)
                        new_h = int(rgb.shape[0]*self.upscale)
                        rgb_for_sam = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                    if rgb_for_sam.mean() < 4:  # empty/black tiles
                        continue

                    masks = self.generator.generate(rgb_for_sam)
                    for m in masks:
                        seg = m.get("segmentation")
                        if seg is None or seg.sum() == 0: continue
                        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            if len(c) < 4: continue
                            if self.upscale and self.upscale != 1.0:
                                c = (c.astype(np.float32) / self.upscale).astype(np.float32)
                            poly = polygon_from_contour(c, src.transform)
                            if poly is None: continue
                            geoms.append(poly)
                            attrs.append({
                                "pred_iou": float(m.get("predicted_iou", np.nan)),
                                "stability": float(m.get("stability_score", np.nan)),
                                "px_m": float(px_m)
                            })
        meta = {"crs": crs.to_string() if isinstance(crs, CRS) else str(crs)}
        return geoms, {"attrs": attrs, "meta": meta}
