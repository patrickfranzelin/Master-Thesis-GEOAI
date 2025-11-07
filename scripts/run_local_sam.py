#!/usr/bin/env python3
from __future__ import annotations
import os, cv2, geopandas as gpd, rasterio
from tqdm import tqdm
from src.utils.env import set_cache_env
from src.utils.io import load_yaml, ensure_dir, save_json
from src.sam.sam_segmenter import SamSegmenter
from src.post.filters import PolygonPostProcessor
from src.geo.tiler import crop_for_polygon

def main():
    set_cache_env()
    cfg = load_yaml("configs/pipeline.yaml")

    # --- 1) SAM segmentation
    seg = SamSegmenter(
        ckpt=cfg["sam"]["checkpoint"],
        generator_kwargs=cfg["sam"]["generator"],
        tile_size=cfg["sam"]["tile_size"],
        overlap=cfg["sam"]["overlap"],
        upscale=cfg["sam"]["upscale"],
    )
    seg.load(device="auto")
    polys, meta = seg.segment_building_candidates(cfg["data"]["geotiff"])
    crs = meta["meta"]["crs"]

    # --- 2) Post-processing
    post = PolygonPostProcessor(cfg["post"], src_crs=crs)
    polys = post.merge_close(post.filter_and_clean(polys, meta["attrs"]))

    # --- 3) Save polygons and crops
    gdf = gpd.GeoDataFrame({}, geometry=polys, crs=crs)
    ensure_dir(cfg["data"]["out_gpkg"])
    gdf.to_file(cfg["data"]["out_gpkg"], layer=cfg["data"]["out_layer"], driver="GPKG")

    ensure_dir(cfg["data"]["out_crops_dir"])
    with rasterio.open(cfg["data"]["geotiff"]) as src:
        sample = polys[:min(20, len(polys))]
        for i, poly in enumerate(tqdm(sample, desc="Crops")):
            try:
                rgb, poly_xy = crop_for_polygon(src, poly, pad_px=16)
                png_path = os.path.join(cfg["data"]["out_crops_dir"], f"crop_{i:03d}.png")
                cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                save_json({"poly_xy": poly_xy}, png_path.replace(".png", ".json"))
            except Exception as e:
                print(f"⚠️ Skipped crop {i}: {e}")

    print(f" Saved polygons → {cfg['data']['out_gpkg']}")
    print(f" Saved crops → {cfg['data']['out_crops_dir']}")

if __name__ == "__main__":
    main()
