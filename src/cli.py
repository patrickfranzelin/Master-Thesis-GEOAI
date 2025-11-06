#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json
import rasterio, geopandas as gpd, numpy as np, cv2
from tqdm import tqdm

from src.utils.env import set_cache_env
from src.utils.io import load_yaml, ensure_dir, save_json
from src.sam.sam_segmenter import SamSegmenter
from src.post.filters import PolygonPostProcessor
from src.mllm.internvl_client import InternVL3Points
from src.viz.annotate import draw_points


def main():
    set_cache_env()
    cfg = load_yaml(os.environ.get("PIPELINE_CFG", "configs/pipeline.yaml"))

    # 1) SAM -> raw polygons
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

    # 2) Post-process (filter, clean, merge)
    post = PolygonPostProcessor(cfg["post"], src_crs=crs)
    polys = post.filter_and_clean(polys, meta["attrs"])
    polys = post.merge_close(polys)

    # 3) Save polygons to GPKG
    gdf = gpd.GeoDataFrame({}, geometry=polys, crs=crs)
    ensure_dir(cfg["data"]["out_gpkg"])
    gdf.to_file(cfg["data"]["out_gpkg"], layer=cfg["data"]["out_layer"], driver="GPKG")

    # 4) MLLM evaluation on a sample (or all)
    mllm = InternVL3Points(cfg["mllm"]["model_id"], device=cfg["mllm"]["device"], max_new_tokens=cfg["mllm"]["max_new_tokens"])

    results = []
    with rasterio.open(cfg["data"]["geotiff"]) as src:
        # pick N polygons or iterate all (careful: time!)
        sample = polys[: min(20, len(polys))]  # adjust as needed
        for p in tqdm(sample, desc="MLLM points"):
            rgb, poly_xy = crop_for_polygon(src, p, pad_px=16)
            res = mllm.infer_points(rgb, poly_xy)
            if res:
                results.append(res)

            # optional preview for the last one
        if results:
            inside = results[-1]["inside"]; outside = results[-1]["outside"]
            preview = draw_points(rgb, inside, outside)
            ensure_dir(cfg["data"]["out_points_png"])
            cv2.imwrite(cfg["data"]["out_points_png"], cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    # 5) Save JSON with all points
    save_json(results, cfg["data"]["out_points_png"].replace(".png", ".json"))

    print(f"Saved polygons to {cfg['data']['out_gpkg']} (layer: {cfg['data']['out_layer']})")
    print(f"Saved points preview to {cfg['data']['out_points_png']} and JSON next to it")

if __name__ == "__main__":
    main()
