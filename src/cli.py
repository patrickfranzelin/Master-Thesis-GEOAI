#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json
import rasterio, geopandas as gpd, numpy as np, cv2
from shapely.geometry import Polygon, box
from tqdm import tqdm
from utils.env import set_cache_env
from utils.io import load_yaml, ensure_dir, save_json
from sam.sam_segmenter import SamSegmenter
from post.filters import PolygonPostProcessor
from mllm.internvl_client import InternVL3Points
from viz.annotate import draw_points

def crop_for_polygon(src, poly: Polygon, pad_px: int = 16):
    # crop around polygon bbox (in pixel space: approximate via bounds -> row/col via inverse affine)
    inv = ~src.transform
    minx, miny, maxx, maxy = poly.bounds
    cmin, rmin = inv * (minx, miny)
    cmax, rmax = inv * (maxx, maxy)
    r0, r1 = int(max(0, np.floor(min(rmin, rmax)) - pad_px)), int(min(src.height, np.ceil(max(rmin, rmax)) + pad_px))
    c0, c1 = int(max(0, np.floor(min(cmin, cmax)) - pad_px)), int(min(src.width, np.ceil(max(cmin, cmax)) + pad_px))
    win = rasterio.windows.Window(c0, r0, max(1, c1-c0), max(1, r1-r0))
    arr = src.read(window=win, out_dtype=np.uint8)
    rgb = np.moveaxis(arr[:3], 0, -1)
    # polygon to local crop coords
    xs = (np.array([p[0] for p in poly.exterior.coords]) - (src.transform.c + c0*src.transform.a)) / src.transform.a
    ys = (np.array([p[1] for p in poly.exterior.coords]) - (src.transform.f + r0*src.transform.e)) / src.transform.e
    poly_xy = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    return rgb, poly_xy

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
