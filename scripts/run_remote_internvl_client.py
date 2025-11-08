#!/usr/bin/env python3
"""
Run MLLM verification on randomly sampled SAM polygons with a numbered grid overlay.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # <– Must be first

from src.utils.io import save_points_to_gpkg

CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ.setdefault("PROJ_LIB",  fr"{CONDA}\Library\share\proj")
os.environ.setdefault("GDAL_DATA", fr"{CONDA}\Library\share\gdal")
try:
    from pyproj import datadir as _pd
    _pd.set_data_dir(os.environ["PROJ_LIB"])
except Exception:
    pass



import os, json, random, requests
import numpy as np, cv2, geopandas as gpd, rasterio
from tqdm import tqdm
from src.viz.annotate import overlay_numbered_grid, is_black_or_empty, plot_mllm_points
from src.geo.tiler import crop_for_polygon, local_to_global_points, sample_polygons
from src.mllm.prompts import grid_points_prompt


# ==============================================================
# CONFIG
# ==============================================================
GPKG_PATH = "outputs/buildings_sam_tiles.gpkg"
GEOTIFF_PATH = "data/ortho_4.tif"
API_URL = "https://9a8sbumc2yel96-7860.proxy.runpod.net/infer_points"

OUT_JSON = "outputs/mllm_results.json"
OUT_PREVIEW_DIR = "outputs/mllm_previews"
N_SAMPLES = 20
PAD_PX = 64
GRID_SIZE = 25
RANDOM_SEED = 42
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (255, 255, 255)
THICKNESS = 1
# ==============================================================


def main():
    os.makedirs(OUT_PREVIEW_DIR, exist_ok=True)
    gdf = gpd.read_file(GPKG_PATH)
    if gdf.empty:
        print("❌ No polygons found in GPKG.")
        return

    polys = sample_polygons(gdf, N_SAMPLES, RANDOM_SEED)
    results = []
    out_gpkg = "outputs/mllm_points_global.gpkg"

    with rasterio.open(GEOTIFF_PATH) as src:
        for i, poly in enumerate(tqdm(polys, desc="Grid verification", ncols=90)):
            try:
                crop, poly_xy, (c0, r0) = crop_for_polygon(src, poly, pad_factor=0.3, min_pad_px=PAD_PX)
            except Exception as e:
                print(f"⚠️ Crop failed for polygon {i}: {e}")
                continue

            if is_black_or_empty(crop):
                continue

            # Grid overlay + inference
            # Save raw crop that MLLM receives
            crop_path = os.path.join(OUT_PREVIEW_DIR, f"sample_{i:02d}_crop.png")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            # Grid overlay (the one actually sent to MLLM)
            # Grid overlay + polygon drawing
            vis = overlay_numbered_grid(crop, GRID_SIZE)
            cv2.polylines(vis, [np.array(poly_xy, np.int32)], True, (255, 0, 0), 2)

            # Save what MLLM actually sees
            ml_input_path = os.path.join(OUT_PREVIEW_DIR, f"sample_{i:02d}_ml_input.png")
            cv2.imwrite(ml_input_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # Prepare for sending
            prompt = grid_points_prompt()
            _, tmp_png = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            files = {"image": (f"sample_{i}.png", tmp_png.tobytes(), "image/png")}
            data = {"poly_json": json.dumps({"poly_xy": poly_xy, "prompt": prompt})}

            try:
                r = requests.post(API_URL, files=files, data=data, timeout=300)
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                print(f"⚠️ Request {i} failed: {e}")
                continue

            inside = resp.get("inside", [])
            outside = resp.get("outside", [])
            vis = plot_mllm_points(vis, inside, outside, poly_xy)

            out_img = os.path.join(OUT_PREVIEW_DIR, f"sample_{i:02d}_preview.png")
            cv2.imwrite(out_img, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # Optional: save global CRS points
            if inside and outside:
                inside_global = local_to_global_points(inside, src.transform, c0, r0)
                outside_global = local_to_global_points(outside, src.transform, c0, r0)
                save_points_to_gpkg(out_gpkg, inside_global, outside_global, i, src.crs)
            # Optional: store per-polygon metadata for analysis
            results.append({
                "id": i,
                "area": float(poly.area),
                "valid": True,
                "inside_count": len(inside),
                "outside_count": len(outside),
                "grid_input": os.path.basename(out_img),
                "response": resp
            })

    # Save combined JSON results
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done. {len(results)} polygons processed.")
    print(f"📁 JSON: {OUT_JSON}")
    print(f"🗺️  Points: {out_gpkg}")
    print(f"🖼️  Previews: {OUT_PREVIEW_DIR}")



if __name__ == "__main__":
    main()
