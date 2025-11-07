#!/usr/bin/env python3
import os
CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ.setdefault("PROJ_LIB",  fr"{CONDA}\Library\share\proj")
os.environ.setdefault("GDAL_DATA", fr"{CONDA}\Library\share\gdal")
try:
    from pyproj import datadir as _pd
    _pd.set_data_dir(os.environ["PROJ_LIB"])
except Exception:
    pass
import os, json, random, requests
import rasterio, geopandas as gpd
import numpy as np, cv2
import rasterio.features
from shapely.geometry import Polygon
from tqdm import tqdm
#!/usr/bin/env python3


print(f" Using PROJ_LIB={os.environ['PROJ_LIB']}")
print(f" Using GDAL_DATA={os.environ['GDAL_DATA']}")

# -------------------------------
# CONFIG
# -------------------------------
GPKG_PATH = r"D:\git\Master-Thesis-GEOAI\outputs\buildings_sam_tiles.gpkg"
GEOTIFF_PATH = "data/ortho_4.tif"       # set correct path
API_URL = "https://9a8sbumc2yel96-7860.proxy.runpod.net/infer_points"
OUT_JSON = "outputs/mllm_results.json"
OUT_PREVIEW_DIR = "outputs/mllm_previews"
N_SAMPLES = 40          # number of polygons to evaluate
PAD_FACTOR = 0.25      # dynamic padding fraction
MIN_PAD_PX = 64        # minimum padding in pixels

# -------------------------------
# Helper: crop around polygon with dynamic padding
# -------------------------------
def crop_for_polygon(src, geom, pad_px=0, *args, **kwargs):
    """Crop raster window around Polygon or MultiPolygon with optional padding."""
    from shapely.geometry import Polygon, MultiPolygon, mapping
    import numpy as np
    import rasterio
    import rasterio.features  # explicit import

    # Handle MultiPolygons: pick largest sub-polygon
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)

    mask = rasterio.features.geometry_mask(
        [mapping(geom)], transform=src.transform,
        invert=True, out_shape=(src.height, src.width)
    )

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, []

    x1, x2 = max(0, xs.min() - pad_px), min(src.width, xs.max() + pad_px)
    y1, y2 = max(0, ys.min() - pad_px), min(src.height, ys.max() + pad_px)

    rgb = np.transpose(src.read([1, 2, 3], window=((y1, y2), (x1, x2))), (1, 2, 0))

    # ✅ NumPy 2.0 safe normalization
    ptp_val = np.ptp(rgb) if np.ptp(rgb) != 0 else 1e-6
    rgb = np.clip((rgb - np.min(rgb)) / ptp_val * 255, 0, 255).astype(np.uint8)

    # Local polygon coordinates relative to crop window
    poly_xy = [(int(x - x1), int(y - y1)) for x, y in np.array(geom.exterior.coords)]
    return rgb, poly_xy





# -------------------------------
# MAIN
# -------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(OUT_PREVIEW_DIR, exist_ok=True)

    gdf = gpd.read_file(GPKG_PATH)
    if len(gdf) == 0:
        print("❌ No polygons found in GPKG.")
        return

    polys = random.sample(list(gdf.geometry), min(N_SAMPLES, len(gdf)))
    results = []

    with rasterio.open(GEOTIFF_PATH) as src:
        for i, poly in enumerate(tqdm(polys, desc="MLLM inference")):
            try:
                crop, poly_xy = crop_for_polygon(src, poly, PAD_FACTOR, MIN_PAD_PX)
            except Exception as e:
                print(f"⚠️ Skipping polygon {i}: {e}")
                continue
            if crop is None:
                continue

            # Prepare and send request
            _, tmp_png = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            files = {"image": (f"sample_{i}.png", tmp_png.tobytes(), "image/png")}
            data = {"poly_json": json.dumps({"poly_xy": poly_xy})}

            try:
                r = requests.post(API_URL, files=files, data=data, timeout=300)
                r.raise_for_status()
                resp = r.json()
                results.append(resp)
            except Exception as e:
                print(f"⚠️ Request {i} failed:", e)
                continue

            # Draw returned points for quick visual QA
            vis = crop.copy()
            if isinstance(resp, dict):
                inside = resp.get("inside", [])
                outside = resp.get("outside", [])
                for (x, y) in inside:
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                for (x, y) in outside:
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)

            cv2.polylines(vis, [np.array(poly_xy, np.int32)], True, (255, 255, 0), 2)
            cv2.imwrite(os.path.join(OUT_PREVIEW_DIR, f"sample_{i}_preview.png"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Save summary JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Finished. {len(results)} polygons processed.")
    print(f"📁 JSON: {OUT_JSON}")
    print(f"🖼️  Previews: {OUT_PREVIEW_DIR}/sample_*.png")


if __name__ == "__main__":
    main()
