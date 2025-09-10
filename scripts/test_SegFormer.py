# --- PROJ/GDAL wie in deinem Skript ---
import os
CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ.setdefault("PROJ_LIB",  fr"{CONDA}\Library\share\proj")
os.environ.setdefault("GDAL_DATA", fr"{CONDA}\Library\share\gdal")

import warnings, torch, cv2, numpy as np, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# =================== USER PARAMS ===================
GEOTIFF_PATH = r"C:\git\Master-Thesis-GEOAI\data\ortho_4.tif"
OUT_GPKG     = r"C:\git\Master-Thesis-GEOAI\outputs\buildings_segformer.gpkg"
LAYER_NAME   = "buildings_segformer"

# HF-Checkpoint (zum Testen ADE20K; austauschbar)
CKPT = "nvidia/segformer-b5-finetuned-ade-640-640"

TILE_SIZE   = 512
OVERLAP     = 64
MIN_AREA_M2 = 8.0
MAX_AREA_M2 = 20000.0
SIMPLIFY_TOL_M = 0.25
MERGE_GAP_M    = 1.0
# ===================================================

def img_to_rgb_uint8(tile):
    arr = tile
    if arr.shape[0] >= 3: arr = arr[:3]
    else:
        while arr.shape[0] < 3: arr = np.vstack([arr, arr[-1:]])
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [2, 98])
        scale = max(hi - lo, 1e-6)
        arr = np.clip((arr - lo) / scale, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return arr

def polygon_from_contour(contour, transform):
    coords = contour.squeeze(1)
    if len(coords) < 3: return None
    X = transform.c + coords[:,0]*transform.a + coords[:,1]*transform.b
    Y = transform.f + coords[:,0]*transform.d + coords[:,1]*transform.e
    poly = Polygon(np.column_stack([X, Y]))
    if not poly.is_valid: poly = poly.buffer(0)
    return poly if (poly and poly.is_valid and not poly.is_empty) else None

def is_meter_crs(crs_obj: CRS) -> bool:
    try:
        return crs_obj.is_projected and any("metre" in ai.unit_name.lower() for ai in crs_obj.axis_info)
    except Exception:
        return False

def to_meter(geom, src_crs):
    src = CRS.from_user_input(src_crs)
    if is_meter_crs(src): return geom, "m"
    dst = CRS.from_epsg(3857)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shp_transform(lambda x,y: tr.transform(x,y), geom), "3857"

def back_from_meter(geom_m, src_crs):
    src = CRS.from_user_input(src_crs)
    if is_meter_crs(src): return geom_m
    tr = Transformer.from_crs(3857, src, always_xy=True)
    return shp_transform(lambda x,y: tr.transform(x,y), geom_m)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = SegformerForSemanticSegmentation.from_pretrained(CKPT).to(device).eval()

    # „building“-ID aus id2label ableiten (robust gg. verschiedene Checkpoints)
    id2label = model.config.id2label
    building_ids = [int(i) for i,l in id2label.items() if "building" in l.lower()]
    if not building_ids:
        print("⚠️ In diesem Checkpoint gibt es kein 'building' Label. Anderen Checkpoint wählen.")
        return
    building_ids = set(building_ids)

    geoms = []
    attrs = []

    with rasterio.Env(), rasterio.open(GEOTIFF_PATH) as src:
        crs = src.crs
        width, height = src.width, src.height
        px_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0 or 0.1
        min_area_by_px = (4*4) * (px_m*px_m)
        min_area_m2 = max(MIN_AREA_M2, min_area_by_px)

        tile_w = min(TILE_SIZE, width)
        tile_h = min(TILE_SIZE, height)
        y_step = max(1, tile_h - OVERLAP)
        x_step = max(1, tile_w - OVERLAP)

        for top in tqdm(range(0, height, y_step), desc="Tiles (rows)"):
            for left in range(0, width, x_step):
                w = min(tile_w, width - left)
                h = min(tile_h, height - top)
                if w<=0 or h<=0: continue

                window = Window(left, top, w, h)
                transform = src.window_transform(window)
                tile = src.read(out_dtype=np.float32, window=window, resampling=Resampling.bilinear)
                rgb = img_to_rgb_uint8(tile)

                # leichte Kontrast-Boost
                try:
                    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                    l,a,b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    l2 = clahe.apply(l)
                    rgb = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2RGB)
                except: pass

                inputs = processor(images=rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    upsampled = torch.nn.functional.interpolate(
                        outputs.logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
                    )[0]
                    pred = upsampled.argmax(dim=0).detach().cpu().numpy().astype(np.int32)

                mask = np.isin(pred, list(building_ids)).astype(np.uint8)*255
                if mask.mean() < 1:  # leer
                    continue

                # Konturen -> Polygone
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys_meter = []
                for c in cnts:
                    if len(c) < 4: continue
                    poly = polygon_from_contour(c, transform)
                    if not poly: continue
                    poly_m, mode = to_meter(poly, crs)
                    area = float(poly_m.area)
                    if area < min_area_m2 or area > MAX_AREA_M2: continue
                    # leichte Cleanups
                    poly_m = poly_m.buffer(0.30).buffer(-0.30)
                    if not poly_m.is_valid or poly_m.is_empty: continue
                    poly_geo = back_from_meter(poly_m, crs)
                    polys_meter.append(poly_geo)

                if not polys_meter: continue
                # Merge nahe Dächer
                meter_crs = CRS.from_user_input(crs)
                if not is_meter_crs(meter_crs):
                    # merge im 3857
                    tmp = [to_meter(p, crs)[0] for p in polys_meter]
                    merged = unary_union([p.buffer(MERGE_GAP_M/2) for p in tmp]).buffer(-MERGE_GAP_M/2)
                    # zurück
                    if merged.geom_type == "Polygon": merged = [merged]
                    back = [back_from_meter(p, crs) for p in (merged.geoms if merged.geom_type=="MultiPolygon" else merged)]
                    geoms.extend(back)
                else:
                    merged = unary_union([p.buffer(MERGE_GAP_M/2) for p in polys_meter]).buffer(-MERGE_GAP_M/2)
                    if merged.geom_type == "Polygon": merged = [merged]
                    geoms.extend(list(merged.geoms if hasattr(merged, "geoms") else [merged]))

                attrs.extend([{}] * (len(geoms) - len(attrs)))

    if not geoms:
        print("ℹ️ Keine Gebäude gefunden – anderes Checkpoint versuchen oder Schwellen lockern.")
        return

    gdf = gpd.GeoDataFrame(attrs[:len(geoms)], geometry=geoms, crs=crs)
    gdf["__wkb__"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop_duplicates(subset="__wkb__").drop(columns="__wkb__")
    os.makedirs(os.path.dirname(OUT_GPKG), exist_ok=True)
    try:
        if os.path.exists(OUT_GPKG): os.remove(OUT_GPKG)
    except: pass
    gdf.to_file(OUT_GPKG, layer=LAYER_NAME, driver="GPKG")
    print(f"✅ Gespeichert: {OUT_GPKG} ({len(gdf)} Polygone)")

if __name__ == "__main__":
    main()
