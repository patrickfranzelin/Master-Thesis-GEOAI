# --- PROJ/GDAL Pfade zuerst (vor allen geo-Imports!) ---
import os
CONDA = os.environ.get("CONDA_PREFIX", r"C:\Users\franz\miniconda3\envs\geoai")
os.environ.setdefault("PROJ_LIB",  fr"{CONDA}\Library\share\proj")
os.environ.setdefault("GDAL_DATA", fr"{CONDA}\Library\share\gdal")
try:
    from pyproj import datadir as _pd
    _pd.set_data_dir(os.environ["PROJ_LIB"])
except Exception:
    pass

import warnings
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.ops import transform as shp_transform
import geopandas as gpd
from pyproj import Transformer, CRS
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import torch

# =================== USER PARAMS ===================
GEOTIFF_PATH = r"C:\git\Master-Thesis-GEOAI\data\ortho_4.tif"
SAM_CKPT     = r"C:\git\Master-Thesis-GEOAI\sam_vit_b.pth"
OUT_GPKG     = r"C:\git\Master-Thesis-GEOAI\outputs\buildings_sam_tiles.gpkg"
LAYER_NAME   = "buildings_sam"

# Tile-Setup
TILE_SIZE        = 512        # kleinere Patches
OVERLAP          = 32
POINTS_PER_SIDE  = 8         # dichteres Sampling
SAM_UPSCALE      = 1.5        # 1.0 = aus; 1.5–2.0 hilft oft stark


# SAM-Qualität/Speed
SAM_ARGS = dict(
    points_per_side=POINTS_PER_SIDE,
    points_per_batch=8,          # weniger VRAM; 16 wenn viel VRAM
    crop_n_layers=0,
    crop_overlap_ratio=0.0,
    pred_iou_thresh=0.85,        # etwas lockerer
    stability_score_thresh=0.90, # etwas lockerer
    box_nms_thresh=0.6,
    min_mask_region_area=0
)


# Gebäude-Filter
MIN_AREA_M2        = 8.0
MAX_AREA_M2        = 20000.0
SIMPLIFY_TOL_M     = 0.25     # leichte Glättung
# Gebäude-Score-Parameter
MIN_PIX_SIDE       = 4        # min. 4x4 Pixel
ASPECT_MAX         = 6.0      # sehr lang/schmal -> Straße/Zaun
BUILDING_SCORE_MIN = 0.45   # mehr Recall
MERGE_GAP_M        = 1.00   # zerhackte Dächer besser mergen
# leichte Lochschließung/Glättung in Meter
CLEAN_BUFFER_M     = 0.30     # +buffer
CLEAN_DEBUFFER_M   = 0.30     # -buffer

# ===================================================

def to_rgb_uint8(tile_bxhxw):
    arr = tile_bxhxw
    if arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        while arr.shape[0] < 3:
            arr = np.vstack([arr, arr[-1:]])
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [2, 98])
        scale = max(hi - lo, 1e-6)
        arr = np.clip((arr - lo) / scale, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return arr

def apply_alpha_mask(rgb, alpha):
    if alpha is None: return rgb
    if alpha.ndim == 2:
        mask = alpha == 0
        if mask.any():
            rgb = rgb.copy()
            rgb[mask] = 0
    return rgb

def imgcoord_to_world(x, y, transform):
    X = transform.c + x*transform.a + y*transform.b
    Y = transform.f + x*transform.d + y*transform.e
    return X, Y

def polygon_from_contour(contour, transform):
    coords = contour.squeeze(1)
    if len(coords) < 3:
        return None
    ring = [imgcoord_to_world(float(x), float(y), transform) for (x, y) in coords]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (poly and poly.is_valid and not poly.is_empty) else None

def is_meter_crs(crs_obj: CRS) -> bool:
    try:
        return crs_obj.is_projected and any("metre" in ai.unit_name.lower() for ai in crs_obj.axis_info)
    except Exception:
        return False

def project_to_m(geom, src_crs):
    if src_crs is None:
        raise ValueError("Kein CRS vorhanden.")
    src = CRS.from_user_input(src_crs)
    if is_meter_crs(src):
        return geom, "src_is_m"
    dst = CRS.from_epsg(3857)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    g2 = shp_transform(lambda x, y: tr.transform(x, y), geom)
    return g2, "used_3857"

def rectang(contour):
    c_int = np.round(contour).astype(np.int32)
    area_px = float(cv2.contourArea(c_int))
    w, h = cv2.minAreaRect(c_int)[1]

    rect_area = float(w*h) if (w>0 and h>0) else max(area_px, 1e-6)
    return float(area_px / rect_area)

def solid(contour):
    c_int = np.round(contour).astype(np.int32)
    area_px = float(cv2.contourArea(c_int))
    hull = cv2.convexHull(c_int)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    return float(area_px / hull_area)

def compute_features(rgb, contour):
    # für OpenCV-Zeichnen/BBox immer int32 verwenden
    c_int = np.round(contour).astype(np.int32)

    r = rectang(c_int)
    s = solid(c_int)
    x, y, w, h = cv2.boundingRect(c_int)
    aspect = max(w, h) / max(min(w, h), 1)

    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.drawContours(mask, [c_int], -1, 1, -1)

    px = rgb[mask==1].reshape(-1,3).astype(np.float32)
    if px.size == 0:
        mean = np.array([0,0,0], np.float32)
        std  = np.array([0,0,0], np.float32)
    else:
        mean = px.mean(axis=0); std = px.std(axis=0)
    brightness = float(mean.mean())
    texture = float(std.mean())
    return dict(rectang=r, solidity=s, aspect=aspect, brightness=brightness, texture=texture)

def building_score(feat, area_m2):
    # harte Gates
    if feat["aspect"] > ASPECT_MAX:
        return 0.0
    if area_m2 < 10 or area_m2 > MAX_AREA_M2:
        return 0.0
    # weiche Komponenten
    s_rect = np.clip((feat["rectang"] - 0.45) / (0.90 - 0.45), 0, 1)
    s_solid= np.clip((feat["solidity"] - 0.70) / (0.98 - 0.70), 0, 1)
    s_bright = 1.0 - np.abs((feat["brightness"]-140.0)/140.0) # Dächer ≈ mittelhell
    s_tex    = 1.0 - np.clip((feat["texture"]-45.0)/45.0, 0, 1)
    s_area   = np.clip((np.sqrt(area_m2) - 3.0) / (35.0 - 3.0), 0, 1)
    score = 0.35*s_rect + 0.30*s_solid + 0.15*s_bright + 0.10*s_tex + 0.10*s_area
    return float(np.clip(score, 0, 1))

def clean_polygon(poly_m):
    # Löcher schließen + Glättung (in Meter-CRS)
    try:
        if CLEAN_BUFFER_M > 0:
            poly_m = poly_m.buffer(CLEAN_BUFFER_M)
        if CLEAN_DEBUFFER_M > 0:
            poly_m = poly_m.buffer(-CLEAN_DEBUFFER_M)
        if not poly_m.is_valid:
            poly_m = poly_m.buffer(0)
    except Exception:
        pass
    return poly_m

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(device=device)
    mask_gen = SamAutomaticMaskGenerator(model=sam, **SAM_ARGS)

    geoms, attrs = [], []

    with rasterio.Env():
        with rasterio.open(GEOTIFF_PATH) as src:
            if src.crs is None or src.transform is None:
                print("⚠️  TIF ohne CRS/Transform. Vektoren wären in Pixelcoords.")
            crs = src.crs
            width, height = src.width, src.height

            # GSD-aware min Fläche: ≥ 4x4 Pixel
            px_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0 or 0.1
            min_area_by_px = (MIN_PIX_SIDE*MIN_PIX_SIDE) * (px_m*px_m)
            min_area_m2 = max(MIN_AREA_M2, min_area_by_px)

            tile_w = min(TILE_SIZE, width)
            tile_h = min(TILE_SIZE, height)
            y_step = max(1, tile_h - OVERLAP)
            x_step = max(1, tile_w - OVERLAP)

            for top in tqdm(range(0, height, y_step), desc="Tiles (rows)"):
                for left in range(0, width, x_step):
                    w = min(tile_w, width  - left)
                    h = min(tile_h, height - top)
                    if w <= 0 or h <= 0: continue

                    window = Window(left, top, w, h)
                    transform = src.window_transform(window)
                    tile = src.read(out_dtype=np.float32, window=window, resampling=Resampling.bilinear)

                    # Alpha mask optional
                    alpha = None
                    try:
                        if src.count >= 4:
                            alpha = src.read(4, window=window, out_dtype=np.uint8)
                    except Exception:
                        pass

                    rgb = to_rgb_uint8(tile)
                    rgb = apply_alpha_mask(rgb, alpha)

                    # leichte Kontrastverstärkung (hilft bei Dachkanten)
                    try:
                        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l2 = clahe.apply(l)
                        rgb = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
                    except Exception:
                        pass

                    # Upscaling für SAM (mehr Pixel -> feinere Kanten)
                    rgb_for_sam = rgb
                    if SAM_UPSCALE != 1.0:
                        new_w = int(rgb.shape[1] * SAM_UPSCALE)
                        new_h = int(rgb.shape[0] * SAM_UPSCALE)
                        rgb_for_sam = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                    if rgb_for_sam.mean() < 4:
                        continue

                    try:
                        with torch.no_grad():
                            masks = mask_gen.generate(rgb_for_sam)

                    except RuntimeError as e:
                        if "CUDA" in str(e).upper():
                            print("⚠️ CUDA OOM – PPS/TILE_SIZE reduzieren.")
                            continue
                        else:
                            raise

                    core_l = OVERLAP // 2
                    core_t = OVERLAP // 2
                    core_r = w - OVERLAP // 2
                    core_b = h - OVERLAP // 2

                    tile_polys_geo = []
                    tile_polys_m   = []

                    for m in masks:
                        seg = m.get("segmentation")
                        if seg is None: continue
                        binmask = seg.astype(np.uint8)
                        if binmask.sum() == 0: continue

                        cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            if len(c) < 4:
                                continue

                            # Kontur von Upscale zurück auf Original-Pixel
                            if 'SAM_UPSCALE' in globals() and SAM_UPSCALE != 1.0:
                                c = (c.astype(np.float32) / SAM_UPSCALE).astype(np.float32)

                            M = cv2.moments(c)
                            if M["m00"] == 0: continue
                            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                            if not (core_l <= cx < core_r and core_t <= cy < core_b):
                                continue

                            feat = compute_features(rgb, c)
                            if feat["aspect"] > ASPECT_MAX:
                                continue

                            poly_geo = polygon_from_contour(c, transform)
                            if not poly_geo: continue

                            poly_m, mode = project_to_m(poly_geo, crs)
                            area_m2 = float(poly_m.area)
                            if area_m2 < min_area_m2 or area_m2 > MAX_AREA_M2:
                                continue

                            # Gebäude-Score
                            score = building_score(feat, area_m2)
                            if score < BUILDING_SCORE_MIN:
                                continue

                            # polygon cleanup in Meter-CRS
                            poly_m = clean_polygon(poly_m)

                            # zurück ins Quell-CRS, falls nötig
                            if mode == "used_3857":
                                back = Transformer.from_crs(3857, CRS.from_user_input(crs), always_xy=True)
                                poly_geo = shp_transform(lambda x, y: back.transform(x, y), poly_m)
                            else:
                                poly_geo = poly_m

                            if not poly_geo.is_valid or poly_geo.is_empty:
                                continue

                            tile_polys_geo.append(poly_geo)
                            tile_polys_m.append(poly_m)

                            attrs.append({
                                "pred_iou": float(m.get("predicted_iou", np.nan)),
                                "stability": float(m.get("stability_score", np.nan)),
                                "rectang": float(feat["rectang"]),
                                "solidity": float(feat["solidity"]),
                                "aspect": float(feat["aspect"]),
                                "brightness": float(feat["brightness"]),
                                "texture": float(feat["texture"]),
                                "class_score": float(score),
                                "area_m2": float(area_m2)
                            })

                    # Merge nahe beieinanderliegende Fragmente innerhalb des Tiles
                    if tile_polys_m:
                        merged_m = unary_union([p.buffer(MERGE_GAP_M/2) for p in tile_polys_m]).buffer(-MERGE_GAP_M/2)
                        # zurück ins Quell-CRS
                        if isinstance(merged_m, (list, tuple)):
                            merged_geos = merged_m
                        else:
                            merged_geos = [merged_m] if merged_m.geom_type != "GeometryCollection" else list(merged_m.geoms)
                        fixed_geos = []
                        for gm in merged_geos:
                            if gm.is_empty: continue
                            if gm.geom_type == "Polygon":
                                poly_m = clean_polygon(gm)
                                if not poly_m.is_valid or poly_m.is_empty: continue
                                if is_meter_crs(CRS.from_user_input(crs)):
                                    fixed_geos.append(poly_m)
                                else:
                                    back = Transformer.from_crs(3857, CRS.from_user_input(crs), always_xy=True)
                                    fixed_geos.append(shp_transform(lambda x,y: back.transform(x,y), poly_m))
                            elif gm.geom_type == "MultiPolygon":
                                for pp in gm.geoms:
                                    if pp.is_empty: continue
                                    poly_m = clean_polygon(pp)
                                    if not poly_m.is_valid or poly_m.is_empty: continue
                                    if is_meter_crs(CRS.from_user_input(crs)):
                                        fixed_geos.append(poly_m)
                                    else:
                                        back = Transformer.from_crs(3857, CRS.from_user_input(crs), always_xy=True)
                                        fixed_geos.append(shp_transform(lambda x,y: back.transform(x,y), poly_m))
                        # Ersetze tile_polys_geo durch gemergte
                        if fixed_geos:
                            tile_polys_geo = fixed_geos

                    # (Optional) könnte man attrs hier konsolidieren; für Einfachheit lassen wir attrs pro Maske.

                    geoms.extend(tile_polys_geo)

    if not geoms:
        print("ℹ️  Keine Gebäude-Polygone gefunden. Senke BUILDING_SCORE_MIN, lockere Filter oder erhöhe POINTS_PER_SIDE.")
        return

    gdf = gpd.GeoDataFrame(attrs[:len(geoms)], geometry=geoms, crs=crs)  # attrs könnte > geoms sein; trimmen
    gdf["__wkb__"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop_duplicates(subset="__wkb__").drop(columns="__wkb__")

    os.makedirs(os.path.dirname(OUT_GPKG), exist_ok=True)
    if os.path.exists(OUT_GPKG):
        try:
            os.remove(OUT_GPKG)
        except Exception:
            pass

    # optional: nur hohe Scores schreiben
    # gdf = gdf[gdf["class_score"] >= BUILDING_SCORE_MIN]

    gdf.to_file(OUT_GPKG, layer=LAYER_NAME, driver="GPKG")
    print(f"✅ Gespeichert: {OUT_GPKG} (Layer: {LAYER_NAME}) – {len(gdf)} Polygone")

if __name__ == "__main__":
    main()
