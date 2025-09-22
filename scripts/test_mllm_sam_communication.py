# scripts/02_mllm_qa.py
# MVP: QA of SAM/SegFormer building polygons with an MLLM and simple geometry/mask fixes.
# Now with clear progress prints & lightweight error handling.

import os, io, base64, json, warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio import features
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import CRS
from PIL import Image, ImageDraw
import cv2
import random
import torch
from segment_anything import sam_model_registry, SamPredictor
# ---------- USER PARAMS ----------
BASE_DIR     = Path(__file__).resolve().parent.parent
GEOTIFF_PATH = str(BASE_DIR / "data" / "ortho_4.tif")
IN_GPKG      = str(BASE_DIR / "outputs" / "buildings_segformer.gpkg")
IN_LAYER     = "buildings_segformer"
OUT_GPKG     = str(BASE_DIR / "outputs" / "buildings_mllm_corrected.gpkg")
OUT_LAYER    = "buildings_mllm"

MODEL        = "gpt-4o-mini"   # upgrade to gpt-4o for higher quality
CHIP_SIZE_PX = 384             # 256–512 is a good range
PAD_METERS   = 3.0
CONF_DROP    = 0.20
MIN_AREA_M2  = 8.0
LOG_EVERY_N  = 25              # print a heartbeat every N features
SHOW_FIX_JSON= False           # True = print model JSON for each feature
DEBUG_CHIP_DIR = BASE_DIR / "outputs" / "debug_chips"

# ---------------------------------

# ---- OpenAI client ----
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

SYSTEM_PROMPT = """
You are a strict QA inspector for building footprints in orthoimagery.

TASK
- You receive an RGB chip (satellite/ortho) with a red outline showing a candidate footprint.
- Judge the candidate VERY strictly: accept only if the outline matches the visible roof edges precisely
  (no shift, no rounded corners, correct angles, no partial roof).

OUTPUT
- Return ONLY a single valid JSON object (no extra text). Use exactly this schema and no extra keys:

{
  "is_building": true or false,
  "confidence": <float from 0.0 to 1.0>,
  "issues": [
    "shadow_inclusion","vegetation_overlap","road_or_path","open_polygon",
    "partial_roof","shape_inaccurate","not_a_building","too_small","missing_part","other"
  ],
  "suggested_fix": [
    // If is_building=true, ALWAYS include at least:
    {"op":"mask_refine","enabled": true}
    // Additional optional fixes allowed:
    // {"op":"buffer","meters": <float>},
    // {"op":"simplify","tolerance_m": <float>},
    // {"op":"remove_small_components","min_area_m2": <float>},
    // {"op":"erode","meters": <float>}, {"op":"dilate","meters": <float>},
    // {"op":"snap_to_rect"}
  ],
  "positive_points": [[x,y], ...],
  "negative_points": [[x,y], ...]
}

RULES
- Coordinates in positive_points/negative_points are pixel positions (x,y) in the provided chip
  (origin top-left, x to the right, y down). Integers preferred.
- The chip includes a visible white grid with labels (step = 50 px). Use these labels to help place coordinates correctly.
- If and only if is_building = true:
    * Provide at least 4 positive_points placed ON the roof interior.
    * Provide at least 4 negative_points clearly OUTSIDE the building
      (on vegetation, road, bare ground or shadows off the roof). Negative points must NOT be on the roof.
    * Include {"op":"mask_refine","enabled": true} in suggested_fix.
- If is_building = false:
    * Set confidence low (<= 0.3), list relevant issues,
      and set positive_points = [] and negative_points = [] (or omit them).
- Be conservative: if unsure, mark is_building=false.
- Do not invent geometry outside what is visible; do not output values out of the chip bounds.
- Output must be valid JSON only (no comments, no trailing commas, no prose).
"""


def add_grid_overlay(img, step=50):
    """Draw a white grid with labels every <step> pixels on the chip."""
    h, w, _ = img.shape
    overlay = img.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(overlay, str(x), (x+2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(overlay, str(y), (2, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    return overlay


def tiles_to_pixels(tiles, step=50):
    points = []
    for tx, ty in tiles:
        px = tx
        py = ty
        points.append([px, py])
    return np.array(points, dtype=np.float32)

def ask_mllm_for_fixes(img_b64_png: str, out_json_path: Path) -> dict:
    """Send one chip to the MLLM and get JSON fixes back, also save to disk."""
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Inspect the footprint and respond with JSON. "
                                     "If possible, also return 'positive_points' and 'negative_points' "
                                     "as pixel coordinates (x,y) in the image chip."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64_png}}
        ]}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=msg,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(resp.choices[0].message.content)
        # Save raw JSON to file
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"[WARN] Failed to parse model JSON: {e}")
        return {"is_building": True, "confidence": 0.5, "issues": [], "suggested_fix": []}


def run_sam_refine(chip_rgb_uint8, poly_world, chip_transform,
                   sam_ckpt_path: str, model_type: str = "vit_b",
                   mllm_result: dict = None):
    """
    Verfeinert das Polygon mit SAM.
    Nutzt MLLM positive/negative Punkte, falls vorhanden.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(chip_rgb_uint8)

    box_xyxy = polygon_to_chip_bbox(poly_world, chip_transform)

    # ---- Punkte sammeln ----
    if mllm_result:
        pos_px = np.array([[float(x), float(y)] for x, y in mllm_result.get("positive_points", [])], dtype=np.float32)
        neg_px = np.array([[float(x), float(y)] for x, y in mllm_result.get("negative_points", [])], dtype=np.float32)

    else:
        pos_world = sample_points_in_polygon(poly_world, n=8) or [poly_world.representative_point().coords[0]]
        pos_px = project_xy_to_chip(pos_world, chip_transform)
        outer = poly_world.buffer(0.7)
        ring = outer.difference(poly_world)
        neg_world = sample_points_in_polygon(ring, n=8) if not ring.is_empty else []
        neg_px = project_xy_to_chip(neg_world, chip_transform) if neg_world else np.empty((0, 2), np.float32)

    if len(pos_px) == 0 and box_xyxy is None:
        return None

    point_coords = np.vstack([pos_px, neg_px]) if len(neg_px) else pos_px
    point_labels = np.hstack([np.ones(len(pos_px), dtype=np.int32),
                              np.zeros(len(neg_px), dtype=np.int32)]) if len(neg_px) else np.ones(len(pos_px), np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) else None,
        point_labels=point_labels if len(point_coords) else None,
        box=None,
        multimask_output=False,
    )
    if masks is None or len(masks) == 0:
        return None

    mask = (masks[0].astype(np.uint8) > 0).astype(np.uint8)

    # Debug speichern
    debug_mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(debug_mask)
    mask_img.save(f"{DEBUG_CHIP_DIR}/poly_{random.randint(0, 9999)}_mask.png")

    overlay = chip_rgb_uint8.copy()
    overlay[mask > 0] = (255, 0, 0)
    Image.fromarray(overlay).save(f"{DEBUG_CHIP_DIR}/poly_{random.randint(0, 9999)}_overlay.png")

    return mask_to_polygon(mask, chip_transform)


# ---- utilities ----
def to_png_b64(arr_rgb_uint8) -> str:
    im = Image.fromarray(arr_rgb_uint8)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def render_chip(rgb, poly_xy, draw_outline=True):
    img = Image.fromarray(rgb.copy())
    if draw_outline and poly_xy:
        drw = ImageDraw.Draw(img)
        drw.line(poly_xy + [poly_xy[0]], fill=(255, 0, 0), width=3)
    return np.array(img)

def crs_is_meter(crs):
    try:
        c = CRS.from_user_input(crs)
        return c.is_projected and any("metre" in ai.unit_name.lower() for ai in c.axis_info)
    except Exception:
        return False

def meters_to_pixels(m, px_m):
    return max(1, int(round(m / max(px_m, 1e-6))))

def polygon_to_chip_pixels(poly, chip_transform):
    inv = ~chip_transform
    coords = []
    if poly.geom_type == "Polygon":
        x, y = poly.exterior.xy
        coords = [tuple((inv * (xx, yy))) for xx, yy in zip(x, y)]
    elif poly.geom_type == "MultiPolygon":
        largest = max(poly.geoms, key=lambda a: a.area)
        x, y = largest.exterior.xy
        coords = [tuple((inv * (xx, yy))) for xx, yy in zip(x, y)]
    return [(int(round(cx)), int(round(cy))) for cx, cy in coords]

def rasterize_polygon(poly, out_shape, transform):
    shapes = [(poly, 1)]
    return features.rasterize(
        shapes=shapes, out_shape=out_shape, transform=transform,
        fill=0, all_touched=True, dtype="uint8"
    )

def mask_to_polygon(mask, transform):
    cnts,_ = cv2.findContours((mask > 0).astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in cnts:
        if len(c) < 4:   # zu wenig Punkte → Rechteck als Fallback
            x,y,w,h = cv2.boundingRect(c)
            xs = [x, x+w, x+w, x]
            ys = [y, y, y+h, y+h]
        else:
            xs = c[:,0,0]; ys = c[:,0,1]

        # zurück in Weltkoordinaten
        X = transform.c + xs*transform.a + ys*transform.b
        Y = transform.f + xs*transform.d + ys*transform.e
        p = Polygon(np.column_stack([X,Y])).buffer(0)
        if p.is_valid and not p.is_empty:
            polys.append(p)

    if not polys:
        return None
    mp = unary_union(polys)
    return mp if isinstance(mp, (Polygon, MultiPolygon)) else None

def polygon_to_chip_bbox(poly, chip_transform):
    """BBox in Chip-Pixelkoordinaten (x0,y0,x1,y1)"""
    inv = ~chip_transform
    minx, miny, maxx, maxy = poly.bounds
    (x0, y0) = inv * (minx, miny)
    (x1, y1) = inv * (maxx, maxy)
    # ordnen & clampen
    x0, x1 = int(min(x0, x1)), int(max(x0, x1))
    y0, y1 = int(min(y0, y1)), int(max(y0, y1))
    return np.array([x0, y0, x1, y1], dtype=np.int32)

def sample_points_in_polygon(poly, n=8):
    """gleichmäßig zufällige Punkte im Polygon (WKT)"""
    pts = []
    minx, miny, maxx, maxy = poly.bounds
    tries = 0
    while len(pts) < n and tries < n * 200:
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        if poly.contains(Polygon([(rx, ry)]).centroid):
            pts.append((rx, ry))
        tries += 1
    return pts

def project_xy_to_chip(points, chip_transform):
    """Koordinatenliste (Welt) => Chip-Pixel"""
    inv = ~chip_transform
    out = []
    for x, y in points:
        px, py = inv * (x, y)
        out.append([float(px), float(py)])
    return np.array(out, dtype=np.float32)
def save_mllm_points(chip_rgb, pos_px, neg_px, out_path):
    img = chip_rgb.copy()
    for x, y in pos_px.astype(int):
        cv2.circle(img, (x, y), 4, (0,255,0), -1)  # grün = positiv
    for x, y in neg_px.astype(int):
        cv2.circle(img, (x, y), 4, (0,0,255), -1)  # rot = negativ
    Image.fromarray(img).save(out_path)

# ---- main loop ----
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Checking inputs...")
    if not os.path.exists(IN_GPKG):
        raise FileNotFoundError(f"Input GPKG not found: {IN_GPKG}")
    if not os.path.exists(GEOTIFF_PATH):
        raise FileNotFoundError(f"Ortho image not found: {GEOTIFF_PATH}")

    print(f"[2/7] Reading polygons from {IN_GPKG} (layer='{IN_LAYER}') ...")
    gdf = gpd.read_file(IN_GPKG, layer=IN_LAYER)
    gdf = gdf.sample(5, random_state=42)
    n = len(gdf)
    if n == 0:
        print("[INFO] No polygons to QA. Exiting.")
        return
    print(f"[OK] Loaded {n} candidate building polygons.")

    print(f"[3/7] Opening ortho image: {GEOTIFF_PATH}")
    os.makedirs(DEBUG_CHIP_DIR, exist_ok=True)

    with rasterio.open(GEOTIFF_PATH) as src:
        crs = src.crs
        if not crs_is_meter(crs):
            print("[WARN] CRS is not meter-based. Morphology sizes will be approximate.")
        px_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0 or 0.1
        print(f"[INFO] Approx pixel size ~ {px_m:.3f} m/px")

        out_geoms, out_rows = [], []

        print("[4/7] Processing polygons with MLLM QA...")
        for i, (idx, row) in enumerate(gdf.iterrows()):
            if i >= 5:  # statt idx!
                break

            poly = row.geometry
            if poly is None or poly.is_empty:
                print(f"  - Polygon #{idx}: skipped (empty geometry)")
                continue

            centroid = poly.centroid
            print(f"Polygon #{idx} | centroid=({centroid.x:.1f}, {centroid.y:.1f}), area={poly.area:.1f}")

            # clear debug file on first run
            # Debug sammeln (am Ende einmal schreiben)
            if "debug_geoms" not in locals():
                debug_geoms = []
            debug_geoms.append(poly)

            if poly is None or poly.is_empty:
                if idx % LOG_EVERY_N == 0:
                    print(f"  - #{idx}: skipped (empty geometry)")
                continue

            # chip bounds
            minx, miny, maxx, maxy = poly.bounds
            bx = max(minx - PAD_METERS, src.bounds.left)
            by = max(miny - PAD_METERS, src.bounds.bottom)
            Bx = min(maxx + PAD_METERS, src.bounds.right)
            By = min(maxy + PAD_METERS, src.bounds.top)

            try:
                window = from_bounds(bx, by, Bx, By, transform=src.transform)
                tile = src.read(window=window, out_dtype=np.uint8)
                rgb = tile[:3] if tile.shape[0] >= 3 else np.vstack([tile]*3)
                rgb = np.moveaxis(rgb, 0, -1)

                chip_transform = src.window_transform(window)

                # scale chip to target size
                h, w = rgb.shape[:2]
                scale = CHIP_SIZE_PX / max(h, w)
                if abs(scale - 1.0) > 1e-3:
                    rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                    chip_transform = rasterio.Affine(
                        chip_transform.a/scale, chip_transform.b,      chip_transform.c,
                        chip_transform.d,      chip_transform.e/scale, chip_transform.f
                    )

                poly_xy = polygon_to_chip_pixels(poly, chip_transform)

                chip = render_chip(rgb, poly_xy, draw_outline=True)

                # ---- Grid overlay für MLLM ----
                chip_grid = add_grid_overlay(chip, step=50)

                # Save debug chip mit Grid
                chip_path = DEBUG_CHIP_DIR / f"poly_{i}.png"
                Image.fromarray(chip_grid).save(chip_path)
                print(f"  - Saved debug chip with grid: {chip_path}")

                # Bild für MLLM als Base64
                img_b64 = to_png_b64(chip_grid)

                # ---- MLLM call
                result = ask_mllm_for_fixes(img_b64, DEBUG_CHIP_DIR / f"poly_{i}_mllm.json")
                pos_px = np.array(result.get("positive_points", []), dtype=np.int32)
                neg_px = np.array(result.get("negative_points", []), dtype=np.int32)
                if len(pos_px) or len(neg_px):
                    save_mllm_points(chip, pos_px, neg_px, DEBUG_CHIP_DIR / f"poly_{i}_points.png")

                print(f"  - MLLM result (conf={result.get('confidence', 0):.2f}, issues={result.get('issues', [])})")
                if SHOW_FIX_JSON:
                    print(f"  - #{idx}: MLLM JSON: {result}")

                is_b = bool(result.get("is_building", True))
                conf = float(result.get("confidence", 0.5))
                fixes = result.get("suggested_fix", [])

                if (not is_b) or (conf < CONF_DROP):
                    if idx % LOG_EVERY_N == 0:
                        print(f"  - #{idx}: dropped (is_building={is_b}, conf={conf:.2f})")
                    continue

                # ---- apply fixes
                geom = poly
                mask = rasterize_polygon(geom, out_shape=chip.shape[:2], transform=chip_transform)

                for fx in fixes:
                    try:
                        op = fx.get("op")
                        if op == "buffer":
                            meters = float(fx.get("meters", 0))
                            geom = geom.buffer(meters)
                        elif op == "simplify":
                            tol = float(fx.get("tolerance_m", 0.5))
                            geom = geom.simplify(tol, preserve_topology=True)
                        elif op == "mask_refine":
                            if fx.get("enabled", False):
                                # SAM-Checkpoint (anpassen falls Pfad anders)
                                sam_ckpt = r"C:\git\Master-Thesis-GEOAI\sam_vit_b.pth"
                                if os.path.exists(sam_ckpt):
                                    new_geom = run_sam_refine(chip, geom, chip_transform,
                                                              sam_ckpt_path=sam_ckpt,
                                                              model_type="vit_b",
                                                              mllm_result=result)

                                    if new_geom and not new_geom.is_empty:
                                        geom = new_geom
                                        # Maske nachziehen für evtl. weitere Morphologie-Ops
                                        mask = rasterize_polygon(geom, out_shape=chip.shape[:2],
                                                                 transform=chip_transform)
                                        print("    [SAM] refined mask applied")
                                else:
                                    print(f"    [WARN] SAM checkpoint not found at {sam_ckpt} – skip mask_refine")

                        elif op == "remove_small_components":
                            amin = float(fx.get("min_area_m2", 6.0))
                            if geom.geom_type == "MultiPolygon":
                                geom = MultiPolygon([p for p in geom.geoms if p.area >= amin])
                        elif op in ("erode", "dilate"):
                            meters = float(fx.get("meters", 0.5))
                            k = meters_to_pixels(abs(meters), px_m)
                            kernel = np.ones((max(1, 2*k+1), max(1, 2*k+1)), np.uint8)
                            if op == "erode":
                                mask = cv2.erode(mask, kernel, iterations=1)
                            else:
                                mask = cv2.dilate(mask, kernel, iterations=1)
                            new_geom = mask_to_polygon(mask, chip_transform)
                            if new_geom:
                                geom = new_geom
                        elif op == "snap_to_rect":
                            rect = geom.minimum_rotated_rectangle
                            geom = rect.buffer(0)
                        # else: ignore unknown ops
                    except Exception as e:
                        print(f"    [WARN] Failed to apply fix {fx}: {e}")

                if not geom or geom.is_empty:
                    if idx % LOG_EVERY_N == 0:
                        print(f"  - #{idx}: empty after fixes → skip")
                    continue
                if geom.area < max(MIN_AREA_M2, 4*px_m*4*px_m):
                    if idx % LOG_EVERY_N == 0:
                        print(f"  - #{idx}: too small after fixes (area={geom.area:.2f})")
                    continue

                out_geoms.append(geom)
                out_rows.append(row.drop(labels=["geometry"], errors="ignore").to_dict())

                if idx % LOG_EVERY_N == 0:
                    print(f"  - #{idx}: kept (conf={conf:.2f}, fixes={len(fixes)})")

            except Exception as e:
                if idx % LOG_EVERY_N == 0:
                    print(f"  - #{idx}: ERROR while processing polygon → {e}")

    print(f"[5/7] QA finished. Kept {len(out_geoms)} / {n} polygons.")

    if not out_geoms:
        print("[INFO] No polygons survived after MLLM QA. Nothing to write.")
        return

    print(f"[6/7] Writing output to {OUT_GPKG} (layer='{OUT_LAYER}') ...")
    os.makedirs(os.path.dirname(OUT_GPKG), exist_ok=True)
    if os.path.exists(OUT_GPKG):
        try:
            os.remove(OUT_GPKG)
        except Exception as e:
            print(f"[WARN] Could not remove existing file: {e}")

    out = gpd.GeoDataFrame(out_rows, geometry=out_geoms, crs=gdf.crs)
    out.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    # Debug-Geometrien einmalig speichern
    if 'debug_geoms' in locals() and debug_geoms:
        dbg_path = BASE_DIR / "outputs" / "debug_polygons.gpkg"
        if dbg_path.exists():
            dbg_path.unlink()
        gpd.GeoDataFrame(geometry=debug_geoms, crs=gdf.crs).to_file(dbg_path, layer="test5", driver="GPKG")
        print(f"[INFO] Wrote {len(debug_geoms)} debug polygons → {dbg_path}")

    print(f"[7/7] ✅ Done. Wrote {len(out)} features → {OUT_GPKG} (layer={OUT_LAYER})")

    # Overlay finales Polygon auf Chip
    final_img = chip.copy()
    if geom and not geom.is_empty:
        # transform Polygon nach Chip-Pixelcoords
        poly_xy = polygon_to_chip_pixels(geom, chip_transform)
        drw = ImageDraw.Draw(Image.fromarray(final_img))
        drw.line(poly_xy + [poly_xy[0]], fill=(0, 255, 255), width=2)  # cyan = final
        Image.fromarray(final_img).save(DEBUG_CHIP_DIR / f"poly_{i}_final.png")


if __name__ == "__main__":
    main()
