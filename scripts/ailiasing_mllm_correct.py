import os, io, json, base64, warnings
from pathlib import Path
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from PIL import Image, ImageDraw
import numpy as np
import cv2
from openai import OpenAI

# ---------- USER PARAMS ----------
BASE_DIR     = Path(__file__).resolve().parent.parent
IN_GPKG      = str(BASE_DIR / "outputs" / "buildings_mllm_corrected.gpkg")
IN_LAYER     = "buildings_mllm"
OUT_GPKG     = str(BASE_DIR / "outputs" / "buildings_mllm_smoothed.gpkg")
OUT_LAYER    = "buildings_mllm_smoothed"
DEBUG_DIR    = BASE_DIR / "outputs" / "debug_smooth"

MODEL        = "gpt-4.1"   # oder gpt-4o, gpt-4o-mini

# ---------- OPENAI Client ----------
client = OpenAI()  # liest OPENAI_API_KEY aus deiner Umgebung

SYSTEM_PROMPT = """
You are a geometry simplifier for building footprints.
Input: a WKT polygon and optionally an image with the polygon.
Task:
- Remove aliasing / zig-zag edges.
- Keep only the main structural corners of the building.
- Ensure polygon is valid and closed.
Output: only return a single WKT polygon (no JSON).
"""

# ---------- Utils ----------
def to_png_b64(arr_rgb_uint8) -> str:
    im = Image.fromarray(arr_rgb_uint8)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def render_poly_debug(poly, size=512):
    """Render simple polygon debug image for MLLM context."""
    poly = get_main_polygon(poly)   # <-- hier einbauen
    minx, miny, maxx, maxy = poly.bounds
    scale = size / max(maxx-minx, maxy-miny)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    drw = ImageDraw.Draw(img)
    coords = [((x-minx)*scale, (maxy-y)*scale) for x,y in poly.exterior.coords]
    drw.polygon(coords, outline="red", fill="yellow")
    return np.array(img)


def ask_mllm_for_simplify(poly):
    """Send one polygon (WKT + Debugimage) to MLLM and get smoothed WKT back."""
    wkt_str = poly.wkt
    debug_img = render_poly_debug(poly)
    img_b64 = to_png_b64(debug_img)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Here is the polygon in WKT:\n{wkt_str}\nPlease simplify."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}
        ]}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def get_main_polygon(geom):
    """Falls MultiPolygon: nimm die größte Fläche, sonst gib direkt zurück."""
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom
# ---------- Main ----------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"[1/4] Reading polygons from {IN_GPKG} ...")
    gdf = gpd.read_file(IN_GPKG, layer=IN_LAYER)
    print(f"[OK] Loaded {len(gdf)} polygons")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    out_geoms, out_rows = [], []

    print("[2/4] Sending polygons to MLLM for smoothing...")
    for i, row in gdf.iterrows():
        poly = row.geometry
        if not poly or poly.is_empty:
            continue
        try:
            new_wkt = ask_mllm_for_simplify(poly)
            new_poly = wkt.loads(new_wkt)
            if new_poly.is_valid and not new_poly.is_empty:
                out_geoms.append(new_poly)
                out_rows.append(row.drop(labels=["geometry"], errors="ignore").to_dict())

                # Debug speichern
                dbg_img = render_poly_debug(new_poly)
                Image.fromarray(dbg_img).save(DEBUG_DIR / f"poly_{i}_smoothed.png")
                print(f"  - Polygon {i} smoothed ✓")
        except Exception as e:
            print(f"  - Polygon {i}: ERROR {e}")

    print(f"[3/4] Writing {len(out_geoms)} smoothed polygons → {OUT_GPKG}")
    if out_geoms:
        out = gpd.GeoDataFrame(out_rows, geometry=out_geoms, crs=gdf.crs)
        out.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    else:
        print("[WARN] No polygons smoothed.")

    print("[4/4] Done ✅")

if __name__ == "__main__":
    main()
