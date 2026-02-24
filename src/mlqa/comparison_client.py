# src/mlqa/comparison_client.py

import base64
import json
import re
import cv2
import os
from pathlib import Path
from openai import OpenAI

from src.patches.extractor import extract_patch_pixel
from src.patches.create_patch_output import create_comparison_patch

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)


# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

COMPARISON_SYSTEM = """
You are an expert geospatial analyst specializing in aerial imagery and building footprint quality.
Your task is to compare two polygon segmentations of the same building roof.
Output ONLY valid JSON. No markdown, no explanations.
"""

COMPARISON_USER = """
You are shown a side-by-side aerial image of the SAME building.

- LEFT panel: The ORIGINAL polygon (drawn in RED).
- RIGHT panel: The REFINED polygon (drawn in GREEN).

Both polygons attempt to segment the roof of the building.

Evaluate which polygon is BETTER by these criteria:
1. Tighter fit — the polygon boundary closely follows the actual roof edges.
2. Less error — fewer areas where the polygon over- or under-shoots the roof.
3. Finer segmentation — corners and overhangs are captured correctly.

Return ONLY valid JSON:

{
  "better": "refined" | "original" | "equal",
  "reason": "Short explanation (max 2 sentences)"
}
"""


# --------------------------------------------------
# UTILS
# --------------------------------------------------

def _encode_path(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}


def _ask_comparison(comp_path: Path) -> dict:
    img_b64 = _encode_path(comp_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COMPARISON_USER},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ]
    )

    raw = response.choices[0].message.content
    print("\n--- COMPARISON RAW ---")
    print(raw)
    print("----------------------")

    return _parse(raw)


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def compare_polygons(
    img_big,
    start_poly_px,
    refined_poly_px,
    out_dirs: dict,
    bid: int,
    context: float = 2.0,
    out_size: int = 512,
) -> dict:
    """
    1. Crops a shared patch around start_poly_px via extract_patch_pixel.
    2. Projects BOTH polygons into that crop space.
    3. Renders side-by-side via create_comparison_patch → saved to out_dirs["comparison"].
    4. Asks MLLM which polygon better segments the roof.

    Args:
        img_big:           Full aerial image (numpy BGR).
        start_poly_px:     Original polygon as Shapely Polygon in img_big pixel coords.
        refined_poly_px:   Refined SAM polygon as Shapely Polygon in img_big pixel coords.
        out_dirs:          Dict with at least key "comparison" → Path.
        bid:               Building ID for filename.
        context:           Context factor passed to extract_patch_pixel (default 2.0).
        out_size:          Patch size in pixels (default 512).

    Returns:
        {
            "better":      "refined" | "original" | "equal" | "unknown",
            "reason":      str,
            "comp_path":   Path  (saved comparison PNG)
        }
    """
    # --- 1. Crop patch around the start polygon (anchor) ---
    crop, start_poly_crop, crop_info = extract_patch_pixel(
        img_big, start_poly_px, out_size=out_size, context=context
    )

    # --- 2. Project refined polygon into the same crop space ---
    x1, y1, w_crop, h_crop = crop_info
    sx = out_size / w_crop
    sy = out_size / h_crop

    from shapely.affinity import translate, scale as shp_scale
    refined_shifted  = translate(refined_poly_px, xoff=-x1, yoff=-y1)
    refined_poly_crop = shp_scale(refined_shifted, xfact=sx, yfact=sy, origin=(0, 0))

    # --- 3. Render + save comparison patch ---
    comp_path = create_comparison_patch(
        img=crop,
        start_poly_px=start_poly_crop,
        refined_poly_px=refined_poly_crop,
        out_dirs=out_dirs,
        bid=bid,
    )

    # --- 4. Ask MLLM ---
    result = _ask_comparison(comp_path)

    return {
        "better":    result.get("better", "unknown"),
        "reason":    result.get("reason", ""),
        "comp_path": comp_path,
    }
