import json
import re
from openai import OpenAI
from pathlib import Path
import base64
import os

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)


class MLQAParseError(Exception):
    pass


# ==================================================
# PROMPTS
# ==================================================

PRESENCE_SYSTEM = """
You are an expert geospatial analyst specializing in aerial imagery.
Your task is to detect man-made roof structures within specified polygon boundaries.
Output ONLY valid JSON. No markdown, no explanations.
"""

PRESENCE_USER = """
Input: An aerial image patch containing a GREEN polygon outlining a specific area.

Instructions:
1. Inspect the area strictly INSIDE the green polygon boundaries.
2. Look for visual patterns typical of man-made roofs (e.g., rusted metal sheets, shingles, concrete slabs, or uniform geometric textures).
3. Ignore surrounding grass, water, or vegetation unless it is part of a structure.

Question: Is there clear visual evidence of a roof or man-made structure contained within the green polygon?

Expected Output Format:
{
  "house_present": true
}
OR
{
  "house_present": false
}
"""

COVERAGE_SYSTEM = """
You are an expert geospatial analyst specializing in building footprint coverage.
Your task is to evaluate if a polygon captures the main body of a visible roof and if the whole building is visible in the patch.
Output ONLY valid JSON.
"""

COVERAGE_USER = """
Input: An aerial image patch with a GREEN polygon representing a building footprint.

Instructions:
Evaluate if the green polygon accurately captures the MAIN body of the visible roof.

Criteria for TRUE:
- The green polygon covers the majority (>50%) of the visible roof area
- All roof edges are visible in the image
- The roof structure is complete within the image.

Criteria for FALSE:
- A significant portion of the building lies outside the green polygon.
- Polygon segments only a small part of the roof.
- The image cuts off the building.

Expected Output Format:
{
  "full_house_present": true
}
OR
{
  "full_house_present": false
}
"""

ERROR_SYSTEM = """
You are a precise geospatial quality analyst evaluating building footprint polygons in aerial imagery.
Output ONLY valid JSON. No markdown. No text outside JSON.
"""

ERROR_USER = """
You see an aerial image with a GREEN polygon drawn over a building area.

STEP 1 — Examine each side independently (NORTH=top, SOUTH=bottom, EAST=right, WEST=left):
- Does the roof extend BEYOND the polygon on this side? → UNDERSEGMENTATION on that side
- Does the polygon extend BEYOND the roof on this side? → OVERSEGMENTATION on that side
- Is the polygon shifted/rotated but roughly correct size? → MISALIGNMENT
- Is the roof cut off at the image border? → PARTIAL_VISIBILITY

STEP 2 — List ALL errors you observe. A polygon can have BOTH oversegmentation on one side
AND undersegmentation on another side simultaneously.

STEP 3 — For each error, specify which sides are affected.

Valid categories: NO_ERROR, UNDERSEGMENTATION, OVERSEGMENTATION, MISALIGNMENT, SHAPE_SIMPLIFICATION, PARTIAL_VISIBILITY

Return ONLY this JSON:
{
  "errors": [
    {
      "error_category": "UNDERSEGMENTATION",
      "error_location": ["EAST", "SOUTH"],
      "error_description": "Roof extends beyond polygon on east and south sides."
    }
  ]
}

If no error exists:
{
  "errors": [
    {
      "error_category": "NO_ERROR",
      "error_location": ["NONE"],
      "error_description": "Polygon accurately matches the visible roof."
    }
  ]
}
"""


# ==================================================
# UTILS
# ==================================================

def _parse_json_safe(raw):
    # Strip Qwen3 thinking block before parsing
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except Exception:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        cleaned = re.sub(r',\s*}', '}', cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            print("JSON parse failed → returning empty dict")
            return {}


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(system_prompt: str, user_prompt: str, image_b64: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True}
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content

        print("\n--- MLLM RAW RESPONSE ---")
        print(raw)
        print("-------------------------\n")

        return _parse_json_safe(raw)

    except Exception as e:
        print(f"⚠ MLQA temporary error → skipping: {e}")
        return None


# ==================================================
# PUBLIC API
# ==================================================

def analyze_patch(image_path: Path):

    img_b64 = _encode_image(image_path)

    # --------------------------
    # 1. Presence
    # --------------------------
    presence = _ask(PRESENCE_SYSTEM, PRESENCE_USER, img_b64)

    if presence is None or not presence.get("house_present", False):
        return {
            "house_present": False,
            "full_house_present": False,
            "errors": [],
            "error_description": "No roof structure detected or MLQA failure",
        }

    # --------------------------
    # 2. Coverage
    # --------------------------
    coverage = _ask(COVERAGE_SYSTEM, COVERAGE_USER, img_b64)
    coverage_value = coverage.get("full_house_present", False) if coverage else False

    # --------------------------
    # 3. Multi-error classification
    # --------------------------
    error_info = _ask(ERROR_SYSTEM, ERROR_USER, img_b64)

    if error_info is None:
        errors = []
        error_description = "MLQA_ERROR"
    else:
        errors = error_info.get("errors", [])
        error_description = "; ".join(
            e.get("error_description", "")
            for e in errors
            if e.get("error_description")
        ) or "MLQA_ERROR"

    return {
        "house_present": True,
        "full_house_present": coverage_value,
        "errors": errors,
        "error_description": error_description,
    }
