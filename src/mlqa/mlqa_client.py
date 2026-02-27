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
# PROMPTS (UNCHANGED)
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
You are an expert geospatial analyst specializing in building footprint validation.

Your task is to classify geometric errors between a visible roof and a given polygon.

You must assign structured error categories from a fixed list.

Output ONLY valid JSON.
No markdown.
No explanations outside JSON.
"""

ERROR_USER = """
Input: An aerial image patch with a GREEN polygon outlining a building footprint.

Task:
Compare the green polygon to the visible roof structure inside the image.

Classify the geometric relationship using ONE of the following categories:

- NO_ERROR
- UNDERSEGMENTATION
- OVERSEGMENTATION
- MISALIGNMENT
- SHAPE_SIMPLIFICATION
- PARTIAL_VISIBILITY

Additionally:
- Specify where the mismatch occurs:
  Choose from: NORTH, SOUTH, EAST, WEST, CENTER, MULTIPLE, NONE
- Provide a short human-readable description.

Return ONLY valid JSON:

{
  "error_category": "...",
  "error_location": ["..."],
  "error_description": "..."
}
"""


# ==================================================
# UTILS
# ==================================================

def _parse_json_safe(raw):
    try:
        return json.loads(raw)
    except Exception:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        cleaned = re.sub(r',\s*}', '}', cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            print("⚠ JSON parse failed → returning empty dict")
            return {}


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(system_prompt: str, user_prompt: str, image_b64: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=512,
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
            "error_description": "No roof structure detected or MLQA failure"
        }

    # --------------------------
    # 2. Coverage
    # --------------------------
    coverage = _ask(COVERAGE_SYSTEM, COVERAGE_USER, img_b64)

    if coverage is None:
        coverage_value = False
    else:
        coverage_value = coverage.get("full_house_present", False)

    # --------------------------
    # 3. Error classification
    # --------------------------
    error_info = _ask(ERROR_SYSTEM, ERROR_USER, img_b64)

    if error_info is None:
        error_desc = "MLQA_ERROR"
    else:
        error_desc = error_info.get("error_description")

    return {
        "house_present": True,
        "full_house_present": coverage_value,
        "error_description": error_desc
    }