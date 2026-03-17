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


# ==================================================
# PROMPTS (STRONGER + MULTI-ERROR SAFE)
# ==================================================

ERROR_SYSTEM = """
You are a strict geospatial QA system.

Rules:
- Output ONLY valid JSON
- No markdown, no explanations
- You MUST return ALL applicable errors
- You MUST evaluate ALL FOUR SIDES independently
"""

ERROR_USER = """
You see an aerial image with a GREEN polygon over a building.

IMPORTANT:
- Evaluate ALL four sides independently:
  NORTH (top), SOUTH (bottom), EAST (right), WEST (left)
- Even if partially cropped, still evaluate visible evidence
- MULTIPLE errors MUST be returned if present

----------------------------------------

STEP 1 — Per-side evaluation

For EACH side answer ALL:

1. Does the roof extend BEYOND the polygon?
   → UNDERSEGMENTATION

2. Does the polygon extend BEYOND the roof?
   → OVERSEGMENTATION

3. Is the polygon shifted or rotated incorrectly?
   → MISALIGNMENT

4. Is the roof cut off by the image boundary?
   → PARTIAL_VISIBILITY

----------------------------------------

STEP 2 — Combine ALL observed issues

- A polygon CAN have multiple errors
- DO NOT collapse to a single category
- DO NOT ignore minor errors

----------------------------------------

STEP 3 — Return STRICT JSON

Valid categories:
NO_ERROR, UNDERSEGMENTATION, OVERSEGMENTATION,
MISALIGNMENT, SHAPE_SIMPLIFICATION, PARTIAL_VISIBILITY

Format:

{
  "errors": [
    {
      "error_category": "UNDERSEGMENTATION",
      "error_location": ["EAST"],
      "error_description": "Roof extends beyond polygon on east side."
    },
    {
      "error_category": "OVERSEGMENTATION",
      "error_location": ["WEST"],
      "error_description": "Polygon extends beyond roof on west side."
    }
  ]
}

If truly perfect:

{
  "errors": [
    {
      "error_category": "NO_ERROR",
      "error_location": ["NONE"],
      "error_description": "Polygon matches roof."
    }
  ]
}
"""


# ==================================================
# UTILS
# ==================================================

def _parse_json_safe(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        return json.loads(raw)
    except Exception:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            print("⚠ JSON parse failed → returning empty dict")
            return {}


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(image_b64: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=800,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True}
            },
            messages=[
                {"role": "system", "content": ERROR_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ERROR_USER},
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

        print("\n--- ERROR RAW ---")
        print(raw)
        print("------------------\n")

        return _parse_json_safe(raw)

    except Exception as e:
        print(f"⚠ ERROR client failed: {e}")
        return None


# ==================================================
# POST-PROCESSING (VERY IMPORTANT)
# ==================================================

def _enforce_multi_error(errors: list):
    """
    Prevent single-class collapse.
    """

    if not errors:
        return []

    categories = {e.get("error_category") for e in errors}

    # If only one category → force diversity hint
    if len(categories) == 1 and "NO_ERROR" not in categories:
        errors.append({
            "error_category": "SHAPE_SIMPLIFICATION",
            "error_location": ["UNKNOWN"],
            "error_description": "Possible simplification or missing detail."
        })

    return errors


def _build_description(errors: list):
    return "; ".join(
        e.get("error_description", "")
        for e in errors
        if e.get("error_description")
    ) or "MLQA_ERROR"


# ==================================================
# PUBLIC API
# ==================================================

def analyze_errors(image_path: Path):

    img_b64 = _encode_image(image_path)

    result = _ask(img_b64)

    if result is None:
        return {
            "errors": [],
            "error_description": "MLQA_ERROR"
        }

    errors = result.get("errors", [])


    errors = _enforce_multi_error(errors)

    return {
        "errors": errors,
        "error_description": _build_description(errors)
    }