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
# STAGE 1 — ALIGNMENT
# ==================================================

ALIGNMENT_SYSTEM = """
You are a geospatial QualityAssurance system.
Return ONLY valid JSON.
"""

ALIGNMENT_USER = """
You see an aerial image with a GREEN polygon over a building.

Task:
Determine if the polygon is SHIFTED.

STRICT PROCEDURE (follow exactly):

1. Compare polygon edges with roof edges on each side:
   - TOP
   - BOTTOM
   - LEFT
   - RIGHT

2. For each side decide:
   - ALIGNED → edge overlaps roof edge
   - NOT_ALIGNED → clear gap or offset

3. Count how many sides are NOT_ALIGNED.

Decision rule:
- If 2 or more sides are NOT_ALIGNED → MISALIGNED (true)
- Otherwise → ALIGNED (false)

IMPORTANT:
- Ignore shape errors (missing parts, simplification)
- Only evaluate POSITION (shift)

Return ONLY:

{ "misaligned": true }

or

{ "misaligned": false }
"""

# ==================================================
# STAGE 2 — TAGS
# ==================================================

TAGS_SYSTEM = """
You are a geospatial QA system.
Return ONLY valid JSON.
"""

TAGS_USER = """
You see a arial image with a building on it, overlaying is a green polygon that in theory should follow perfectly the outlines  
with the building. Now i want to classify the error types of the green polygon. 
I now that nearly all of the Polygons are shifted according to the real building from the arial image ignore that fact and concentrate
on the other error types. Imagine that the shift isnt there.

ERROR TYPES:

- SHAPE_MISMATCH → wrong building footprint shape (NOT rotation)
- ORIENTATION_MISMATCH → correct shape but rotated incorrectly
- MISSING_PARTS → parts of roof outside polygon
- EXTRA_PARTS → polygon includes non-building areas

Return ONLY:

{
  "tags": ["..."]
}
"""

VALID_TAGS = {
    "STRUCTURE_MATCH",
    "SHAPE_MISMATCH",
    "ORIENTATION_MATCH",
    "ORIENTATION_MISMATCH",
    "MISSING_PARTS",
    "EXTRA_PARTS",
    "OVERSIMPLIFIED"
}

# ==================================================
# STAGE 3 — DESCRIPTION
# ==================================================

DESCRIPTION_SYSTEM = """
You are a geospatial QA assistant.
Return ONLY valid JSON.
"""

DESCRIPTION_TEMPLATE = """
You see an aerial image with a building and a GREEN polygon. Compare polygon outline with building outline
 Identify geometric differences as: shifted, missing parts, shape correctness, extra parts, level of detail, orientation

Then summarize in a short sentence the precice geometric errors (if any).
Return ONLY:

{{
  "description": "..."
}}
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
            print("JSON parse failed")
            return {}


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(system_prompt: str, user_prompt: str, image_b64: str, max_tokens=200):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=max_tokens,
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

        print("\n--- RAW ---")
        print(raw)
        print("-----------\n")

        return _parse_json_safe(raw)

    except Exception as e:
        print(f"⚠ request failed: {e}")
        return None


def _clean_tags(tags):
    if not isinstance(tags, list):
        return []
    return [t for t in tags if t in VALID_TAGS]


# ==================================================
# PUBLIC API
# ==================================================

def analyze_start_polygon(image_path: Path):

    img_b64 = _encode_image(image_path)

    # --------------------------
    # ALIGNMENT
    # --------------------------
    alignment_result = _ask(
        ALIGNMENT_SYSTEM,
        ALIGNMENT_USER,
        img_b64
    )

    if alignment_result is None:
        return {"status": "error"}

    misaligned = alignment_result.get("misaligned", False)

    # --------------------------
    # TAGS
    # --------------------------
    tag_result = _ask(
        TAGS_SYSTEM,
        TAGS_USER,
        img_b64
    )

    if tag_result is None:
        return {"status": "error"}

    tags = _clean_tags(tag_result.get("tags", []))

    # --------------------------
    # 3️ DESCRIPTION
    # --------------------------
    desc_prompt = DESCRIPTION_TEMPLATE.format(
        misaligned=str(misaligned).lower(),
        tags=", ".join(tags)
    )

    desc_result = _ask(
        DESCRIPTION_SYSTEM,
        desc_prompt,
        img_b64,
        max_tokens=100
    )

    if desc_result is None:
        description = ""
    else:
        description = desc_result.get("description", "")

    # --------------------------
    # FINAL OUTPUT
    # --------------------------
    return {
        "misaligned": misaligned,
        "tags": tags,
        "description": description
    }