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
You are a geospatial QA system.
Return ONLY valid JSON.
"""

ALIGNMENT_USER = """
You see an aerial image with a GREEN polygon over a building.

Question:
Is the polygon shifted relative to the building?

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
You see an aerial image with a GREEN polygon over a building.

The alignment has already been evaluated.

Classify the SHAPE using ONLY these tags:

- STRUCTURE_MATCH
- SHAPE_MISMATCH
- MISSING_PARTS
- EXTRA_PARTS
- OVERSIMPLIFIED

Rules:
- Multiple tags allowed
- Only choose from the list
- No explanations

Return ONLY:

{
  "tags": ["..."]
}
"""

VALID_TAGS = {
    "STRUCTURE_MATCH",
    "SHAPE_MISMATCH",
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
You see an aerial image with a GREEN polygon over a building.

Known:
- misaligned: {misaligned}
- tags: {tags}

Write ONE short, simple sentence describing the issue and the area.

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