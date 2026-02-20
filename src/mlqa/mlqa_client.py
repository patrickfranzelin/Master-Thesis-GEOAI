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
Output ONLY valid JSON,
"""

COVERAGE_USER = """
Input: An aerial image patch with a GREEN polygon representing a building footprint.

Instructions:
Evaluate if the green polygon accurately captures the MAIN body of the visible roof.

Criteria for TRUE:
- The green polygon covers the majority (>50%) of the visible roof area.
- The roof structure is complete within the image.

Criteria for FALSE:
- A significant portion of the building lies outside the green polygon.
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

def _parse_json_safe(raw):
    """Robustly parse JSON, handling markdown code blocks."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Remove markdown code blocks if present
        cleaned = re.sub(r"```json|```", "", raw).strip()
        # Remove any trailing commas before closing braces (common LLM error)
        cleaned = re.sub(r',\s*}', '}', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise MLQAParseError(
                f"Failed to parse MLQA response as JSON. Raw: {raw[:200]}"
            )

def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(system_prompt: str, user_prompt: str, image_b64: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
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


# ---------------------------------------------------
# Public API
# ---------------------------------------------------

def analyze_patch(image_path: Path):
    img_b64 = _encode_image(image_path)

    # Step 1: Presence check
    presence = _ask(PRESENCE_SYSTEM, PRESENCE_USER, img_b64)

    if not presence.get("house_present", False):
        return {
            "house_present": False,
            "full_house_present": False
        }

    # Step 2: Coverage check
    coverage = _ask(COVERAGE_SYSTEM, COVERAGE_USER, img_b64)

    return {
        "house_present": True,
        "full_house_present": coverage.get("full_house_present", False)
    }

