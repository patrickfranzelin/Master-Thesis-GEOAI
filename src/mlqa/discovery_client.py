"""
Discovery mode MLQA client for detecting all houses in a patch.
Used when house_present=False to find other buildings that might exist.
"""
import base64
import json
import re
from pathlib import Path
from openai import OpenAI
import os

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)

DISCOVERY_PROMPT = """
You are analyzing an aerial image patch to find ALL buildings/houses.

The GREEN polygon from the dataset is INCORRECT - it doesn't contain a house.
BUT there might be OTHER buildings in this patch.

TASK: Find ALL buildings/houses visible in this image patch, even if small or partial.

For EACH building you find:
1. Place 2-3 points INSIDE the roof area (distributed, not clustered)
2. Mark its approximate location

Return JSON with list of buildings:

{
  "buildings_found": [
    {
      "building_id": 1,
      "description": "rectangular metal roof, center-left",
      "inside_points": [[x1,y1], [x2,y2], [x3,y3]],
      "confidence": "high|medium|low"
    },
    {
      "building_id": 2,
      "description": "mud compound, top-right corner",
      "inside_points": [[x1,y1], [x2,y2]],
      "confidence": "high|medium|low"
    }
  ],
  "negative_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "total_buildings": 2
}

Rules:
- Look EVERYWHERE in the image, not just near the green polygon
- Include partial buildings at edges if roof is visible
- negative_points = 4-6 points clearly NOT on any roof (vegetation, paths, shadows)
- If NO buildings found at all, return empty buildings_found list
- Integers only, no decimals
"""


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parse_json_safe(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try removing markdown code blocks
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Return empty result if JSON parsing completely fails
            return {
                "buildings_found": [],
                "negative_points": [],
                "total_buildings": 0
            }


def discover_all_houses(image_path: Path):
    """
    Discover all buildings in the patch using MLQA.
    
    Returns:
        dict with:
            - buildings_found: list of building dicts with points
            - negative_points: points clearly off all buildings
            - total_buildings: count
    """
    
    img_b64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=1024,  # More tokens for multiple buildings
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DISCOVERY_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    )

    raw = response.choices[0].message.content
    result = _parse_json_safe(raw)
    
    # Ensure structure is valid
    if "buildings_found" not in result:
        result["buildings_found"] = []
    if "negative_points" not in result:
        result["negative_points"] = []
    if "total_buildings" not in result:
        result["total_buildings"] = len(result["buildings_found"])
    
    return result
