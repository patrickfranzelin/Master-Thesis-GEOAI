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
You are analyzing an aerial image to find buildings.

The GREEN polygon doesn't contain a house, but there might be OTHER buildings in this image.

TASK: Count how many buildings/houses you can see. For EACH building, place 2-3 points on its roof.

Return JSON ONLY:

{
  "total_buildings": 0,
  "building1_points": [[x,y], [x,y]],
  "building2_points": [[x,y], [x,y]],
  "building3_points": [[x,y], [x,y]],
  "negative_points": [[x,y], [x,y], [x,y]]
}

Rules:
- total_buildings = count of buildings you see (0 to 3)
- For each building: place 2-3 roof points in building1_points, building2_points, building3_points
- If you see 0 buildings, only include total_buildings and negative_points
- If you see 1 building, include total_buildings, building1_points, and negative_points
- negative_points = 3-4 points clearly NOT on any roof (grass, road, shadows)
- Use integers only, no decimals
- Return ONLY JSON, no explanations
"""


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parse_json_safe(raw):
    """
    Parse the simplified discovery JSON format.
    Converts building1_points, building2_points, etc. to buildings_found list.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try removing markdown code blocks
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Return empty result if JSON parsing completely fails
            return {
                "buildings_found": [],
                "negative_points": [],
                "total_buildings": 0
            }
    
    # Convert simplified format to standard format
    total = data.get("total_buildings", 0)
    negative_pts = data.get("negative_points", [])
    
    buildings_found = []
    for i in range(1, 4):  # Support up to 3 buildings
        key = f"building{i}_points"
        if key in data and data[key]:
            buildings_found.append({
                "building_id": i,
                "inside_points": data[key],
                "description": f"building_{i}",
                "confidence": "medium"
            })
    
    return {
        "buildings_found": buildings_found,
        "negative_points": negative_pts,
        "total_buildings": total
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
