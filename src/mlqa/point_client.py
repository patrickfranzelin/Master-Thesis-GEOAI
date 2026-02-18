import base64
import json
import re
from pathlib import Path
from openai import OpenAI
import os
import cv2

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)

POINT_PROMPT = """
You are a precise spatial locator.

A BLUE STAR marks the center of the TARGET HOUSE.

Task:
1. Identify the roof of the house marked by the star.
2. Select 2 points ON the roof (spread apart).
3. Select 2 points OFF the roof (on ground or water).

Important:
- Coordinate system: 
  - (0,0) = top-left corner
  - (1000,1000) = bottom-right corner
- Points must lie clearly on visible roof pixels.

Return ONLY valid JSON:

{
  "inside": [[x,y],[x,y]],
  "outside": [[x,y],[x,y]]
}
"""

def _encode(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def _parse(raw):
    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {"inside": [], "outside": []}


def _denormalize(points, width, height):
    """Convert 0–1000 coordinates to actual pixel coords."""
    real = []
    for x, y in points:
        px = int((x / 1000) * width)
        py = int((y / 1000) * height)
        real.append([px, py])
    return real


def analyze_points(image_path: Path):

    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    img_b64 = _encode(image_path)

    r = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": POINT_PROMPT},
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

    raw = r.choices[0].message.content

    print("\n--- POINT MLLM RAW ---")
    print(raw)
    print("----------------------")

    parsed = _parse(raw)

    inside_norm = parsed.get("inside", [])
    outside_norm = parsed.get("outside", [])

    inside = _denormalize(inside_norm, width, height)
    outside = _denormalize(outside_norm, width, height)

    return {
        "inside": inside,
        "outside": outside
    }
