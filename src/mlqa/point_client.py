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

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

POINT_PROMPT_POSITIVE = """
You are a precise spatial locator.

A BLUE STAR marks the center of the TARGET HOUSE.

Task:
1. Identify the roof of the house marked by the star.
2. Select 3 points ON the roof (spread apart).

Important:
- Coordinate system:
  - (0,0) = top-left corner
  - (1000,1000) = bottom-right corner
- Points must lie clearly on visible roof pixels.

Return ONLY valid JSON:

{
  "inside": [[x,y],[x,y],[x,y]]
}
"""

POINT_PROMPT_NEGATIVE = """
You are a precise spatial locator.

A BLUE STAR marks the center of the TARGET HOUSE.

Task:
1. Identify the roof of the house marked by the star.
2. Select 3 points OUTSIDE the roof.
   - Choose points clearly NOT on the roof
   - Use ground, vegetation, roads, or shadows.

Important:
- Coordinate system:
  - (0,0) = top-left corner
  - (1000,1000) = bottom-right corner
- Points must NOT lie on roof pixels.

Return ONLY valid JSON:

{
  "outside": [[x,y],[x,y],[x,y]]
}
"""

# --------------------------------------------------
# UTILS
# --------------------------------------------------

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
            return {}


def _denormalize(points, width, height):
    """Convert 0–1000 coordinates to actual pixel coords."""
    real = []
    for x, y in points:
        px = int((x / 1000) * width)
        py = int((y / 1000) * height)
        real.append([px, py])
    return real


def _call_model(prompt_text, image_b64):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
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

    print("\n--- MLLM RAW ---")
    print(raw)
    print("----------------")

    return _parse(raw)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def analyze_points(image_path: Path):

    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    img_b64 = _encode(image_path)

    # ----------------------------
    # 1Positive points call
    # ----------------------------
    pos_result = _call_model(POINT_PROMPT_POSITIVE, img_b64)
    inside_norm = pos_result.get("inside", [])

    # ----------------------------
    #  Negative points call
    # ----------------------------
    neg_result = _call_model(POINT_PROMPT_NEGATIVE, img_b64)
    outside_norm = neg_result.get("outside", [])

    # ----------------------------
    # Convert to pixel coords
    # ----------------------------
    inside = _denormalize(inside_norm, width, height)
    outside = _denormalize(outside_norm, width, height)

    return {
        "inside": inside,
        "outside": outside
    }