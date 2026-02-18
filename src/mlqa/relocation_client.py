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

RELOCATION_PROMPT = """
You are a precise spatial locator.

A GREEN polygon marks an INCORRECT building footprint.

Select exactly ONE point clearly inside the MAIN roof
closest to this green polygon.

Important:
- Coordinates must be integers.
- Coordinate system:
  - (0,0) = top-left corner
  - (1000,1000) = bottom-right corner
- The point must lie clearly on visible roof pixels.
- Avoid placing the point near image borders.

Return ONLY valid JSON:

{
  "inside": [[x,y]]
}
"""

def _encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def _parse(raw):
    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {"inside": []}

def _denormalize(points, width, height):
    real = []
    for x, y in points:
        px = int((x / 1000) * width)
        py = int((y / 1000) * height)
        real.append([px, py])
    return real


def relocate_building(image_path: Path):

    img = cv2.imread(str(image_path))

    # Resize for stable spatial reasoning
    img_resized = cv2.resize(img, (1024, 1024))
    height, width = img_resized.shape[:2]

    img_b64 = _encode_image(img_resized)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RELOCATION_PROMPT},
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

    print("\n--- RELOCATION RAW ---")
    print(raw)
    print("----------------------")

    parsed = _parse(raw)

    inside_norm = parsed.get("inside", [])

    inside = _denormalize(inside_norm, width, height)

    return {
        "inside": inside
    }
