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

POINT_PROMPT = """
You are a precise pixel locator.

BLUE STAR marks the TARGET HOUSE at image center.

- All inside points must lie on visible roof pixels, not just center.
- Distribute inside points across roof area.

Return exactly:

{
  "inside": [[x,y],[x,y]],
  "outside": [[x,y],[x,y],[x,y]]
}

Rules:
- 2 points distributed across THIS roof (not same location)
- 3 points clearly outside THIS roof

- integers only
- JSON ONLY
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

def analyze_points(image_path: Path):

    img_b64 = _encode(image_path)

    r = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
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

    return _parse(r.choices[0].message.content)
