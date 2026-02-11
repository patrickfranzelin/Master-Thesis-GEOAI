# src/mlqa/discovery_client.py

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
You are a precise pixel locator.
Find buildings visible in this aerial image patch.

For buildings: Place points clearly inside the roof area.

Also provide: negative points clearly NOT on any building.

Return ONLY JSON in this format:

{
  "buildings": [
    {"inside_points": [[x1,y1],[x2,y2]]}
  ],
  "negative_points": [[x1,y1],[x2,y2]]
}
"""

def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parse_json_safe(raw):
    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {"buildings": [], "negative_points": []}


def discover_all_houses(image_path: Path):
    img_b64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=512,
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
    print("\n--- MLLM RAW RESPONSE ---")
    print(raw)
    print("-------------------------\n")

    result = _parse_json_safe(raw)

    if "buildings" not in result:
        result["buildings"] = []

    if "negative_points" not in result:
        result["negative_points"] = []

    return result
