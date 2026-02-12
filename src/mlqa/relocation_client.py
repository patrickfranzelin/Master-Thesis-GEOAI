# src/mlqa/relocation_client.py

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

RELOCATION_PROMPT = """
You are a precise pixel locator.

The GREEN polygon is an incorrect building footprint.

Find the main roof structure closest to this polygon.

Place points inside that roof
Place points outside that roof

Return JSON only:

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

def relocate_building(image_path: Path):
    img_b64 = _encode(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=300,
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
    print("----------------------\n")

    return _parse(raw)
