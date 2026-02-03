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


QA_PROMPT = """
You see an aerial image patch.

GREEN polygon = building footprint.

House_present = false if no roof exists inside or touching the polygon

Error_description= Your job is to give me a description whats AND where is somthing wrong with the polygon.

{
  "house_present": true | false,
  "error_description": string,
}

"""

def _parse_json_safe(raw):

    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {
                "house_present": False,
                "error_description": "PARSE_ERROR",
                #"whole_house_in_patch": False
            }


def _encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------
# Public API
# ---------------------------------------------------

def analyze_patch(image_path: Path):

    img_b64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QA_PROMPT},
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

    return _parse_json_safe(raw)
