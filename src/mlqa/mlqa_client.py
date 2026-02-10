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
    """Raised when MLQA response cannot be parsed as valid JSON."""
    pass


LOW_CONTRAST_QA_PROMPT = """
Aerial ortho patch from Sahel/Africa. GREEN polygon = building footprint from dataset. 
CENTER STAR marks patch center. Grid helps orientation.

TASK: Detect if ANY man‑made structure exists INSIDE/TOUCHING GREEN polygon.

✅ HOUSE EXAMPLES (even if mud/earth‑colored, irregular, courtyard‑style):
- Corrugated iron roofs (shiny rectangles)  
- Mud brick compounds (earth‑tone rectangles, internal walls)
- Flat roofs blending with dirt but rectangular/geometric 
- Small protrusions/extensions = part of house

❌ NO HOUSE:
- Pure vegetation/grass inside polygon 
- Only sandy ground, paths, shadows

Rules:
- house_present=true if ANY roof/structure inside polygon (even partial, faint, earth‑colored)
- house_present=false ONLY if polygon clearly empty (vegetation/sand only)

Error_description: SPECIFIC location + problem. Examples:
- "No structure inside, only grass"
- "Green polygon offset east, misses mud compound" 
- "Polygon too small, cuts NW corner of iron roof"

Strict JSON:
{
  "house_present": true/false,
  "error_description": "exact description"
}
"""
QA_PROMPT = """
You see an aerial image patch.

GREEN polygon = building footprint.

Definitions:
- "house_present": true if any roof or man-made structure is inside or touching the polygon.
- "full_house_present": true if the polygon covers nearly all of the house footprint (area). 
  If the polygon cuts off large parts of the roof or only covers a small corner, set it to false.

Return ONLY a single JSON object with this exact schema, no extra text:

{
  "house_present": true or false,
  "full_house_present": true or false
  "error_description": None
}
"""

def _parse_json_safe(raw):
    """
    Parse MLQA response JSON.
    
    Raises MLQAParseError if parsing fails completely.
    This ensures parse failures abort the pipeline instead of 
    creating false negatives.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Parse failure is a critical error - don't return false data
            raise MLQAParseError(
                f"Failed to parse MLQA response as JSON. Raw response: {raw[:200]}"
            )


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
