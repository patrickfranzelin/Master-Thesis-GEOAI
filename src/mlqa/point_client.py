import base64
import json
import re
from openai import OpenAI
from pathlib import Path
import os
import cv2

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)


class PointParseError(Exception):
    pass


# ==================================================
# SYSTEM PROMPT
# ==================================================

POINT_SYSTEM = """
You are a precise spatial locator specializing in aerial imagery.
You must output ONLY valid JSON.
No markdown.
No explanations.
"""


# ==================================================
# USER PROMPTS
# ==================================================

def _build_positive_user(already_placed: list[list[int]]) -> str:
    avoid_str = (
        f"\nAlready placed points (DO NOT place near these): {already_placed}"
        if already_placed else ""
    )

    return f"""
A BLUE STAR marks the center of the TARGET HOUSE.

Task:
1. Identify the roof of the house marked by the star.
2. Select exactly 1 point ON the roof.
   - It must be clearly on visible roof pixels.
   - It must be spatially separated from previous points.{avoid_str}

Coordinate system:
- (0,0) = top-left
- (1000,1000) = bottom-right

Return ONLY valid JSON:

{{
  "inside": [[x,y]]
}}
""".strip()


NEGATIVE_USER = """
A BLUE STAR marks the center of the TARGET HOUSE.

Task:
1. Identify the roof of the house marked by the star.
2. Select 3 points OUTSIDE the roof.
   - Choose ground, vegetation, roads, or shadows.
   - Points must NOT lie on roof pixels.
   - Spread them apart spatially.

Coordinate system:
- (0,0) = top-left
- (1000,1000) = bottom-right

Return ONLY valid JSON:

{
  "outside": [[x,y],[x,y],[x,y]]
}
""".strip()


# ==================================================
# UTILS
# ==================================================

def _parse_json_safe(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise PointParseError(
                f"Failed to parse point response. Raw: {raw[:200]}"
            )


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _ask(user_prompt: str, image_b64: str, max_tokens=128) -> dict:

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": POINT_SYSTEM
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
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

    print("\n--- POINT RAW ---")
    print(raw)
    print("------------------")

    return _parse_json_safe(raw)


def _denormalize(points: list, width: int, height: int) -> list:
    return [
        [int((x / 1000) * width), int((y / 1000) * height)]
        for x, y in points
    ]


# ==================================================
# POSITIVE (ITERATIVE)
# ==================================================

def _collect_positive_points(image_b64: str, n: int) -> list[list[int]]:

    collected = []

    for i in range(n):

        prompt = _build_positive_user(collected)

        result = _ask(prompt, image_b64)

        pts = result.get("inside", [])

        if pts:
            collected.append(pts[0])
        else:
            print(f"[WARN] No valid positive point ({i+1}/{n})")

    return collected


# ==================================================
# PUBLIC API
# ==================================================

def analyze_points(image_path: Path, n_points: int = 3) -> dict:

    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    img_b64 = _encode_image(image_path)

    # --------------------------
    # Positive (step-by-step)
    # --------------------------
    inside_norm = _collect_positive_points(
        img_b64,
        n_points
    )

    # --------------------------
    # Negative (single call)
    # --------------------------
    neg_result = _ask(
        NEGATIVE_USER,
        img_b64,
        max_tokens=256
    )

    outside_norm = neg_result.get("outside", [])

    # --------------------------
    # Convert to pixel coords
    # --------------------------
    inside = _denormalize(inside_norm, width, height)
    outside = _denormalize(outside_norm, width, height)

    return {
        "inside": inside,
        "outside": outside
    }