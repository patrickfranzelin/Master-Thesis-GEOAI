import base64
import json
import re
from pathlib import Path
from openai import OpenAI
import cv2
import numpy as np  # Add for shape fix

# ================= CONFIG =================
import os

RUNPOD_ID = os.environ["RUNPOD_ID"]
MODEL_NAME = "qwen3vl8b"

client = OpenAI(
    api_key="EMPTY",
    base_url=f"https://{RUNPOD_ID}-7860.proxy.runpod.net/v1"
)

folder_path = Path(r"C:\git\Master-Thesis-GEOAI\outputs\gdb_results")
output_folder = Path(r"C:\git\Master-Thesis-GEOAI\outputs\test_qwen8b_pointplacement")
output_folder.mkdir(exist_ok=True)

# =========================================


def parse_json_safe(raw):
    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {"inside": [], "outside": []}


def encode_image(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_image_shape(path):
    """Safe image shape getter"""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None, None
        return img.shape[1], img.shape[0]  # width, height
    except:
        return None, None


# Collect images
image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
print(f"Found {len(image_files)} images")

all_results = []

for img_path in image_files:

    print("\n================================================")
    print(f"Processing: {img_path.name}")

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print("Failed loading image")
            all_results.append({
                "filename": img_path.name,
                "house_points": [],
                "offhouse_points": [],
                "raw_output": "Failed to load image"
            })
            continue

        img_b64 = encode_image(img_path)
        img_w, img_h = img.shape[1], img.shape[0]  # Fixed: numpy array has .shape

        prompt = """
        You are a precise pixel locator.

        The BLUE STAR marks the TARGET HOUSE and is located at the CENTER of the image.

        Analyze the aerial image with the grid.

        Your task is to output exactly 6 GRID-SNAPPED pixel coordinates relative to the image:

        - 3 coordinates INSIDE THE ROOF of the TARGET HOUSE (the house with the blue star)
        - 3 coordinates clearly OUTSIDE THAT SAME HOUSE (grass, road, yard, etc — NOT any other building)

        IMPORTANT:
        - The target house is the one marked by the BLUE STAR in the center.
        - INSIDE points must lie on the roof of THIS house only.
        - OUTSIDE points must be outside THIS house (not on any building).
        - Ignore all other buildings.

        Return JSON ONLY:

        {
          "inside": [[x1,y1],[x2,y2],[x3,y3]],
          "outside": [[x4,y4],[x5,y5],[x6,y6]]
        }

        Rules:
        - Each coordinate must be two integers [x,y]
        - Use grid intersections or clear grid-aligned positions
        - Do NOT add explanations
        - Do NOT add text outside JSON
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=512
        )

        raw = response.choices[0].message.content
        parsed = parse_json_safe(raw)

        inside = parsed.get("inside", [])
        outside = parsed.get("outside", [])

        print(f"Parsed {len(inside)} inside / {len(outside)} outside")

        all_results.append({
            "filename": img_path.name,
            "house_points": inside,
            "offhouse_points": outside,
            "raw_output": raw,
            "image_shape": [img_h, img_w]  # Add for debugging
        })

        # Visualization
        overlay = img.copy()

        for pt in inside:
            if isinstance(pt, list) and len(pt) == 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < img_w and 0 <= y < img_h:
                    cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)  # Green inside

        for pt in outside:
            if isinstance(pt, list) and len(pt) == 2:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < img_w and 0 <= y < img_h:
                    cv2.circle(overlay, (x, y), 12, (0, 0, 255), -1)  # Red outside

        out_path = output_folder / f"{img_path.stem}_qwen3vl_result.png"
        cv2.imwrite(str(out_path), overlay)

        print(f"Saved {out_path.name}")

    except Exception as e:
        print(f"ERROR on {img_path.name}:", e)
        all_results.append({
            "filename": img_path.name,
            "house_points": [],
            "offhouse_points": [],
            "raw_output": f"Error: {str(e)}"
        })


# Save summary
summary_path = output_folder / "qwen3vl_all_results_summary.json"
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n================ COMPLETE =================")
print(f"Summary saved: {summary_path}")
print(f"Processed: {len(all_results)} images")
