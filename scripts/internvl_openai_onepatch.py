import base64
import json
import re
from pathlib import Path
from openai import OpenAI
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup
client = OpenAI(api_key="EMPTY", base_url="https://7ygcmpo7igft4k-7860.proxy.runpod.net/v1")
folder_path = Path(r"D:\git\Master-Thesis-GEOAI\data\gdb_results")
output_folder = folder_path / "results"
output_folder.mkdir(exist_ok=True)


def add_grid_overlay(img, step=50):
    return img

def parse_json_safe(raw):
    """Extract JSON even from malformed output."""
    try:
        return json.loads(raw)
    except:
        cleaned = re.sub(r'```json?\s*|\s*```', '', raw).strip()
        try:
            return json.loads(cleaned)
        except:
            return {"inside": [], "outside": []}


# Process all images in folder
image_files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))
print(f"🔍 Found {len(image_files)} images to process")

all_results = []

for img_path in image_files:
    print(f"\n{'=' * 60}")
    print(f"📸 Processing: {img_path.name}")

    try:
        # === STEP 1: LOAD + GRID OVERLAY ===
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            print(f"❌ Failed to load {img_path.name}")
            continue

        img_grid = add_grid_overlay(img_raw)
        grid_path = output_folder / f"{img_path.stem}_grid.png"
        cv2.imwrite(str(grid_path), img_grid)
        img_b64 = base64.b64encode(grid_path.read_bytes()).decode("utf-8")

        prompt = """You are a precise pixel locator. Analyze the aerial image with the grid

        Output exactly 6 GRID-snapped coordinates:
        - 3 randomly distributed INSIDE THE ROOF (roofs/buildings)
        - 3 clearly OUTSIDE THE ROOF (grass/road etc)

        JSON ONLY:
        {
          "inside": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  // INSIDE BUILDING
          "outside": [[x5,y5],[x6,y6],[x7,y7],[x8,y8]]  // OUTSIDE BUILDING
        }

        Rules:
        - Each coord = 2 integers [x,y]
        - ALWAYS JSON """

        resp = client.chat.completions.create(
            model="internvl8b",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
            }],
            temperature=0.0, max_tokens=256,
            response_format={"type": "json_object"}
        )

        raw_out = resp.choices[0].message.content
        out_json = parse_json_safe(raw_out)
        inside = out_json.get("inside", [])
        outside = out_json.get("outside", [])

        print(f"📊 Parsed: {len(inside)} house pts, {len(outside)} off-house pts")

        # Store result
        result = {
            "filename": img_path.name,
            "house_points": inside,
            "offhouse_points": outside,
            "raw_output": raw_out
        }
        all_results.append(result)

        # === STEP 3: VISUALIZE RESULT ===
        overlay = img_raw.copy()
        # House points (green)
        for pt in inside:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)
            cv2.putText(overlay, f"[{x},{y}]", (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Off-house (red)
        for pt in outside:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 12, (0, 0, 255), -1)
            cv2.putText(overlay, f"[{x},{y}]", (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add faint grid reference
        overlay_grid = add_grid_overlay(overlay, step=50)

        # Save RESULT
        result_path = output_folder / f"{img_path.stem}_result.png"
        cv2.imwrite(str(result_path), overlay_grid)
        print(f"💾 Saved: {result_path.name}")

    except Exception as e:
        print(f"❌ Error processing {img_path.name}: {str(e)}")
        continue

# === SAVE SUMMARY ===
summary_path = output_folder / "all_results_summary.json"
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"🎉 BATCH PROCESSING COMPLETE!")
print(f"📁 Results saved to: {output_folder}")
print(f"📋 Summary JSON: {summary_path}")
print(f"✅ Processed {len(all_results)} images successfully")

# Print summary table
print("\n📊 SUMMARY TABLE:")
print("Filename" + " " * 20 + "House pts  Off-house pts")
print("-" * 60)
for result in all_results:
    house_count = len(result["house_points"])
    off_count = len(result["offhouse_points"])
    print(f"{result['filename'][:25]:25} | {house_count:8} | {off_count:10}")
