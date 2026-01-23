import base64
import json
import re
from pathlib import Path
from openai import OpenAI
import cv2
import numpy as np
import matplotlib.pyplot as plt

client = OpenAI(api_key="EMPTY", base_url="https://7ygcmpo7igft4k-7860.proxy.runpod.net/v1")
img_path = Path(r"D:\git\Master-Thesis-GEOAI\data\testbild_house.png")

def add_grid_overlay(img, step=50):
    """Add prominent white grid with cyan labels."""
    h, w, _ = img.shape
    overlay = img.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 2)
        cv2.putText(overlay, str(x), (x+3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
        cv2.putText(overlay, str(y), (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return overlay

def parse_json_safe(raw):
    """Extract JSON even from malformed output."""
    try: return json.loads(raw)
    except:
        cleaned = re.sub(r'```json?\s*|\s*```', '', raw).strip()
        try: return json.loads(cleaned)
        except: return {"inside": [], "outside": []}

# === STEP 1: LOAD + GRID OVERLAY ===
print("📸 Loading image + adding GRID...")
img_raw = cv2.imread(str(img_path))
img_grid = add_grid_overlay(img_raw)  # MLLM sees THIS
grid_path = img_path.with_name(img_path.stem + "_grid.png")
cv2.imwrite(str(grid_path), img_grid)
img_b64 = base64.b64encode(grid_path.read_bytes()).decode("utf-8")

# Show INPUT to MLLM
plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
plt.title("✅ INPUT TO MLLM (House + Red Polygon + 50px GRID)")
plt.axis('off'); plt.tight_layout(); plt.show()

# === STEP 2: MLLM ANALYSIS ===
prompt = """Precise inspector. See WHITE GRID (50px steps, cyan labels).

RED polygon + house visible.

Output exactly 8 GRID-snapped coordinates:
- 4 ON HOUSE/ROOF 
- 4 OFF CLEARLY OF THE HOUSE (grass/road)

JSON ONLY:
{
  "inside": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  // HOUSE
  "outside": [[x5,y5],[x6,y6],[x7,y7],[x8,y8]]  // NOT HOUSE
}

Use grid intersections!"""

resp = client.chat.completions.create(
    model="internvl8b",
    messages=[{
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    }],
    temperature=0.0, max_tokens=300,
    response_format={"type": "json_object"}
)

raw_out = resp.choices[0].message.content
print("\n🤖 MLLM Raw Output:")
print(raw_out)

out_json = parse_json_safe(raw_out)
inside = out_json.get("inside", [])
outside = out_json.get("outside", [])

print(f"\n📊 Parsed: {len(inside)} house pts, {len(outside)} off-house pts")

# === STEP 3: VISUALIZE RESULT ===
overlay = img_raw.copy()
# House points (green)
for pt in inside:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(overlay, (x, y), 12, (0, 255, 0), -1)
    cv2.putText(overlay, f"[{x},{y}]", (x+15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# Off-house (red)
for pt in outside:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(overlay, (x, y), 12, (0, 0, 255), -1)
    cv2.putText(overlay, f"[{x},{y}]", (x+15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Add faint grid reference
overlay_grid = add_grid_overlay(overlay, step=50)

# Save + show RESULT
result_path = img_path.with_name(img_path.stem + "_result.png")
cv2.imwrite(str(result_path), overlay_grid)
print(f"\n💾 RESULT saved: {result_path}")

plt.figure(figsize=(14,10))
plt.imshow(cv2.cvtColor(overlay_grid, cv2.COLOR_BGR2RGB))
plt.title("🎯 FINAL RESULT: Green=House, Red=Off-House | Coords labeled")
plt.axis('off'); plt.tight_layout(); plt.show()

print("\n📋 SUMMARY:")
print(json.dumps({"house_points": inside, "offhouse_points": outside}, indent=2))
print(f"\n✅ Done! Check {result_path}")
