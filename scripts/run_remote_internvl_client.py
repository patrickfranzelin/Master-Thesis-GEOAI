#!/usr/bin/env python3
import os, json, requests
from tqdm import tqdm
from src.utils.io import save_json

API_URL = os.environ.get("RUNPOD_URL", "https://<your-runpod-id>.runpod.io:8080/infer_points")
CROP_DIR = "outputs/crops"
OUT_JSON = "outputs/mllm_results.json"

def main():
    results = []
    for fname in tqdm(sorted(f for f in os.listdir(CROP_DIR) if f.endswith(".png")), desc="Remote inference"):
        png_path = os.path.join(CROP_DIR, fname)
        json_path = png_path.replace(".png", ".json")

        with open(json_path) as f:
            poly_json = f.read()

        with open(png_path, "rb") as f:
            files = {"image": (fname, f, "image/png")}
            data = {"poly_json": poly_json}
            r = requests.post(API_URL, files=files, data=data, timeout=300)
            results.append(r.json())

    save_json(results, OUT_JSON)
    print(f"✅ Collected {len(results)} results → {OUT_JSON}")

if __name__ == "__main__":
    main()
