import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
import cv2
import numpy as np
import os
#!/usr/bin/env python3
import os

# ============================================================
# CACHE PATH FIX — must be set before importing transformers
# ============================================================
os.environ["HF_HOME"] = "/data/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"   # backward compat only
os.environ["TORCH_HOME"] = "/data/torch_cache"
os.environ["XDG_CACHE_HOME"] = "/data/.cache"
os.environ["TMPDIR"] = "/data/tmp"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

print("✅ Environment paths set")
print("HF cache:", os.getenv("HF_HOME"))
print("Torch cache:", os.getenv("TORCH_HOME"))


# ============================================================
# CONFIGURATION
# ============================================================
model_id = "OpenGVLab/InternVL3-8B"
image_path = "/data/Master-Thesis-GEOAI/Theory/img_4.png"
out_path = "/data/Master-Thesis-GEOAI/outputs/internvl_points.png"
question = (
    "<image>\n"
    "You are a precise visual inspector.\n"
    "Look at the red polygon in this aerial image.\n"
    "Output exactly eight 2D pixel coordinates in JSON format like this:\n"
    "{'inside': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 'outside': [[x5, y5], [x6, y6], [x7, y7], [x8, y8]]}\n"
    "Rules:\n"
    "- Each coordinate must be two integers.\n"
    "- Choose 4 points well distributed **inside** the polygon (on the roof, not clustered).\n"
    "- Choose 4 points clearly **outside** the polygon (on grass or road, not clustered).\n"
    "- Do not include explanations or words — output only the JSON object."
)


# ============================================================
# LOAD MODEL
# ============================================================
print(f"🔄 Loading model: {model_id}")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
print(f"🖼️ Loading image: {image_path}")

transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

image = Image.open(image_path)
pixel_values = transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


# ============================================================
# INFERENCE
# ============================================================
print("🚀 Running inference...")
generation_config = dict(max_new_tokens=256, do_sample=False)

response = model.chat(tokenizer, pixel_values, question, generation_config)

print("\n🧠 Model answer:")
print("----------------------------------------------------")
print(response)
print("----------------------------------------------------")

# ============================================================
# PARSE POINTS AND DRAW
# ============================================================
try:
    matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", response)
    coords = np.array(matches, dtype=int)

    if coords.shape[0] >= 8:
        inside = coords[:4]
        outside = coords[4:8]

        img = cv2.imread(image_path)
        for (x, y) in inside:
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        for (x, y) in outside:
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        print(f"✅ Saved visualization with points to {out_path}")
    else:
        print("⚠️ Could not extract 8 points from model response.")

except Exception as e:
    print(f"⚠️ Failed to parse model output: {e}")
