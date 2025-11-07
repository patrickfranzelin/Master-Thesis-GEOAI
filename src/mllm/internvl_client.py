from __future__ import annotations
import re, os, cv2, numpy as np
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer
import torch
from .prompts import points_prompt


def set_hf_env():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")


class InternVL3Points:
    def __init__(self, model_id: str, device: str, max_new_tokens: int = 256):
        set_hf_env()
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        dtype = torch.bfloat16 if (dev == "cuda") else torch.float32

        print(f"🔧 Loading model: {model_id} on {dev}")
        # ✅ Use AutoModel (NOT AutoModelForCausalLM)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if dev == "cuda" else None,
            trust_remote_code=True
        ).eval().to(dev)

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.dev = dev
        self.max_new_tokens = max_new_tokens
        print("✅ InternVL model initialized successfully.")

    @staticmethod
    def overlay_polygon(img: Image.Image, poly_xy: list[tuple[int, int]]) -> Image.Image:
        draw = ImageDraw.Draw(img)
        draw.line(poly_xy + [poly_xy[0]], fill=(255, 0, 0), width=3)
        return img

    def infer_points(self, rgb_crop: np.ndarray, poly_xy: list[tuple[int, int]]) -> dict | None:
        img = Image.fromarray(rgb_crop)
        img = self.overlay_polygon(img, poly_xy)

        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode
        tfm = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        pixel_values = tfm(img).unsqueeze(0).to(self.dev, dtype=self.model.dtype)
        prompt = points_prompt()

        with torch.no_grad():
            # ✅ Correct call for InternVL2 models
            out = self.model.chat(
                self.tok,
                pixel_values,
                prompt,
                dict(max_new_tokens=self.max_new_tokens, do_sample=False)
            )

        matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", out)
        if len(matches) < 8:
            return None

        coords = [(int(x), int(y)) for x, y in matches[:8]]
        return {"inside": coords[:4], "outside": coords[4:8], "raw": out}
