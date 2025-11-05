import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ============================================================
#  CONFIGURATION
# ============================================================
model_id = "OpenGVLab/InternVL3-8B"
image_path = "/data/Master-Thesis-GEOAI/Theory/img.png"
question = (
    "<image>\nYou are a visual evaluator. "
    "Does the red polygon precisely match the building in the image? "
    "Respond strictly in this format:\n"
    "Answer: [Yes or No]\n"
    "Reason: [one short sentence explaining why]."
)


# ============================================================
#  LOAD MODEL
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
#  IMAGE PREPROCESSING
# ============================================================
print(f"🖼️ Loading image: {image_path}")

transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

image = Image.open(image_path)
pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)

# ============================================================
#  INFERENCE
# ============================================================
print("🚀 Running inference...")
generation_config = dict(max_new_tokens=256, do_sample=False)

response = model.chat(tokenizer, pixel_values, question, generation_config)
print("\n🧠 Model answer:")
print("----------------------------------------------------")
print(response)
print("----------------------------------------------------")
