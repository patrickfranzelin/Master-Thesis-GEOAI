from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# ==========================================
#  InternVL model setup
# ==========================================
model_id = "OpenGVLab/InternVL2-4B"   # or "OpenGVLab/InternVL3-8B"
print(f"🔄 Loading model: {model_id}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧮 Using device: {device}")

# Load processor & model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
).eval()

# ==========================================
#  Load image
# ==========================================
image_path = "/data/Master-Thesis-GEOAI/Theory/img.png"
print(f"🖼️ Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")

# ==========================================
#  Define prompt
# ==========================================
prompt = (
    "You are a visual quality evaluator. "
    "Task: Decide if the red polygon precisely matches the building in the aerial image. "
    "Respond strictly in the following format:\n"
    "Answer: [Yes or No]\n"
    "Reason: [one short sentence explaining why].\n"
    "Do not describe the image beyond this."
)

generation_config = dict(
    max_new_tokens=40,
    do_sample=False,
    temperature=0.1,
    top_p=0.9
)


# ==========================================
#  Inference (Chat-style for InternVL2)
# ==========================================
print("🚀 Running inference...")

# Run chat directly — handles both text + image inputs
response = model.chat(
    processor=processor,
    image=image,
    question=prompt,
    history=None,
    generation_config=dict(max_new_tokens=80)
)


print("\n🧠 Model answer:")
print("----------------------------------------------------")
print(response)
print("----------------------------------------------------")
