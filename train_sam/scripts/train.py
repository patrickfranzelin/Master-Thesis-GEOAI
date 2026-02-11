import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from dataset import SAMBuildingDataset
from losses import DiceLoss
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = PROJECT_ROOT / "train_sam" / "patches"

CHECKPOINT_PATH = r"C:\git\Master-Thesis-GEOAI\models\sam3_weights\sam_vit_b_01ec64.pth"

print("Loading checkpoint from:", CHECKPOINT_PATH)

sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
sam.to(DEVICE)

# Freeze encoder + prompt encoder
for p in sam.image_encoder.parameters():
    p.requires_grad = False

for p in sam.prompt_encoder.parameters():
    p.requires_grad = False

sam.mask_decoder.train()

dataset = SAMBuildingDataset(root_dir=str(PATCH_ROOT))
loader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(
    sam.mask_decoder.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

EPOCHS = 15

for epoch in range(EPOCHS):

    sam.train()
    total_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for images, masks, points, labels in progress_bar:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        points = points.to(DEVICE)
        labels = labels.to(DEVICE)

        image_embeddings = sam.image_encoder(images)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None
        )

        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        pred_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False
        )

        loss = bce(pred_masks, masks) + dice(pred_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss/(progress_bar.n+1):.4f}"
        })

    print(f"Epoch {epoch+1} finished | Mean Loss: {total_loss/len(loader):.4f}\n")

torch.save(sam.mask_decoder.state_dict(), "../../models/sam2_weights/sam_building_decoder_finetuned.pth")
print("Training finished.")
