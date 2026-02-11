import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


class SAMBuildingDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.points_dir = os.path.join(root_dir, "points")

        self.files = sorted(os.listdir(self.img_dir))[:1000]

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        image = cv2.imread(os.path.join(self.img_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask = (mask > 0).astype(np.float32)

        with open(os.path.join(self.points_dir, name.replace(".png", ".json"))) as f:
            pts = json.load(f)

        positive = pts["positive_points"]
        negative = pts["negative_points"]

        augmented = self.aug(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).unsqueeze(0)

        points = positive + negative
        labels = [1]*len(positive) + [0]*len(negative)

        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, mask, points, labels
