"""
WatermarkSegDataset
-------------------
Each sample directory contains:
  watermarked.jpg  – degraded image with semi-transparent watermark
  mask.png         – soft alpha mask (255 = fully watermarked, 0 = clean)

The segmentation model learns to predict the soft mask from the watermarked
image alone.  The soft alpha is used as-is as the training target — not
binarized — so the model learns feathered edges as accurately as possible.
At inference the output is thresholded at 0.5 to produce a binary mask.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _augment(img: np.ndarray, mask: np.ndarray):
    """Joint spatial + photometric augmentation."""
    if random.random() < 0.5:
        img  = cv2.flip(img,  1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.3:
        img  = cv2.flip(img,  0)
        mask = cv2.flip(mask, 0)
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta  = random.uniform(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img, mask


class WatermarkSegDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment

        self.samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
            and (d / "watermarked.jpg").exists()
            and (d / "mask.png").exists()
        ])
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d    = self.samples[idx]
        size = self.image_size

        img  = cv2.imread(str(d / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(d / "mask.png"),        cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {d / 'watermarked.jpg'}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read {d / 'mask.png'}")

        img  = cv2.resize(img,  (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            img, mask = _augment(img, mask)

        # image: BGR → RGB, uint8 → float32 in [-1, 1]
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(rgb.transpose(2, 0, 1))                   # 3xHxW

        # mask: uint8 → float32 in [0, 1]  (soft target, not binarized)
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)  # 1xHxW

        return {"input": img_t, "target": mask_t}
