"""
WatermarkDataset
----------------
Each sample directory contains:
  watermarked.jpg  – degraded image with semi-transparent watermark
  clean.png        – original clean image (lossless ground truth)
  mask.png         – soft alpha mask (255 = fully watermarked, 0 = clean,
                     0-255 = feathered transition zone at the edge)
  meta.json        – blend_mode, position (not used during training)

Model input  : 5-ch tensor in [-1, 1] / [0, 1]
               (RGB watermarked, mask_input, grayscale_gradient)
Model target : clean RGB tensor in [-1, 1]

Two mask tensors are distinguished:
  mask       – original soft mask, Gaussian-blurred for loss weighting.
               The blur creates a smooth edge zone so edge-focused loss terms
               (border, masked-L1) receive gradient information across the
               transition region rather than a hard cutoff.
  mask_input – dilated binary mask fed to the model as guidance.  Dilation
               guarantees the true watermark edge is always *inside* the mask,
               so the model never has to correct pixels it wasn't told about.

Note: old datasets with binary masks (0/255 only) are fully compatible —
they load as {0.0, 1.0} float32 values, same as before.
"""

import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.image_utils import (
    compute_gradient,
    dilate_mask_input,
    dilate_mask_for_loss,
    blur_mask_for_loss,
)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_bgr(path: str, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img  # uint8, BGR, HxWx3


def _read_mask(path: str, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # Linear interpolation preserves the soft alpha gradient at watermark edges.
    # Old binary masks (only 0/255) round-trip correctly: /255 → {0.0, 1.0}.
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = mask.astype(np.float32) / 255.0  # HxW, float32 in [0, 1]
    return mask


def _to_tensor_rgb(bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR HxWx3 → float32 CxHxW in [-1, 1]"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # CxHxW


def _to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    """float32 HxW → float32 1xHxW in {0, 1}"""
    return torch.from_numpy(mask).unsqueeze(0)



# ──────────────────────────────────────────────────────────────────────────────
# augmentation (applied jointly to watermarked, clean, and mask)
# ──────────────────────────────────────────────────────────────────────────────

def _augment(wm: np.ndarray, clean: np.ndarray, mask: np.ndarray):
    """
    Joint augmentation applied identically to watermarked, clean, and mask.

    - Horizontal flip          : doubles effective dataset size for free
    - Vertical flip            : valid for photos (adds variety without distortion)
    - Brightness/contrast jitter: teaches the model that lighting can vary, reduces
                                  colour overfitting to the specific training images
    """
    # flips
    if random.random() < 0.5:
        wm    = cv2.flip(wm,    1)
        clean = cv2.flip(clean, 1)
        mask  = cv2.flip(mask,  1)
    if random.random() < 0.3:
        wm    = cv2.flip(wm,    0)
        clean = cv2.flip(clean, 0)
        mask  = cv2.flip(mask,  0)

    # brightness / contrast jitter — applied identically to both images so the
    # model still sees a consistent (watermarked → clean) pair, just under
    # different global exposure. The watermark signal is relative, so this is safe.
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)   # contrast
        beta  = random.uniform(-15,  15)      # brightness (out of 255)
        wm    = cv2.convertScaleAbs(wm,    alpha=alpha, beta=beta)
        clean = cv2.convertScaleAbs(clean, alpha=alpha, beta=beta)
        # mask is geometry-only — not adjusted

    return wm, clean, mask


# ──────────────────────────────────────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────────────────────────────────────

class WatermarkDataset(Dataset):
    def __init__(self,
                 samples: list[Path],
                 image_size: int = 128,
                 loss_mask_blur_pct: float = 0.0,
                 loss_mask_dilate_pct: float = 0.0):
        self.samples = samples
        self.image_size = image_size
        self.loss_mask_blur_pct = loss_mask_blur_pct
        self.loss_mask_dilate_pct = loss_mask_dilate_pct

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]

        wm    = _read_bgr(d / "watermarked.jpg", self.image_size)
        clean = _read_bgr(d / "clean.png",       self.image_size)
        mask  = _read_mask(d / "mask.png",        self.image_size)

        wm, clean, mask = _augment(wm, clean, mask)

        # mask_input: dilated binary mask fed to the model — ensures the true
        # watermark edge is always inside the mask boundary.
        mask_input = dilate_mask_input(mask)

        # loss mask: Dilate to cover compression footprint, then Gaussian-blur 
        # so edge-focused loss terms get a smooth gradient.
        mask_loss_base = dilate_mask_for_loss(mask, self.loss_mask_dilate_pct, self.image_size)
        mask_loss = blur_mask_for_loss(mask_loss_base, self.loss_mask_blur_pct, self.image_size)

        # explicit boundary signal
        grad_t = compute_gradient(wm)             # 1xHxW in [0,1]

        wm_t         = _to_tensor_rgb(wm)          # 3xHxW in [-1,1]
        clean_t      = _to_tensor_rgb(clean)        # 3xHxW in [-1,1]
        mask_t       = _to_tensor_mask(mask_loss)   # 1xHxW in [0,1] (blurred, for loss)
        mask_input_t = _to_tensor_mask(mask_input)  # 1xHxW in [0,1] (dilated, for model)

        # 5-channel network input: RGB watermarked + dilated mask + gradient field
        inp = torch.cat([wm_t, mask_input_t, grad_t], dim=0)  # 5xHxW

        return {"input": inp, "target": clean_t, "mask": mask_t}


# ──────────────────────────────────────────────────────────────────────────────
# split utility
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(root: str, image_size: int, train_frac: float = 0.9,
                seed: int = 42, loss_mask_blur_pct: float = 0.0,
                loss_mask_dilate_pct: float = 0.0,
                max_samples: int | None = None):
    """Return (train_subset, val_subset) with aug enabled only on train."""
    from torch.utils.data import Subset

    all_samples = sorted([
        d for d in Path(root).iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "clean.png").exists()
        and (d / "mask.png").exists()
    ])
    if not all_samples:
        raise RuntimeError(f"No valid samples found in {root}")

    n = len(all_samples)
    if max_samples is not None:
        n = min(n, max_samples)

    n_tr = int(n * train_frac)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(all_samples), generator=rng).tolist()[:n]

    train_samples = [all_samples[i] for i in idx[:n_tr]]
    val_samples = [all_samples[i] for i in idx[n_tr:]]

    tr_ds = WatermarkDataset(
        samples=train_samples,
        image_size=image_size,
        loss_mask_blur_pct=loss_mask_blur_pct,
        loss_mask_dilate_pct=loss_mask_dilate_pct
    )
    va_ds = WatermarkDataset(
        samples=val_samples,
        image_size=image_size,
        loss_mask_blur_pct=loss_mask_blur_pct,
        loss_mask_dilate_pct=loss_mask_dilate_pct
    )

    return tr_ds, va_ds
