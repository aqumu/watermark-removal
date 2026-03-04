"""
WatermarkDataset
----------------
Each sample directory contains:
  watermarked.jpg  – degraded image with semi-transparent watermark
  clean.png        – original clean image (lossless ground truth)
  mask.png         – soft alpha mask (255 = fully watermarked, 0 = clean,
                     0-255 = feathered transition zone at the edge)
  meta.json        – blend_mode, position (not used during training)

Model input  : (watermarked RGB, mask_input) → 4-ch tensor in [-1, 1] / [0, 1]
Model target : clean RGB tensor in [-1, 1]

Two mask tensors are distinguished:
  mask       – original soft mask used for loss weighting (accurate)
  mask_input – randomly degraded version fed to the model as guidance

mask_input augmentation teaches the model that the provided mask is an
approximation: sometimes it is the ideal soft alpha, sometimes a binary
threshold, sometimes a blurred binary (simulating a manually drawn mask).
This prevents the model from depending on a perfect soft mask at inference.

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
# mask input augmentation
# ──────────────────────────────────────────────────────────────────────────────

def _augment_mask_input(mask: np.ndarray) -> np.ndarray:
    """
    Convert the soft mask to an approximate binary mask before feeding it to
    the model as its 4th input channel.

    The soft mask is never passed directly to the model.  If it were, the
    model would learn to scale its correction by the mask value (e.g. apply
    only 40% correction at an edge pixel with mask=0.4), producing a visible
    bright ring that mirrors the alpha gradient.  By always presenting a
    binary-style mask, the model learns to apply full correction to every
    pixel it is told to fix, regardless of how faint the watermark is there.

    Two equally-likely variants:
      binary         – hard threshold at 0.5
      blurred binary – threshold then Gaussian blur (simulates a
                       hand-drawn or segmentation-model mask)

    The mask used for loss weighting is NOT touched by this function.

    mask : HxW float32 in [0, 1]
    returns HxW float32 in [0, 1]
    """
    binary = (mask >= 0.5).astype(np.float32)
    if random.random() < 0.5:
        return binary
    # blurred binary — mimics softened edges from a manual or predicted mask
    sigma = random.uniform(1.0, 4.0)
    blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
    return np.clip(blurred, 0.0, 1.0)


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
    def __init__(self, root: str, image_size: int = 512, augment: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment

        # collect all sample dirs that have the required files
        self.samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
            and (d / "watermarked.jpg").exists()
            and (d / "clean.png").exists()
            and (d / "mask.png").exists()
        ])
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]

        wm    = _read_bgr(d / "watermarked.jpg", self.image_size)
        clean = _read_bgr(d / "clean.png",       self.image_size)
        mask  = _read_mask(d / "mask.png",        self.image_size)

        if self.augment:
            wm, clean, mask = _augment(wm, clean, mask)

        # mask_input: degraded version fed to the model — teaches robustness to
        # imperfect real-world masks.  mask stays as-is for loss weighting.
        mask_input = _augment_mask_input(mask) if self.augment else mask

        wm_t         = _to_tensor_rgb(wm)          # 3xHxW in [-1,1]
        clean_t      = _to_tensor_rgb(clean)        # 3xHxW in [-1,1]
        mask_t       = _to_tensor_mask(mask)        # 1xHxW in [0,1] (soft, for loss)
        mask_input_t = _to_tensor_mask(mask_input)  # 1xHxW in [0,1] (degraded, for model)

        # 4-channel network input: RGB watermarked + degraded mask
        inp = torch.cat([wm_t, mask_input_t], dim=0)  # 4xHxW

        return {"input": inp, "target": clean_t, "mask": mask_t}


# ──────────────────────────────────────────────────────────────────────────────
# split utility
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(root: str, image_size: int, train_frac: float = 0.9, seed: int = 42):
    """Return (train_subset, val_subset) with aug enabled only on train."""
    from torch.utils.data import Subset

    tr_ds = WatermarkDataset(root, image_size=image_size, augment=True)
    va_ds = WatermarkDataset(root, image_size=image_size, augment=False)

    n     = len(tr_ds)
    n_tr  = int(n * train_frac)
    rng   = torch.Generator().manual_seed(seed)
    idx   = torch.randperm(n, generator=rng).tolist()

    return Subset(tr_ds, idx[:n_tr]), Subset(va_ds, idx[n_tr:])
