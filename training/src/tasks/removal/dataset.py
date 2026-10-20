"""
WatermarkDataset
----------------
Each sample directory contains:
  watermarked.jpg  – degraded image with semi-transparent watermark
  clean.png        – original clean image (lossless ground truth)
  mask.png         – soft alpha mask (255 = fully watermarked, 0 = clean,
                     0-255 = feathered transition zone at the edge)
  meta.json        – blend_mode ("srgb" | "linear"), position metadata

Model input  : 3-ch RGB tensor in [-1, 1] (blind removal — no mask/gradient input)
Model target : clean RGB tensor in [-1, 1]

Two mask tensors are distinguished:
  mask       – original soft mask, Gaussian-blurred for loss weighting.
               The blur creates a smooth edge zone so edge-focused loss terms
               (border, masked-L1) receive gradient information across the
               transition region rather than a hard cutoff.
  mask_input – not used for blind removal; kept as None in the batch dict
               for compatibility with any code that expects the key.

Batch keys returned:
  "input"      – 3×H×W model input tensor
  "target"     – 3×H×W clean RGB tensor in [-1, 1]
  "mask"       – 1×H×W blurred soft mask for loss weighting
  "blend_mode" – scalar int64 tensor: 0 = sRGB, 1 = linear

Note: old datasets with binary masks (0/255 only) are fully compatible —
they load as {0.0, 1.0} float32 values, same as before.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from wm_shared.preprocess import (
    blur_mask_for_loss,
    crop_removal_roi,
)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

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

def _random_shift(wm: np.ndarray,
                  clean: np.ndarray,
                  mask: np.ndarray,
                  shift_px: int = 5):
    """
    Small random translation applied to all tensors identically.

    Jiggles the ROI crop by up to *shift_px* pixels so the model cannot
    rely on absolute pixel coordinates to locate the watermark, while
    preserving the exact watermark geometry (no mirroring or flipping).
    Reflect-padding keeps edges clean.
    """
    h, w = wm.shape[:2]
    dx = random.randint(-shift_px, shift_px)
    dy = random.randint(-shift_px, shift_px)
    if dx == 0 and dy == 0:
        return wm, clean, mask
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    kwargs = dict(borderMode=cv2.BORDER_REFLECT_101)
    wm    = cv2.warpAffine(wm,    m, (w, h), **kwargs)
    clean = cv2.warpAffine(clean, m, (w, h), **kwargs)
    mask  = cv2.warpAffine(mask,  m, (w, h), flags=cv2.INTER_NEAREST, **kwargs)
    return wm, clean, mask


def _jpeg_augment(wm: np.ndarray) -> np.ndarray:
    """
    Re-encode the watermarked image at a random JPEG quality (30–92).

    The watermarked images are stored as .jpg already. Randomly varying the
    effective quality during training teaches the model to separate watermark
    signal from JPEG blocking/ringing artefacts, which occupy the same spatial
    frequency band as the per-pixel white overlay correction. Applied only to
    the watermarked image — the clean ground truth stays lossless.
    """
    quality = random.randint(30, 92)
    _, encoded = cv2.imencode('.jpg', wm, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _webp_augment(wm: np.ndarray) -> np.ndarray:
    """
    Re-encode the watermarked image at a random WebP quality (20–92).

    Real-world images are often delivered as WebP with aggressive compression.
    WebP artefacts differ from JPEG (smoother edges, less ringing), and the
    model must learn to generalise across both codecs. Applied only to the
    watermarked image — the clean ground truth stays lossless.
    """
    quality = random.randint(20, 92)
    _, encoded = cv2.imencode('.webp', wm, [cv2.IMWRITE_WEBP_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────────────────────────────────────

class WatermarkDataset(Dataset):
    def __init__(self,
                 samples: list[Path],
                 image_width: int = 128,
                 image_height: int = 128,
                 loss_mask_blur_pct: float = 0.0,
                 training: bool = True,
                 augment: bool = True,
                 crop_aspect_ratio: float = 3.54,
                 crop_margin_ratio: float = 0.10,
                 crop_min_width_ratio: float = 0.50):
        self.samples = samples
        self.image_width = image_width
        self.image_height = image_height
        self.loss_mask_blur_pct = loss_mask_blur_pct
        self.training = training
        self.augment = augment
        self.crop_aspect_ratio = crop_aspect_ratio
        self.crop_margin_ratio = crop_margin_ratio
        self.crop_min_width_ratio = crop_min_width_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]

        wm = self._read_image_bgr(d / "watermarked.jpg")
        clean = self._read_image_bgr(d / "clean.png")
        mask = self._read_image_mask(d / "mask.png")

        wm, clean, mask, _, _ = crop_removal_roi(
            wm,
            clean,
            mask,
            self.image_width,
            self.image_height,
            crop_aspect_ratio=self.crop_aspect_ratio,
            margin_ratio=self.crop_margin_ratio,
            min_width_ratio=self.crop_min_width_ratio,
        )

        # Read blend mode from metadata so the anchor delta loss can use
        # mode-specific physics. Defaults to sRGB (0) when meta.json is absent.
        blend_mode = 0
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                blend_mode = 1 if meta.get("blend_mode") == "linear" else 0
            except Exception:
                pass

        if self.training and self.augment:
            shift_px = max(1, int(self.image_width * 0.01))
            wm, clean, mask = _random_shift(wm, clean, mask, shift_px=shift_px)
            if random.random() < 0.5:
                wm = random.choice([_jpeg_augment, _webp_augment])(wm)

        # loss mask: Gaussian-blur so edge-focused loss terms get a smooth gradient.
        mask_loss = blur_mask_for_loss(mask, self.loss_mask_blur_pct, (self.image_width, self.image_height))

        wm_t        = _to_tensor_rgb(wm)          # 3xHxW in [-1,1]
        clean_t     = _to_tensor_rgb(clean)       # 3xHxW in [-1,1]
        mask_loss_t = _to_tensor_mask(mask_loss)  # 1xHxW in [0,1] (blurred, for loss)
        mask_raw_t  = _to_tensor_mask(mask)       # 1xHxW in [0,1] (raw for loss geometry)

        return {
            "wm":         wm_t,          # 3xHxW
            "target":     clean_t,       # 3xHxW
            "mask_loss":  mask_loss_t,   # 1xHxW (blurred for criterion)
            "mask_raw":   mask_raw_t,    # 1xHxW (raw for loss geometry)
            "blend_mode": torch.tensor(blend_mode, dtype=torch.long),
        }


    # ── internal loading ───────────────────────────────────────────────────

    def _read_image_bgr(self, path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        return img

    def _read_image_mask(self, path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        mask = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read {path}")
        return mask.astype(np.float32) / 255.0


def make_splits(root: str, image_width: int, image_height: int, train_frac: float = 0.9,
                seed: int = 42, loss_mask_blur_pct: float = 0.0,
                max_samples: int | None = None,
                augment: bool = True,
                crop_aspect_ratio: float = 3.54,
                crop_margin_ratio: float = 0.10,
                crop_min_width_ratio: float = 0.50):
    """Return (train_subset, val_subset) with aug enabled only on train."""
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
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(all_samples), generator=rng).tolist()

    if max_samples is not None and train_frac >= 1.0:
        n_train = min(max_samples, n)
        n_val = min(max_samples, n - n_train)
        train_samples = [all_samples[i] for i in idx[:n_train]]
        val_samples   = [all_samples[i] for i in idx[n_train:n_train + n_val]]
        if not val_samples:
            val_samples = train_samples
    else:
        n_pool = n if max_samples is None else min(n, max_samples)
        n_tr = int(n_pool * train_frac)
        pool_idx = idx[:n_pool]
        train_samples = [all_samples[i] for i in pool_idx[:n_tr]]
        val_samples   = [all_samples[i] for i in pool_idx[n_tr:]]
        if not val_samples:
            val_samples = train_samples

    tr_ds = WatermarkDataset(
        samples=train_samples,
        image_width=image_width,
        image_height=image_height,
        loss_mask_blur_pct=loss_mask_blur_pct,
        training=True,
        augment=augment,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )
    va_ds = WatermarkDataset(
        samples=val_samples,
        image_width=image_width,
        image_height=image_height,
        loss_mask_blur_pct=loss_mask_blur_pct,
        training=False,
        augment=False,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )

    return tr_ds, va_ds
