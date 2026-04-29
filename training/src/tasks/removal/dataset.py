"""
WatermarkDataset
----------------
Each sample directory contains:
  watermarked.jpg  – degraded image with semi-transparent watermark
  clean.png        – original clean image (lossless ground truth)
  mask.png         – soft alpha mask (255 = fully watermarked, 0 = clean,
                     0-255 = feathered transition zone at the edge)
  meta.json        – blend_mode ("srgb" | "linear"), position metadata

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

Batch keys returned:
  "input"      – 5×H×W model input tensor
  "target"     – 3×H×W clean RGB tensor in [-1, 1]
  "mask"       – 1×H×W blurred soft mask for loss weighting
  "blend_mode" – scalar int64 tensor: 0 = sRGB, 1 = linear

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

from wm_shared.preprocess import (
    compute_gradient,
    dilate_mask_input,
    blur_mask_for_loss,
    crop_removal_roi,
)
from ...common.restoration import build_store_signature, load_or_build_preprocessed_sample


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

def _augment(wm: np.ndarray,
             clean: np.ndarray,
             mask: np.ndarray,
             mask_input: np.ndarray | None = None):
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
        if mask_input is not None:
            mask_input = cv2.flip(mask_input, 1)
    if random.random() < 0.3:
        wm    = cv2.flip(wm,    0)
        clean = cv2.flip(clean, 0)
        mask  = cv2.flip(mask,  0)
        if mask_input is not None:
            mask_input = cv2.flip(mask_input, 0)

    # brightness / contrast jitter — applied identically to both images so the
    # model still sees a consistent (watermarked → clean) pair, just under
    # different global exposure. The watermark signal is relative, so this is safe.
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)   # contrast
        beta  = random.uniform(-15,  15)      # brightness (out of 255)
        wm    = cv2.convertScaleAbs(wm,    alpha=alpha, beta=beta)
        clean = cv2.convertScaleAbs(clean, alpha=alpha, beta=beta)
        # mask is geometry-only — not adjusted

    return wm, clean, mask, mask_input


def _jpeg_augment(wm: np.ndarray) -> np.ndarray:
    """
    Re-encode the watermarked image at a random JPEG quality (70–92).

    The watermarked images are stored as .jpg already. Randomly varying the
    effective quality during training teaches the model to separate watermark
    signal from JPEG blocking/ringing artefacts, which occupy the same spatial
    frequency band as the per-pixel white overlay correction. Applied only to
    the watermarked image — the clean ground truth stays lossless.
    """
    quality = random.randint(70, 92)
    _, encoded = cv2.imencode('.jpg', wm, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────────────────────────────────────

class WatermarkDataset(Dataset):
    def __init__(self,
                 samples: list[Path],
                 dataset_root: str | Path,
                 preprocessed_store_root: str | Path | None,
                 preprocessed_store_signature: str | None,
                 image_width: int = 128,
                 image_height: int = 128,
                 loss_mask_blur_pct: float = 0.0,
                 training: bool = True,
                 crop_aspect_ratio: float = 3.54,
                 crop_margin_ratio: float = 0.10,
                 crop_min_width_ratio: float = 0.50):
        self.samples = samples
        self.dataset_root = Path(dataset_root)
        self.preprocessed_store_root = Path(preprocessed_store_root) if preprocessed_store_root is not None else None
        self.preprocessed_store_signature = preprocessed_store_signature
        self.image_width = image_width
        self.image_height = image_height
        self.loss_mask_blur_pct = loss_mask_blur_pct
        self.training = training
        self.crop_aspect_ratio = crop_aspect_ratio
        self.crop_margin_ratio = crop_margin_ratio
        self.crop_min_width_ratio = crop_min_width_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def _load_preprocessed_sample(self, sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self.preprocessed_store_root is None or self.preprocessed_store_signature is None:
            return None
        return load_or_build_preprocessed_sample(
            store_root=self.preprocessed_store_root,
            dataset_root=self.dataset_root,
            sample_dir=sample_dir,
            signature=self.preprocessed_store_signature,
            image_width=self.image_width,
            image_height=self.image_height,
            crop_aspect_ratio=self.crop_aspect_ratio,
            crop_margin_ratio=self.crop_margin_ratio,
            crop_min_width_ratio=self.crop_min_width_ratio,
            use_augmented_mask=self.training,
        )

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]

        cached = self._load_preprocessed_sample(d)
        if cached is not None:
            wm, clean, mask, mask_input = cached
        else:
            wm = self._read_image_bgr(d / "watermarked.jpg")
            clean = self._read_image_bgr(d / "clean.png")
            mask = self._read_image_mask(d / "mask.png")

            wm, clean, mask, _, mask_input = crop_removal_roi(
                wm,
                clean,
                mask,
                self.image_width,
                self.image_height,
                crop_aspect_ratio=self.crop_aspect_ratio,
                margin_ratio=self.crop_margin_ratio,
                min_width_ratio=self.crop_min_width_ratio,
                use_augmented_mask=False,
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
                pass  # malformed meta — fall back to sRGB default

        if self.training:
            wm, clean, mask, mask_input = _augment(wm, clean, mask, mask_input)
            # JPEG re-compression: teaches the model to separate watermark signal
            # from JPEG blocking/ringing artefacts. Applied after flips/jitter so
            # the augmented exposure still sees fresh compression noise.
            if random.random() < 0.5:
                wm = _jpeg_augment(wm)

        # loss mask: Gaussian-blur so edge-focused loss terms get a smooth gradient.
        mask_loss = blur_mask_for_loss(mask, self.loss_mask_blur_pct, (self.image_width, self.image_height))

        wm_t         = _to_tensor_rgb(wm)          # 3xHxW in [-1,1]
        clean_t      = _to_tensor_rgb(clean)       # 3xHxW in [-1,1]
        mask_loss_t  = _to_tensor_mask(mask_loss)  # 1xHxW in [0,1] (blurred, for loss)
        mask_raw_t   = _to_tensor_mask(mask)       # 1xHxW in [0,1] (raw for loss geometry)
        mask_input_t = _to_tensor_mask(mask_input) # 1xHxW in [0,1] (model hint)

        return {
            "wm":         wm_t,          # 3xHxW
            "target":     clean_t,       # 3xHxW
            "mask_loss":  mask_loss_t,   # 1xHxW (blurred for criterion)
            "mask_raw":   mask_raw_t,    # 1xHxW (raw for GPU preprocessing)
            "mask_input": mask_input_t,  # 1xHxW (precomputed model hint)
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
                crop_aspect_ratio: float = 3.54,
                crop_margin_ratio: float = 0.10,
                crop_min_width_ratio: float = 0.50,
                preprocessed_store_root: str | Path | None = None,
                preprocessed_store_signature: str | None = None):
    """Return (train_subset, val_subset) with aug enabled only on train."""
    from torch.utils.data import Subset
    # If a signature was passed, we might want to override it with train/val variants
    # if it doesn't already have the -aug suffix.
    # But cleaner to just rebuild it here if we have all the params.
    train_signature = build_store_signature(
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=True,
    )
    val_signature = build_store_signature(
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=False,
    )

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
    val_samples   = [all_samples[i] for i in idx[n_tr:]]

    tr_ds = WatermarkDataset(
        samples=train_samples,
        dataset_root=root,
        preprocessed_store_root=preprocessed_store_root,
        preprocessed_store_signature=train_signature,
        image_width=image_width,
        image_height=image_height,
        loss_mask_blur_pct=loss_mask_blur_pct,
        training=True,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )
    va_ds = WatermarkDataset(
        samples=val_samples,
        dataset_root=root,
        preprocessed_store_root=preprocessed_store_root,
        preprocessed_store_signature=val_signature,
        image_width=image_width,
        image_height=image_height,
        loss_mask_blur_pct=loss_mask_blur_pct,
        training=False,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )

    return tr_ds, va_ds
