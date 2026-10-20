"""
precompute_raw_prob_masks.py
============================
Pre-computes raw segmentation probability masks for removal training.

The script runs the segmentation model on each dataset sample, crops the result
with the same ROI logic used by removal training, and stores the raw
probability mask directly as the model hint. No mask alignment, template
placement, or jitter augmentation is applied.

Per-sample cache contents:
  watermarked.jpg  - ROI-cropped, resized watermarked image
  clean.png        - ROI-cropped, resized clean image
  mask.png         - ROI-cropped, resized GT loss mask
  mask_input.npy   - raw probability mask from the segmentation model
  mask_input.png   - preview of the cached probability mask
  cache_meta.json  - provenance information for the cache entry
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.common.restoration import (
    STORE_VERSION,
    build_store_signature,
    clear_preprocessed_store,
    get_store_root,
    iter_sample_dirs,
    write_preprocessed_sample,
)
from training.src.tasks.segmentation.model import build_seg_model
from wm_shared.config import load_yaml_config, validate_removal_config
from wm_shared.preprocess import crop_removal_roi


def _load_seg_model(seg_cfg: dict, checkpoint_path: Path, device: torch.device):
    model = build_seg_model(seg_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def _predict_probability_mask(seg_model, wm_bgr: np.ndarray, size: int, device: torch.device) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = cv2.resize(wm_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    inp = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    prob = torch.sigmoid(seg_model(inp)).squeeze().detach().cpu().numpy().astype(np.float32)
    h, w = wm_bgr.shape[:2]
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def _build_sample(
    sample_dir: Path,
    seg_model,
    seg_size: int,
    device: torch.device,
    image_width: int,
    image_height: int,
    crop_aspect_ratio: float,
    crop_margin_ratio: float,
    crop_min_width_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wm_bgr = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
    clean_bgr = cv2.imread(str(sample_dir / "clean.png"), cv2.IMREAD_COLOR)
    gt_mask_gray = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (wm_bgr, clean_bgr, gt_mask_gray)):
        raise FileNotFoundError(f"Missing image files in {sample_dir}")

    prob_mask = _predict_probability_mask(seg_model, wm_bgr, seg_size, device)
    wm_r, clean_r, gt_mask_r, _, mask_input_r = crop_removal_roi(
        wm_bgr,
        clean_bgr,
        prob_mask,
        image_width,
        image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=crop_margin_ratio,
        min_width_ratio=crop_min_width_ratio,
    )
    if gt_mask_r.max() > 1.01:
        gt_mask_r = gt_mask_r / 255.0

    return wm_r, clean_r, gt_mask_r.astype(np.float32), mask_input_r.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute raw segmentation mask cache for removal training")
    parser.add_argument("--config", required=True, help="Removal training config YAML")
    parser.add_argument("--seg-checkpoint", required=True, help="Segmentation checkpoint (.pth)")
    parser.add_argument("--seg-config", default="training/configs/seg.yaml", help="Segmentation config YAML")
    parser.add_argument("--store-dir", default=None, help="Override preprocessed store root")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples (0 = all)")
    parser.add_argument("--clear", action="store_true", help="Delete existing store entries before building")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = Path(__file__).resolve().parent
    removal_cfg = validate_removal_config(load_yaml_config(config_path))
    ds_cfg = removal_cfg["dataset"]

    seg_cfg_path = Path(args.seg_config).resolve()
    seg_cfg = load_yaml_config(seg_cfg_path)
    seg_size = int(seg_cfg["dataset"]["image_size"])

    image_width = int(ds_cfg["image_width"])
    image_height = int(ds_cfg["image_height"])
    crop_aspect_ratio = float(ds_cfg.get("crop_aspect_ratio", 3.54))
    crop_margin_ratio = float(ds_cfg.get("crop_margin_ratio", 0.10))
    crop_min_width_ratio = float(ds_cfg.get("crop_min_width_ratio", 0.50))

    dataset_root = (config_dir / ds_cfg["root"]).resolve()
    raw_store = args.store_dir or ds_cfg.get("preprocessed_store_dir")
    store_root = get_store_root(dataset_root, (config_dir / raw_store).resolve() if raw_store else None)
    signature = build_store_signature(
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )

    print(f"Dataset:     {dataset_root}")
    print(f"Store:       {store_root / signature}")
    print(f"Resolution:  {image_width}x{image_height}")

    wm_path = Path(args.seg_checkpoint).resolve()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seg_model = _load_seg_model(seg_cfg, wm_path, device)
    print(f"Seg model:   {wm_path}  (device={device})")

    samples = iter_sample_dirs(dataset_root)
    max_samples = ds_cfg.get("max_samples")
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_root}")

    if args.clear:
        print("Clearing existing store ...")
        clear_preprocessed_store(store_root, signature)

    cache_meta = {
        "cache_version": STORE_VERSION,
        "cache_type": "raw_probability",
        "seg_checkpoint": str(wm_path),
        "seg_config": str(seg_cfg_path),
        "seg_image_size": seg_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    total = len(samples)
    print(f"Processing {total} samples ...\n")
    for idx, sample_dir in enumerate(samples, start=1):
        wm_r, clean_r, gt_mask_r, mask_input_r = _build_sample(
            sample_dir=sample_dir,
            seg_model=seg_model,
            seg_size=seg_size,
            device=device,
            image_width=image_width,
            image_height=image_height,
            crop_aspect_ratio=crop_aspect_ratio,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_width_ratio=crop_min_width_ratio,
        )
        write_preprocessed_sample(
            store_root=store_root,
            dataset_root=dataset_root,
            sample_dir=sample_dir,
            signature=signature,
            wm=wm_r,
            clean=clean_r,
            mask=gt_mask_r,
            mask_input=mask_input_r,
            cache_meta=cache_meta,
        )
        if idx == 1 or idx == total or idx % 100 == 0:
            print(f"[{idx}/{total}] prepared {sample_dir.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
