"""
precompute_aligned_masks.py
============================
Pre-computes mask_input for the removal model training dataset by running the
full watermark-service alignment pipeline (segmentation → subpixel_xcorr → 5px dilation)
on every training sample and saving the result to the preprocessed store.

This replaces the old artificial jitter augmentation with masks that exactly
match what the service produces at inference time, making training and real-world
usage identical.

The per-sample result written to the store:
  watermarked.jpg  – ROI-cropped, resized watermarked image
  clean.png        – ROI-cropped, resized clean image
  mask.png         – ROI-cropped, resized GT loss mask (from mask.png)
  mask_input.png   – aligned template + 5px ELLIPSE dilation (model hint)

The ROI is determined from the aligned template mask (matching the service),
while loss weighting uses the ground-truth mask.

Usage:
    python training/precompute_aligned_masks.py \\
        --config        training/configs/train_512.yaml \\
        --seg-checkpoint C:/path/to/model_seg_2.0.pth \\
        --seg-config    training/configs/seg.yaml \\
        --watermark     data_gen/watermark.png \\
        [--store-dir    data_gen/aligned_store] \\
        [--limit        100] \\
        [--clear]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.benchmark_alignment import (
    build_prob_map,
    method_subpixel_xcorr,
    place_template,
)
from training.src.common.restoration import (
    build_store_signature,
    clear_preprocessed_store,
    get_store_root,
    iter_sample_dirs,
    sample_store_paths,
    write_preprocessed_sample,
)
from wm_shared.config import load_yaml_config, validate_removal_config
from wm_shared.preprocess import crop_by_roi, crop_removal_roi


def _build_aligned_sample(
    sample_dir: Path,
    seg_model,
    seg_size: int,
    device: torch.device,
    template: np.ndarray,
    image_width: int,
    image_height: int,
    crop_aspect_ratio: float,
    crop_margin_ratio: float,
    crop_min_width_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run seg → xcorr → 5px dilation for one sample.

    Returns (wm_r, clean_r, gt_mask_r, mask_input_r) ready to write to the store.
    All arrays are at (image_width, image_height) resolution.
    """
    wm_bgr = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
    clean_bgr = cv2.imread(str(sample_dir / "clean.png"), cv2.IMREAD_COLOR)
    gt_mask_gray = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (wm_bgr, clean_bgr, gt_mask_gray)):
        raise FileNotFoundError(f"Missing image files in {sample_dir}")

    gt_mask_f32 = gt_mask_gray.astype(np.float32) / 255.0
    img_h, img_w = wm_bgr.shape[:2]

    # ── 1. Segmentation ──────────────────────────────────────────────────────
    prob_map = build_prob_map(seg_model, wm_bgr, seg_size, device)  # HxW float32

    # ── 2. Alignment ─────────────────────────────────────────────────────────
    result = method_subpixel_xcorr(prob_map, template, img_w, img_h)

    # ── 3. Place aligned binary template at original resolution ──────────────
    aligned_bin = place_template(template, result.tx, result.ty, result.scale, img_w, img_h)
    aligned_f32 = aligned_bin.astype(np.float32)

    # ── 4. crop_removal_roi with aligned mask ────────────────────────────────
    #   - Determines the ROI from the aligned mask (same as service does)
    #   - Applies fixed 5px ELLIPSE dilation (use_augmented_mask=False)
    #   - Returns mask_input_r = aligned + dilated, cropped & resized
    wm_r, clean_r, _, roi, mask_input_r = crop_removal_roi(
        wm_bgr,
        clean_bgr,
        aligned_f32,
        image_width,
        image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=crop_margin_ratio,
        min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=False,
    )

    # ── 5. Crop GT mask to the same ROI for loss weighting ───────────────────
    gt_mask_r = crop_by_roi(gt_mask_f32, roi, image_width, image_height, is_mask=True)
    if gt_mask_r.max() > 1.0:
        gt_mask_r = gt_mask_r / 255.0

    return wm_r, clean_r, gt_mask_r.astype(np.float32), mask_input_r


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute aligned mask_input for removal model training"
    )
    parser.add_argument("--config",          required=True,
                        help="Removal training config YAML (e.g. training/configs/train_512.yaml)")
    parser.add_argument("--seg-checkpoint",  required=True,
                        help="Segmentation model checkpoint (.pth)")
    parser.add_argument("--seg-config",      default="training/configs/seg.yaml",
                        help="Segmentation model config YAML")
    parser.add_argument("--watermark",       default="data_gen/watermark.png",
                        help="Watermark RGBA PNG (same file used by the service)")
    parser.add_argument("--store-dir",       default=None,
                        help="Override preprocessed store root (default: from config)")
    parser.add_argument("--limit",           type=int, default=0,
                        help="Process at most N samples (0 = all)")
    parser.add_argument("--clear",           action="store_true",
                        help="Delete existing store entries before building")
    parser.add_argument("--device",          default="auto",
                        help="cuda / cpu / auto")
    args = parser.parse_args()

    # ── load configs ─────────────────────────────────────────────────────────
    config_path = Path(args.config).resolve()
    # Config paths (root, preprocessed_store_dir) are relative to the training/ directory,
    # matching how train.py resolves them when invoked from there.
    config_dir  = Path(__file__).resolve().parent

    removal_cfg = validate_removal_config(load_yaml_config(config_path))
    ds_cfg = removal_cfg["dataset"]

    seg_cfg = load_yaml_config(Path(args.seg_config).resolve())
    seg_size = seg_cfg["dataset"]["image_size"]

    image_width       = ds_cfg["image_width"]
    image_height      = ds_cfg["image_height"]
    crop_aspect_ratio = ds_cfg.get("crop_aspect_ratio", 3.54)
    crop_margin_ratio = ds_cfg.get("crop_margin_ratio", 0.10)
    crop_min_width    = ds_cfg.get("crop_min_width_ratio", 0.50)

    # Resolve dataset/store paths relative to the config file directory,
    # matching how train.py resolves them when run from training/.
    dataset_root = (config_dir / ds_cfg["root"]).resolve()
    raw_store    = args.store_dir or ds_cfg.get("preprocessed_store_dir")
    store_root   = get_store_root(dataset_root, (config_dir / raw_store).resolve() if raw_store else None)
    # Non-aug signature → used by validation and _infer_sample (exact aligned masks)
    signature    = build_store_signature(
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width,
    )
    # Aug signature → used by the training DataLoader.  Writing the same aligned
    # masks here lets training pick them up without invalidating any existing cache.
    aug_signature = build_store_signature(
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width,
        use_augmented_mask=True,
    )

    print(f"Dataset:     {dataset_root}")
    print(f"Store (val): {store_root / signature}")
    print(f"Store (trn): {store_root / aug_signature}")
    print(f"Resolution:  {image_width}x{image_height}")

    # ── load watermark template ───────────────────────────────────────────────
    wm_rgba = cv2.imread(str(Path(args.watermark).resolve()), cv2.IMREAD_UNCHANGED)
    if wm_rgba is None:
        raise FileNotFoundError(f"Cannot read watermark: {args.watermark}")
    if wm_rgba.shape[2] < 4:
        raise ValueError("Watermark image must have an alpha channel (RGBA)")
    template = wm_rgba[:, :, 3].astype(np.float32) / 255.0
    print(f"Watermark:   {args.watermark}  ({template.shape[1]}x{template.shape[0]}px)")

    # ── load segmentation model ───────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from training.src.tasks.segmentation.model import build_seg_model
    seg_model = build_seg_model(seg_cfg)
    ckpt = torch.load(args.seg_checkpoint, map_location="cpu", weights_only=True)
    seg_model.load_state_dict(ckpt["model"])
    seg_model.to(device).eval()
    print(f"Seg model:   {args.seg_checkpoint}  (device={device})")

    # ── collect samples ───────────────────────────────────────────────────────
    samples = iter_sample_dirs(dataset_root)
    max_s = ds_cfg.get("max_samples")
    if max_s is not None:
        samples = samples[:max_s]
    if args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_root}")

    if args.clear:
        print(f"\nClearing existing stores …")
        clear_preprocessed_store(store_root, signature)
        clear_preprocessed_store(store_root, aug_signature)

    print(f"\nProcessing {len(samples)} samples …\n")

    failed = 0
    for i, sample_dir in enumerate(samples, start=1):
        try:
            wm_r, clean_r, gt_mask_r, mask_input_r = _build_aligned_sample(
                sample_dir=sample_dir,
                seg_model=seg_model,
                seg_size=seg_size,
                device=device,
                template=template,
                image_width=image_width,
                image_height=image_height,
                crop_aspect_ratio=crop_aspect_ratio,
                crop_margin_ratio=crop_margin_ratio,
                crop_min_width_ratio=crop_min_width,
            )
            # Write to the non-aug store (validation / _infer_sample)
            write_preprocessed_sample(
                store_root=store_root,
                dataset_root=dataset_root,
                sample_dir=sample_dir,
                signature=signature,
                wm=wm_r,
                clean=clean_r,
                mask=gt_mask_r,
                mask_input=mask_input_r,
            )
            # Write the same aligned masks to the aug store so the training
            # DataLoader finds them without falling back to GT+jitter.
            write_preprocessed_sample(
                store_root=store_root,
                dataset_root=dataset_root,
                sample_dir=sample_dir,
                signature=aug_signature,
                wm=wm_r,
                clean=clean_r,
                mask=gt_mask_r,
                mask_input=mask_input_r,
            )
        except Exception as e:
            print(f"  [FAIL] {sample_dir.name}: {e}")
            failed += 1
            continue

        if i == 1 or i == len(samples) or i % 100 == 0:
            print(f"  [{i}/{len(samples)}]  {sample_dir.name}")

    print(f"\nDone.  {len(samples) - failed} succeeded, {failed} failed.")

    marker_payload = json.dumps({
        "cache_type": "aligned",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total": len(samples),
        "failed": failed,
    }, indent=2)

    for sig in (signature, aug_signature):
        ns_dir = store_root / sig
        ns_dir.mkdir(parents=True, exist_ok=True)
        (ns_dir / "_aligned_cache.json").write_text(marker_payload)
        print(f"Store: {ns_dir}")

    if failed > 0:
        print("Re-run with --clear to rebuild from scratch if needed.")


if __name__ == "__main__":
    main()
