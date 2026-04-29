"""
visualize_alignment.py  –  Real-world vs training-augmentation alignment comparison
====================================================================================

For each sample produces a two-row composite:

  Row 1 (real-world):   GT mask | xcorr-aligned template  | diff
  Row 2 (training aug): GT mask | augmented GT mask        | diff

Both diff panels use the same legend:
  green  = GT only (missed)
  red    = predicted/augmented only (extra)
  grey   = agreement

This lets you judge whether the ±4 px jitter in training actually matches
the real-world xcorr alignment error.

Usage:
    python visualize_alignment.py \
        --seg-checkpoint C:/Users/aqumu/PycharmProjects/watermark-service/models/model_seg_2.0.pth \
        --watermark      ../data_gen/watermark.png \
        --dataset        ../data_gen/dataset \
        --output-dir     alignment_vis \
        --n-samples      30
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.benchmark_alignment import (
    build_prob_map,
    gt_scale_from_mask,
    method_subpixel_xcorr,
    place_template,
)


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_mask_overlay(image: np.ndarray, mask_bin: np.ndarray,
                      color: tuple, alpha: float = 0.35) -> np.ndarray:
    out = image.copy().astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[mask_bin > 0] = color
    out = out * (1 - alpha) + overlay * alpha
    return out.clip(0, 255).astype(np.uint8)


def draw_contour(image: np.ndarray, mask_bin: np.ndarray,
                 color: tuple, thickness: int = 2) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(
        mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def make_panel(image: np.ndarray, mask_bin: np.ndarray,
               fill_color: tuple, label: str) -> np.ndarray:
    panel = draw_mask_overlay(image, mask_bin, fill_color, alpha=0.25)
    panel = draw_contour(panel, mask_bin, fill_color, thickness=2)
    cv2.putText(panel, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def make_diff_panel(image: np.ndarray,
                    gt_bin: np.ndarray,
                    other_bin: np.ndarray,
                    label: str,
                    iou: float | None = None,
                    extra: str = "") -> np.ndarray:
    """
    Diff overlay:
      green = GT only (missed)
      red   = other only (extra)
      grey  = agreement
    """
    gt_only    = gt_bin & ~other_bin
    other_only = ~gt_bin & other_bin
    agree      = gt_bin & other_bin

    panel = draw_mask_overlay(image, gt_only,    (0,   200,   0), alpha=0.55)
    panel = draw_mask_overlay(panel, other_only, (0,    0,  200), alpha=0.55)
    panel = draw_mask_overlay(panel, agree,      (180, 180, 180), alpha=0.20)

    cv2.putText(panel, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   0,   0),   1, cv2.LINE_AA)

    if iou is not None:
        metric_str = f"IoU={iou:.3f}"
        if extra:
            metric_str += f"  {extra}"
        cv2.putText(panel, metric_str, (8, panel.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, metric_str, (8, panel.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,   0,   0),   1, cv2.LINE_AA)
    return panel


# ── training augmentation (mirrors dilate_mask_input, augment=True, no final dilation) ──

def apply_training_jitter(gt_bin: np.ndarray, image_size: int) -> np.ndarray:
    """
    Apply positional + scale jitter identical to dilate_mask_input(augment=True)
    but WITHOUT the final 5-px dilation, so we can compare the raw error shape.
    """
    binary = gt_bin.astype(np.uint8)

    # Positional jitter: ±max_translate px (same formula as preprocess.py)
    max_translate = max(1, round(4 * image_size / 256))
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    if tx != 0 or ty != 0:
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        binary = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]))

    # Scale jitter: uniform ±4 px erode/dilate with ELLIPSE kernel
    jitter = random.randint(-4, 4)
    if jitter != 0:
        k = abs(jitter) * 2 + 1
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if jitter > 0:
            binary = cv2.dilate(binary, kernel_s, iterations=1)
        else:
            binary = cv2.erode(binary, kernel_s, iterations=1)

    return binary.astype(np.uint8)


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / max(union, 1.0)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare real-world xcorr alignment errors vs training augmentation errors"
    )
    parser.add_argument("--seg-checkpoint", required=True)
    parser.add_argument("--seg-config",  default="configs/seg.yaml")
    parser.add_argument("--watermark",   default="../data_gen/watermark.png")
    parser.add_argument("--dataset",     default="../data_gen/dataset")
    parser.add_argument("--output-dir",  default="alignment_vis")
    parser.add_argument("--n-samples",   type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-width",   type=int, default=560,
                        help="Max width of each panel in the output image")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load watermark template (alpha channel)
    wm_rgba = cv2.imread(args.watermark, cv2.IMREAD_UNCHANGED)
    if wm_rgba is None:
        raise FileNotFoundError(f"Cannot read watermark: {args.watermark}")
    template = wm_rgba[:, :, 3].astype(np.float32) / 255.0

    # Collect samples
    ds_root = Path(args.dataset)
    sample_dirs = sorted([
        d for d in ds_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "mask.png").exists()
    ])
    if not sample_dirs:
        raise RuntimeError(f"No valid samples in {ds_root}")
    n = min(args.n_samples, len(sample_dirs))
    sample_dirs = random.sample(sample_dirs, n)

    # Load seg model
    import torch
    from wm_shared.config import load_yaml_config
    from training.src.tasks.segmentation.model import build_seg_model

    seg_cfg = load_yaml_config(args.seg_config)
    seg_size = seg_cfg["dataset"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = build_seg_model(seg_cfg)
    ckpt = torch.load(args.seg_checkpoint, map_location="cpu", weights_only=True)
    seg_model.load_state_dict(ckpt["model"])
    seg_model.to(device).eval()
    print(f"Loaded seg model  (device={device})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_divider_h = 6  # px gap between the two rows

    for i, sample_dir in enumerate(sample_dirs):
        wm_bgr     = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        gt_mask_u8 = cv2.imread(str(sample_dir / "mask.png"),  cv2.IMREAD_GRAYSCALE)
        if wm_bgr is None or gt_mask_u8 is None:
            continue

        img_h, img_w = wm_bgr.shape[:2]
        gt_bin = (gt_mask_u8 > 128).astype(np.uint8)

        # ── real-world: subpixel_xcorr ─────────────────────────────────────
        prob_map = build_prob_map(seg_model, wm_bgr, seg_size, device)
        result   = method_subpixel_xcorr(
            prob_map=prob_map, template=template, img_w=img_w, img_h=img_h
        )
        aligned_bin = place_template(template, result.tx, result.ty, result.scale, img_w, img_h)
        iou_real    = compute_iou(aligned_bin, gt_bin)

        # ── training augmentation ──────────────────────────────────────────
        aug_bin  = apply_training_jitter(gt_bin, max(img_h, img_w))
        iou_aug  = compute_iou(aug_bin, gt_bin)

        # ── resize for display ─────────────────────────────────────────────
        scale_f = min(1.0, args.max_width / img_w)
        disp_w  = max(1, int(img_w * scale_f))
        disp_h  = max(1, int(img_h * scale_f))

        def resize_img(img):
            return cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        def resize_mask(m):
            return (cv2.resize(m.astype(np.uint8) * 255, (disp_w, disp_h),
                               interpolation=cv2.INTER_NEAREST) > 128).astype(np.uint8)

        img_s     = resize_img(wm_bgr)
        gt_s      = resize_mask(gt_bin)
        aln_s     = resize_mask(aligned_bin)
        aug_s     = resize_mask(aug_bin)

        # ── row 1: real-world alignment ────────────────────────────────────
        p1_gt   = make_panel(img_s, gt_s, (0, 200, 0), "GT mask")
        p1_aln  = make_panel(img_s, aln_s, (200, 0, 0), "xcorr aligned")
        p1_diff = make_diff_panel(img_s, gt_s, aln_s,
                                  label="Real-world diff",
                                  iou=iou_real)

        # ── row 2: training augmentation ──────────────────────────────────
        p2_gt   = make_panel(img_s, gt_s, (0, 200, 0), "GT mask")
        p2_aug  = make_panel(img_s, aug_s, (200, 0, 0), "Train aug (pre-dilation)")
        p2_diff = make_diff_panel(img_s, gt_s, aug_s,
                                  label="Training aug diff",
                                  iou=iou_aug)

        # ── compose ────────────────────────────────────────────────────────
        div_v = np.full((disp_h, 4, 3), 60, dtype=np.uint8)
        row1  = np.concatenate([p1_gt, div_v, p1_aln, div_v, p1_diff], axis=1)
        row2  = np.concatenate([p2_gt, div_v, p2_aug, div_v, p2_diff], axis=1)

        # Horizontal separator between rows
        row_sep = np.full((row_divider_h, row1.shape[1], 3), 30, dtype=np.uint8)

        composite = np.concatenate([row1, row_sep, row2], axis=0)

        out_path = out_dir / f"{i:03d}_{sample_dir.name}.jpg"
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(
            f"[{i+1}/{n}]  {sample_dir.name}"
            f"  IoU_real={iou_real:.3f}"
            f"  IoU_aug={iou_aug:.3f}"
            f"  → {out_path.name}"
        )

    print(f"\nSaved {n} images to {out_dir}/")


if __name__ == "__main__":
    main()
