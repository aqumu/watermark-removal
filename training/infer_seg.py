"""
infer_seg.py  –  predict a watermark mask from a watermarked image
-------------------------------------------------------------------
Usage:
  python infer_seg.py \\
      --checkpoint checkpoints_seg/best.pth \\
      --watermarked path/to/watermarked.jpg \\
      --output      mask.png \\
      [--config configs/seg.yaml]

  # Debug mode: show N random dataset samples in a matplotlib window
  python infer_seg.py \\
      --checkpoint checkpoints_seg/best.pth \\
      --debug [--n-samples 12] \\
      [--config configs/seg.yaml]
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch

from infer import load_seg_model, predict_mask

_HERE = Path(__file__).parent


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _debug_grid(model, cfg, device, n_samples: int):
    import matplotlib.pyplot as plt

    ds_root = Path(cfg["dataset"]["root"])
    size    = cfg["dataset"]["image_size"]

    candidates = sorted([
        d for d in ds_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "mask.png").exists()
    ])
    if not candidates:
        raise RuntimeError(f"No samples found in {ds_root}")

    samples = random.sample(candidates, min(n_samples, len(candidates)))

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Two samples per row, each sample occupies 3 columns: wm | pred | gt
    samples_per_row = 2
    cols = samples_per_row * 3
    rows = (len(samples) + samples_per_row - 1) // samples_per_row
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("watermarked  |  predicted  |  gt        " * samples_per_row, fontsize=9)

    # Normalise axes to always be 2-D
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, sample_dir in enumerate(samples):
        row = i // samples_per_row
        col = (i % samples_per_row) * 3

        wm = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(sample_dir / "mask.png"),        cv2.IMREAD_GRAYSCALE)

        wm_r = cv2.resize(wm, (size, size), interpolation=cv2.INTER_AREA)
        gt_r = cv2.resize(gt, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb  = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb  = (rgb - _MEAN) / _STD
        inp  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()  # HxW [0,1]

        iou = _iou(prob >= 0.5, gt_r >= 128)

        axes[row, col    ].imshow(cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB))
        axes[row, col    ].set_title(sample_dir.name, fontsize=7)
        axes[row, col + 1].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[row, col + 1].set_title(f"pred  IoU={iou:.2f}", fontsize=7)
        axes[row, col + 2].imshow(gt_r, cmap="hot", vmin=0, vmax=255)
        axes[row, col + 2].set_title("gt", fontsize=7)

    # Hide any unused axes in the last row
    for j in range(len(samples), rows * samples_per_row):
        row = j // samples_per_row
        col = (j % samples_per_row) * 3
        for k in range(3):
            axes[row, col + k].axis("off")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    intersection = (pred_bin & gt_bin).sum()
    union        = (pred_bin | gt_bin).sum()
    return float(intersection) / max(float(union), 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default=str(_HERE / "configs" / "seg.yaml"))
    # normal mode
    parser.add_argument("--watermarked", default=None)
    parser.add_argument("--output",      default="mask.png")
    # debug mode
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    if cfg.get("device", "cpu") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"])
    model  = load_seg_model(cfg, args.checkpoint, device)

    if args.debug:
        _debug_grid(model, cfg, device, args.n_samples)
        return

    if not args.watermarked:
        parser.error("--watermarked is required unless --debug is set")

    wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
    if wm is None:
        raise FileNotFoundError(f"Cannot read {args.watermarked}")

    mask = predict_mask(model, wm, cfg["dataset"]["image_size"], device)
    cv2.imwrite(args.output, mask)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
