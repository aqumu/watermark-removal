"""
infer_seg.py  –  predict a watermark mask from a watermarked image
-------------------------------------------------------------------
Usage:
  python infer_seg.py \\
      --checkpoint checkpoints_seg/epoch_0030.pth \\
      --watermarked path/to/watermarked.jpg \\
      --output      mask.png \\
      [--config configs/seg.yaml]
"""

import argparse
from pathlib import Path

import cv2
import yaml
import torch

from src.seg_model import build_seg_model
from infer import load_seg_model, predict_mask

_HERE = Path(__file__).parent


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--output",      default="mask.png")
    parser.add_argument("--config",      default=str(_HERE / "configs" / "seg.yaml"))
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    model  = load_seg_model(cfg, args.checkpoint, device)

    wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
    if wm is None:
        raise FileNotFoundError(f"Cannot read {args.watermarked}")

    mask = predict_mask(model, wm, cfg["dataset"]["image_size"], device)

    cv2.imwrite(args.output, mask)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
