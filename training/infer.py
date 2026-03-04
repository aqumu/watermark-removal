"""
infer.py  –  run the trained model on a single image + mask
-------------------------------------------------------------
Usage:
  python infer.py \\
      --checkpoint checkpoints/epoch_0060.pth \\
      --watermarked path/to/watermarked.jpg \\
      --mask        path/to/mask.png \\
      --output      result.png \\
      [--size 512]  [--config configs/train.yaml]
"""

import argparse

import cv2
import numpy as np
import torch
import yaml

from src.model import build_model


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def preprocess(wm_path: str, mask_path: str, size: int):
    """
    Returns
    -------
    inp_tensor : torch.Tensor  1x4xHxW  in [-1,1] / {0,1}
    orig_wm    : np.ndarray    HxWx3 uint8 BGR  (original-size, for blending)
    orig_mask  : np.ndarray    HxW  uint8       (original-size, for blending)
    orig_size  : (W, H)
    """
    wm   = cv2.imread(wm_path,   cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Cannot read {wm_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")

    orig_size = (wm.shape[1], wm.shape[0])   # (W, H)

    wm_r   = cv2.resize(wm,   (size, size), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask_r = (mask_r > 127).astype(np.float32)

    rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)       # 1x3xHxW
    mask_t = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0)          # 1x1xHxW

    inp = torch.cat([rgb_t, mask_t], dim=1)   # 1x4xHxW
    return inp, wm, mask, orig_size


@torch.no_grad()
def run(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    """Returns a uint8 BGR image at the model's working resolution."""
    inp   = inp.to(device)
    delta = model(inp)                                     # 1x3xHxW, residual [0,1]
    pred  = (inp[:, :3] - delta).clamp(-1, 1)             # 1x3xHxW, clean [-1,1]
    pred  = pred.squeeze(0).cpu().numpy()                  # 3xHxW
    pred  = ((pred + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pred.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)


def blend_back(pred_bgr, orig_wm, orig_mask, orig_size, feather: int = 9):
    """
    Paste the model prediction back at original resolution.

    feather : radius (px) of the Gaussian used to soften the mask edge.
              0 = hard binary cut (can show seams).
              ~9 = smooth ~20-px transition zone, eliminates most edge artefacts.
    """
    pred_up = cv2.resize(pred_bgr, orig_size, interpolation=cv2.INTER_CUBIC)

    if feather > 0:
        # ksize must be odd
        k = feather * 2 + 1
        soft = cv2.GaussianBlur(orig_mask.astype(np.float32), (k, k), feather / 2)
        m = (soft / 255.0)[:, :, None]
    else:
        m = (orig_mask > 127)[:, :, None].astype(np.float32)

    out = (pred_up * m + orig_wm * (1 - m)).clip(0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--mask",        required=True)
    parser.add_argument("--output",      default="result.png")
    parser.add_argument("--size",        type=int, default=512)
    parser.add_argument("--config",      default="configs/train.yaml")
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    model  = load_model(cfg, args.checkpoint, device)

    inp, orig_wm, orig_mask, orig_size = preprocess(
        args.watermarked, args.mask, args.size
    )

    pred_bgr = run(model, inp, device)
    result   = blend_back(pred_bgr, orig_wm, orig_mask, orig_size)

    cv2.imwrite(args.output, result)
    print(f"Saved → {args.output}  (original resolution: {orig_size[0]}×{orig_size[1]})")


if __name__ == "__main__":
    main()
