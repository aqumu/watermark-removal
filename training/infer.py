"""
infer.py  –  run the trained model on a single image + mask
-------------------------------------------------------------
Usage:
  # provide mask manually
  python infer.py \\
      --checkpoint checkpoints/epoch_0060.pth \\
      --watermarked path/to/watermarked.jpg \\
      --mask        path/to/mask.png \\
      --output      result.png \\
      [--size 512]  [--config configs/train.yaml]

  # auto-generate mask with the segmentation model
  python infer.py \\
      --checkpoint     checkpoints/epoch_0060.pth \\
      --watermarked    path/to/watermarked.jpg \\
      --seg-checkpoint checkpoints_seg/epoch_0030.pth \\
      --output         result.png \\
      [--size 512]  [--config configs/train.yaml]  [--seg-config configs/seg.yaml]
"""

import argparse

import cv2
import numpy as np
import torch
import yaml

from src.model     import build_model
from src.seg_model import build_seg_model


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_seg_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_seg_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(seg_model, wm_bgr: np.ndarray, size: int,
                 device: torch.device) -> np.ndarray:
    """
    Run the segmentation model on a BGR image and return a binary uint8 mask
    (0 / 255) at the original image resolution.
    """
    h, w = wm_bgr.shape[:2]
    wm_r = cv2.resize(wm_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    inp  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pred   = seg_model(inp)                                      # 1x1xHxW in [0,1]
    pred_np = pred.squeeze().cpu().numpy()                       # HxW
    binary  = (pred_np > 0.5).astype(np.uint8) * 255
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)


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
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--watermarked",    required=True)
    parser.add_argument("--mask",           default=None,
                        help="path to binary mask PNG; omit to auto-generate via --seg-checkpoint")
    parser.add_argument("--seg-checkpoint", default=None,
                        help="segmentation model checkpoint for automatic mask prediction")
    parser.add_argument("--seg-config",     default="configs/seg.yaml")
    parser.add_argument("--output",         default="result.png")
    parser.add_argument("--size",           type=int, default=512)
    parser.add_argument("--config",         default="configs/train.yaml")
    args = parser.parse_args()

    if args.mask is None and args.seg_checkpoint is None:
        parser.error("provide either --mask or --seg-checkpoint")

    cfg    = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    model  = load_model(cfg, args.checkpoint, device)

    if args.mask is not None:
        inp, orig_wm, orig_mask, orig_size = preprocess(
            args.watermarked, args.mask, args.size
        )
    else:
        seg_cfg  = load_cfg(args.seg_config)
        seg_model = load_seg_model(seg_cfg, args.seg_checkpoint, device)

        orig_wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
        if orig_wm is None:
            raise FileNotFoundError(f"Cannot read {args.watermarked}")
        orig_size = (orig_wm.shape[1], orig_wm.shape[0])

        seg_size  = seg_cfg["dataset"]["image_size"]
        orig_mask = predict_mask(seg_model, orig_wm, seg_size, device)
        print(f"Mask predicted by segmentation model (seg size={seg_size})")

        # build the removal model input from the predicted mask
        wm_r   = cv2.resize(orig_wm,   (args.size, args.size), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(orig_mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        mask_r = (mask_r > 127).astype(np.float32)

        rgb    = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        mask_t = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0)
        inp    = torch.cat([rgb_t, mask_t], dim=1)

    pred_bgr = run(model, inp, device)
    result   = blend_back(pred_bgr, orig_wm, orig_mask, orig_size)

    cv2.imwrite(args.output, result)
    print(f"Saved → {args.output}  (original resolution: {orig_size[0]}×{orig_size[1]})")


if __name__ == "__main__":
    main()
