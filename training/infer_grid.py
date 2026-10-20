"""
infer_grid.py - run a trained removal model on N random samples and stitch
predictions into a grid image for visual inspection.
-------------------------------------------------------------------------
Usage:
  python infer_grid.py --checkpoint artifacts/checkpoints/removal_256/best.pth
                       --config configs/train_256.yaml
                       [--n-samples 25] [--output grid.png] [--seed 42]
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.restoration import build_preprocessed_sample
from src.tasks.removal.model import build_model
from src.tasks.removal.store_cli import load_removal_cfg
from wm_shared.preprocess import compute_gradient


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("ema_shadow", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_256.yaml")
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--output", default="grid.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    cfg = load_removal_cfg(args.config)
    use_mask = cfg.get("model", {}).get("mask_input", True)
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_cfg)

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = load_model(cfg, args.checkpoint, device)
    print(f"Model on {device}  |  mask_input={'enabled' if use_mask else 'disabled'}", flush=True)

    mw = cfg["dataset"]["image_width"]
    mh = cfg["dataset"]["image_height"]

    ds_root = Path(cfg["dataset"]["root"])
    candidates = sorted([
        d for d in ds_root.iterdir()
        if d.is_dir() and (d / "watermarked.jpg").exists()
    ])
    rng = random.Random(args.seed)
    samples = rng.sample(candidates, min(args.n_samples, len(candidates)))
    print(f"Running inference on {len(samples)} samples ...", flush=True)

    preds = []
    for i, sdir in enumerate(samples):
        # Build preprocessed crop
        wm_r, _, _, mask_input = build_preprocessed_sample(
            sample_dir=sdir,
            image_width=mw, image_height=mh,
            crop_aspect_ratio=cfg["dataset"].get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=cfg["dataset"].get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=cfg["dataset"].get("crop_min_width_ratio", 0.50),
        )

        # Prepare model input
        rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        if use_mask:
            mask_t = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).to(device)
            grad_t = compute_gradient(wm_r).unsqueeze(0).to(device)
            inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)
        else:
            inp = rgb_t

        # Run model
        with torch.no_grad():
            delta = model(inp)
            pred = (rgb_t - delta).clamp(-1, 1)

        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_np = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        preds.append(pred_bgr)

        print(f"  [{i+1}/{len(samples)}] {sdir.name}", flush=True)

        # Keep GPU memory clean
        del rgb_t, inp, delta, pred
        if use_mask:
            del mask_t, grad_t
        torch.cuda.empty_cache()

    cols = args.cols
    rows_list = [np.hstack(preds[i:i + cols]) for i in range(0, len(preds), cols)]
    out = np.vstack(rows_list)
    cv2.imwrite(args.output, out)
    print(f"Saved grid ({out.shape[1]}x{out.shape[0]}) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
