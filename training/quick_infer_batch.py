"""
quick_infer_batch.py  –  throwaway script
------------------------------------------
Loads epoch_9999.pth, grabs 32 random samples from the preprocessed cache,
runs the removal model, and dumps side-by-side comparison images
(watermarked | predicted | ground truth) into an output folder.

Usage:
    cd training
    python quick_infer_batch.py          # defaults should just work
    python quick_infer_batch.py --n 16   # fewer images
    python quick_infer_batch.py --out my_folder
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wm_shared.config import load_yaml_config, validate_removal_config
from wm_shared.preprocess import compute_gradient
from training.src.tasks.removal.model import build_model

# ── defaults ────────────────────────────────────────────────────────────────
CHECKPOINT = (
    TRAINING_ROOT
    / "runs/removal_512/removal_512__20260418_043523"
    / "artifacts/checkpoints/epoch_9999.pth"
)
CONFIG = (
    TRAINING_ROOT
    / "runs/removal_512/removal_512__20260418_043523"
    / "meta/config.yaml"
)
STORE_DIR = REPO_ROOT / "data_gen" / "preprocessed_store" / "removal-512x256-1b28e5eaebf5"
DEFAULT_OUT = TRAINING_ROOT / "quick_infer_output"


def load_model(cfg: dict, ckpt_path: Path, device: torch.device):
    model = build_model(cfg)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    # Prefer EMA shadow weights — they're the averaged version and typically
    # produce noticeably better results than the raw optimiser weights.
    if "ema" in ckpt and "shadow" in ckpt["ema"]:
        print("Using EMA shadow weights")
        model.load_state_dict(ckpt["ema"]["shadow"])
    else:
        print("No EMA weights found, using raw model weights")
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_one(model, wm_bgr: np.ndarray, mask_input: np.ndarray, device: torch.device):
    """Run the model on a single preprocessed sample. Returns predicted BGR uint8."""
    # RGB [-1, 1]
    rgb = cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # mask [0, 1]
    mask_t = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).to(device)

    # gradient
    grad_t = compute_gradient(wm_bgr).unsqueeze(0).to(device)

    inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)  # 5-ch
    delta = model(inp)
    pred = (rgb_t - delta).clamp(-1, 1)

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_bgr = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_bgr, cv2.COLOR_RGB2BGR)
    return pred_bgr


def make_comparison(wm: np.ndarray, pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Horizontal concat: watermarked | predicted | ground truth."""
    gap = np.full((wm.shape[0], 4, 3), 200, dtype=np.uint8)  # light-gray divider
    return np.concatenate([wm, gap, pred, gap, clean], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Quick batch inference with epoch_9999")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--store", type=Path, default=STORE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=32, help="number of images to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── setup ───────────────────────────────────────────────────────────────
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = validate_removal_config(load_yaml_config(str(args.config)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(cfg, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint.name}")

    # ── gather cached samples ───────────────────────────────────────────────
    # Reproduce the exact train/val split used during training so we can
    # test on images the model actually saw (max_samples + seed 42).
    dataset_root = Path(cfg["dataset"]["root"])
    max_samples = cfg["dataset"].get("max_samples")
    all_sample_dirs = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "clean.png").exists()
        and (d / "mask.png").exists()
    ])
    if max_samples is not None:
        import torch as _torch
        rng = _torch.Generator().manual_seed(cfg.get("seed", 42))
        idx = _torch.randperm(len(all_sample_dirs), generator=rng).tolist()[:max_samples]
        selected_names = {all_sample_dirs[i].name for i in idx}
        print(f"Filtering to the {len(selected_names)} training-set samples (max_samples={max_samples})")
    else:
        selected_names = None

    sample_dirs = sorted([
        d for d in args.store.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "mask_input.png").exists()
        and (d / "clean.png").exists()
        and (selected_names is None or d.name in selected_names)
    ])
    if not sample_dirs:
        print(f"ERROR: No preprocessed samples found in {args.store}")
        sys.exit(1)

    chosen = sample_dirs[:args.n]  # take up to N from the training set
    print(f"Running inference on {len(chosen)} samples from {args.store.name} ...")

    # ── run ─────────────────────────────────────────────────────────────────
    for i, sd in enumerate(chosen):
        wm = cv2.imread(str(sd / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask_input = cv2.imread(str(sd / "mask_input.png"), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(sd / "clean.png"), cv2.IMREAD_COLOR)

        mask_f = mask_input.astype(np.float32) / 255.0
        pred = infer_one(model, wm, mask_f, device)

        # save comparison strip
        comp = make_comparison(wm, pred, clean)
        out_path = args.out / f"{sd.name}_compare.png"
        cv2.imwrite(str(out_path), comp)

        # also save the prediction standalone
        pred_path = args.out / f"{sd.name}_pred.png"
        cv2.imwrite(str(pred_path), pred)

        print(f"  [{i+1:>3}/{len(chosen)}] {sd.name}")

    print(f"\nDone — {len(chosen)} images saved to {args.out}")


if __name__ == "__main__":
    main()
