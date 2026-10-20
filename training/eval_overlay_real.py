"""
eval_overlay_real.py  measure leftover mark on real stamped photos
------------------------------------------------------------------
Runs a checkpoint on real photos from held-out listings (no ground truth) and
measures how much of the mark is left: the least-squares amplitude of the
expected mark signal m*(c - b) in the window's detail layer d (b = smoothed
background), weighted towards flat areas:

  amplitude = sum w*d*s / sum w*s*s,   s = m*(c - b)

About 1 on the input if the template matches the real mark's strength, 0 after
a clean removal, negative if the output carries an inverted mark (over-removal).

Usage:
  python training/eval_overlay_real.py --checkpoint <best.pth> [--config ...] [--n 300] [--grid out.png]
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tasks.overlay.data import is_holdout
from src.tasks.overlay.inference import prepare_window
from src.tasks.overlay.model import build_model
from src.tasks.overlay.trainer import contrast_view
from train_overlay import load_cfg
from wm_shared.marks import load_mark

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def mark_amplitude(win_rgb: np.ndarray, m: np.ndarray, overlay: float) -> float:
    g = win_rgb.astype(np.float32) @ _LUMA
    support = cv2.dilate((m > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    keep = 1 - support
    b = cv2.GaussianBlur(g * keep, (0, 0), 5) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 5), 1e-3)
    var = cv2.GaussianBlur((g - b) ** 2 * keep, (0, 0), 5) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 5), 1e-3)
    w = support / (var + 2.0)
    d = g - b
    sig = m * (overlay - b)
    return float((w * d * sig).sum() / max(float((w * sig * sig).sum()), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(TRAINING_DIR / "configs" / "cian_low.yaml"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--grid", default=None, help="write a before/after grid of the first 12 photos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    profile = load_mark(cfg["mark"])
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    hint_scale = float(cfg["data"].get("hint_scale", 8.0))

    holdout = float(cfg["data"].get("holdout", 0.05))
    rows = [r for r in csv.DictReader(open(cfg["data"]["preview_csv"], encoding="utf-8"))
            if is_holdout(r["listing_id"], holdout)]
    random.Random(99).shuffle(rows)
    ev_in, ev_out, scores, tiles = [], [], [], []
    t_model = 0.0
    for r in rows[:args.n]:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = img[..., ::-1].copy()
        (x0, y0), x, (lx, ly, score, style) = prepare_window(rgb, profile, r["listing_id"], hint_scale)
        with torch.no_grad():
            t = time.time()
            delta = model(x.to(args.device)).float()
            t_model += time.time() - t
        pred = (x[:, :3].to(args.device) - delta).clamp(-1, 1)[0].cpu().numpy().transpose(1, 2, 0)
        out = ((pred + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        win = ((x[0, :3].numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
        m = x[0, 6].numpy() / hint_scale
        ev_in.append(mark_amplitude(win, m, profile.overlay))
        ev_out.append(mark_amplitude(out, m, profile.overlay))
        scores.append(abs(score))
        if args.grid and len(tiles) < 12:
            b, a = win[..., ::-1], out[..., ::-1]
            tiles.append(np.concatenate([b, a, contrast_view(b), contrast_view(a)], axis=1))
    ev_in, ev_out, scores = map(np.array, (ev_in, ev_out, scores))
    vis = scores > 0.3   # photos where the mark is clearly located
    print(f"{len(ev_in)} photos; mark amplitude input median {np.median(ev_in):.3f} -> output {np.median(ev_out):.3f}")
    print(f"clearly located ({vis.sum()}): input {np.median(ev_in[vis]):.3f} -> output {np.median(ev_out[vis]):.3f} "
          f"(p10 {np.percentile(ev_out[vis], 10):.3f}, p90 {np.percentile(ev_out[vis], 90):.3f})")
    print(f"model time {t_model / max(len(ev_in), 1) * 1000:.1f} ms/window on {args.device}")
    if tiles:
        cv2.imwrite(args.grid, np.concatenate(tiles, axis=0))
        print(f"grid -> {args.grid}")


if __name__ == "__main__":
    main()
