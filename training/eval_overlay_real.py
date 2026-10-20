"""
eval_overlay_real.py  measure leftover mark on real stamped photos
------------------------------------------------------------------
Runs one or more checkpoints on real photos from held-out listings (no ground
truth) and measures how much of the mark is left: the least-squares amplitude
of the expected mark signal m*(c - b) in the window's detail layer d
(b = smoothed background), weighted towards flat areas:

  amplitude = sum w*d*s / sum w*s*s,   s = m*(c - b)

About 1 on the input if the template matches the real mark's strength, 0 after
a clean removal, negative if the output carries an inverted mark (over-removal).
Also reported: the slope of the leftover against the input amplitude (should be
~0: the model reads the mark's real strength instead of removing a fixed amount).

The grid shows, per photo, the input and each model's output with a strong
local-contrast boost (x8), where faint leftovers become visible.

Usage:
  python training/eval_overlay_real.py --config cfg.yaml --checkpoint a.pth [b.pth ...]
         [--n 400] [--grid out.png] [--grid-rows 10]
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


def boost(rgb_u8: np.ndarray, gain: float = 8.0) -> np.ndarray:
    f = rgb_u8.astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), 12)
    return np.clip(b + (f - b) * gain, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--config", default=str(TRAINING_DIR / "configs" / "cian_teacher.yaml"),
                    help="model / data settings (all checkpoints must share the architecture)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-score", type=float, default=0.3, help="'clearly located' threshold")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--grid-rows", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    profile = load_mark(cfg["mark"])
    hint_scale = float(cfg["data"].get("hint_scale", 8.0))
    models = []
    for path in args.checkpoint:
        model = build_model(cfg)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        models.append((Path(path).parent.parent.parent.name + "/" + Path(path).name, model.to(args.device).eval()))

    holdout = float(cfg["data"].get("holdout", 0.05))
    rows = [r for r in csv.DictReader(open(cfg["data"]["preview_csv"], encoding="utf-8"))
            if is_holdout(r["listing_id"], holdout)]
    random.Random(99).shuffle(rows)
    a_in, scores, a_out = [], [], {name: [] for name, _ in models}
    times = {name: 0.0 for name, _ in models}
    tiles = []
    for r in rows[:args.n]:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        (x0, y0), x, (lx, ly, score, style) = prepare_window(img[..., ::-1].copy(), profile, r["listing_id"], hint_scale)
        win = ((x[0, :3].numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
        m = x[0, 6].numpy() / hint_scale
        a_in.append(mark_amplitude(win, m, profile.overlay))
        scores.append(abs(score))
        row = [boost(win)]
        for name, model in models:
            with torch.no_grad():
                t = time.time()
                delta = model(x.to(args.device)).float()
                times[name] += time.time() - t
            pred = (x[:, :3].to(args.device) - delta).clamp(-1, 1)[0].cpu().numpy().transpose(1, 2, 0)
            out = ((pred + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
            a_out[name].append(mark_amplitude(out, m, profile.overlay))
            row.append(boost(out))
        if args.grid and len(tiles) < args.grid_rows and abs(score) >= args.min_score:
            tiles.append(np.concatenate(row, axis=1)[..., ::-1])

    a_in, scores = np.array(a_in), np.array(scores)
    v = scores >= args.min_score
    print(f"{len(a_in)} photos, {v.sum()} clearly located; input amplitude median {np.median(a_in[v]):.3f} "
          f"(p10 {np.percentile(a_in[v], 10):.3f}, p90 {np.percentile(a_in[v], 90):.3f})")
    for name, _ in models:
        o = np.array(a_out[name])
        slope = np.polyfit(a_in[v], o[v], 1)[0]
        print(f"{name}: leftover median {np.median(o[v]):+.3f}  p10 {np.percentile(o[v], 10):+.3f}  "
              f"p90 {np.percentile(o[v], 90):+.3f}  |leftover| p90 {np.percentile(np.abs(o[v]), 90):.3f}  "
              f"slope vs input {slope:+.2f}  |all photos| median {np.median(np.abs(o)):.3f}  "
              f"{times[name] / len(a_in) * 1000:.1f} ms/window on {args.device}")
    if tiles:
        cv2.imwrite(args.grid, np.concatenate(tiles, axis=0))
        print(f"grid -> {args.grid} (columns: input, " + ", ".join(n for n, _ in models) + "; all boosted x8)")


if __name__ == "__main__":
    main()
