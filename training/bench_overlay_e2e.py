"""
bench_overlay_e2e.py  end-to-end CPU cost per photo for overlay removal
-----------------------------------------------------------------------
Times every stage on real photos: JPEG decode, window preparation (mark
localisation + hint rendering + model input), the ONNX model, and pasting the
window back. The model runs one window at a time and in batches.

Usage:
  python training/bench_overlay_e2e.py --onnx model.onnx [--config cfg.yaml] [--n 64] [--threads 6]
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tasks.overlay.data import is_holdout
from src.tasks.overlay.inference import prepare_window
from train_overlay import load_cfg
from wm_shared.marks import load_mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--config", default=str(TRAINING_DIR / "configs" / "cian_teacher_ft4.yaml"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import onnxruntime as ort
    cv2.setNumThreads(1)
    cfg = load_cfg(args.config)
    profile = load_mark(cfg["mark"])
    hint_scale = float(cfg["data"].get("hint_scale", 8.0))
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    rows = [r for r in csv.DictReader(open(cfg["data"]["preview_csv"], encoding="utf-8"))
            if is_holdout(r["listing_id"], float(cfg["data"].get("holdout", 0.05)))]
    random.Random(3).shuffle(rows)
    rows = rows[:args.n]

    t_dec = t_prep = t_model = t_paste = 0.0
    inputs = []
    for r in rows:
        t = time.perf_counter()
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)[..., ::-1].copy()
        t_dec += time.perf_counter() - t
        t = time.perf_counter()
        (x0, y0), x, _ = prepare_window(img, profile, r["listing_id"], hint_scale)
        t_prep += time.perf_counter() - t
        xn = x.numpy()
        inputs.append(xn)
        t = time.perf_counter()
        delta = sess.run(None, {"x": xn})[0]
        t_model += time.perf_counter() - t
        t = time.perf_counter()
        pred = np.clip(xn[:, :3] - delta, -1, 1)[0].transpose(1, 2, 0)
        out_win = np.rint((pred + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out = img.copy()
        h, w = out_win.shape[:2]
        out[y0:y0 + h, x0:x0 + w] = out_win[:img.shape[0] - y0, :img.shape[1] - x0]
        t_paste += time.perf_counter() - t
    n = len(rows)
    # batched model only (preprocessing is per image and parallelises across processes)
    batches = [np.concatenate(inputs[i:i + args.batch]) for i in range(0, n - args.batch + 1, args.batch)]
    t = time.perf_counter()
    for b in batches:
        sess.run(None, {"x": b})
    t_batch = (time.perf_counter() - t) / max(1, len(batches) * args.batch)
    print(f"{n} photos, {args.threads} threads: decode {t_dec / n * 1000:.1f} ms, prepare (localise + hint) "
          f"{t_prep / n * 1000:.1f} ms, model {t_model / n * 1000:.1f} ms (batched x{args.batch}: {t_batch * 1000:.1f} ms), "
          f"paste {t_paste / n * 1000:.1f} ms -> total {(t_dec + t_prep + t_model + t_paste) / n * 1000:.1f} ms/photo")


if __name__ == "__main__":
    main()
