"""
quantize_overlay_onnx.py  int8 static quantisation of an exported overlay model
------------------------------------------------------------------------------
Calibrates on real stamped windows from held-out listings (built with the same
preprocessing as inference), quantises with ONNX Runtime (QDQ, per-channel
weights), then compares the int8 model with fp32 on other real windows:
speed, output difference and leftover-mark amplitude.

Usage:
  python training/quantize_overlay_onnx.py --onnx model.onnx --config cfg.yaml [--out model.int8.onnx]
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

from eval_overlay_real import mark_amplitude
from src.tasks.overlay.data import is_holdout
from src.tasks.overlay.inference import prepare_window
from train_overlay import load_cfg
from wm_shared.marks import load_mark


def real_windows(cfg: dict, profile, n: int, seed: int) -> list[np.ndarray]:
    holdout = float(cfg["data"].get("holdout", 0.05))
    hint_scale = float(cfg["data"].get("hint_scale", 8.0))
    rows = [r for r in csv.DictReader(open(cfg["data"]["preview_csv"], encoding="utf-8"))
            if is_holdout(r["listing_id"], holdout)]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        _, x, _ = prepare_window(img[..., ::-1].copy(), profile, r["listing_id"], hint_scale)
        out.append(x.numpy())
        if len(out) >= n:
            break
    return out


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = load_cfg(args.config)
    profile = load_mark(cfg["mark"])
    calib = real_windows(cfg, profile, args.calib, seed=7)
    test = real_windows(cfg, profile, args.test, seed=99)

    class Reader(CalibrationDataReader):
        def __init__(self, xs):
            self.it = iter(xs)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"x": x}

    src = Path(args.onnx)
    out = Path(args.out) if args.out else src.with_suffix(".int8.onnx")
    prep = src.with_suffix(".prep.onnx")
    quant_pre_process(str(src), str(prep))
    quantize_static(str(prep), str(out), Reader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.999})
    prep.unlink(missing_ok=True)

    hint_scale = float(cfg["data"].get("hint_scale", 8.0))
    results = {}
    for name, path in (("fp32", src), ("int8", out)):
        sess = session(path, args.threads)
        amps, outs = [], []
        t = 0.0
        for x in test:
            t0 = time.perf_counter()
            delta = sess.run(None, {"x": x})[0]
            t += time.perf_counter() - t0
            pred = np.clip(x[:, :3] - delta, -1, 1)[0].transpose(1, 2, 0)
            img = np.rint((pred + 1) * 127.5).clip(0, 255).astype(np.uint8)
            amps.append(mark_amplitude(img, x[0, 6] / hint_scale, profile.overlay))
            outs.append(img.astype(np.float32))
        results[name] = (np.array(amps), outs, t / len(test))
        print(f"{name}: {results[name][2] * 1000:.1f} ms/window, leftover amplitude median "
              f"{np.median(results[name][0]):.3f}, size {path.stat().st_size / 1e6:.1f} MB")
    diff = np.array([np.abs(a - b).mean() for a, b in zip(results["fp32"][1], results["int8"][1])])
    mx = np.array([np.abs(a - b).max() for a, b in zip(results["fp32"][1], results["int8"][1])])
    print(f"int8 vs fp32 output: mean |diff| {diff.mean():.3f} levels, max {mx.max():.0f} levels")


if __name__ == "__main__":
    main()
