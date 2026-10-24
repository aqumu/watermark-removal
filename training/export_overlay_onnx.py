"""
export_overlay_onnx.py  export an overlay-removal model to ONNX and time it on CPU
--------------------------------------------------------------------------------
Usage:
  python training/export_overlay_onnx.py --checkpoint best.pth --config cfg.yaml --out model.onnx
  python training/export_overlay_onnx.py --widths 48,64,96,128@2 --bench-only   # random weights, timing only

The exported graph takes the 7-channel input from features.build_input and
returns the 3-channel residual, with a dynamic batch dimension.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tasks.overlay.model import OverlayUNet, build_model
from wm_shared.marks import load_mark


def export(model: torch.nn.Module, path: Path, shape) -> None:
    model.eval()
    x = torch.randn(1, *shape)
    torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["residual"],
                      dynamic_axes={"x": {0: "batch"}, "residual": {0: "batch"}}, opset_version=17,
                      dynamo=False)


def time_ort(path: Path, shape, batch: int, threads: int, reps: int = 15) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, *shape).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"x": x})
    t = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"x": x})
    return (time.perf_counter() - t) / reps


def time_torch(model, shape, batch: int, threads: int, reps: int = 15) -> float:
    torch.set_num_threads(threads)
    x = torch.randn(batch, *shape)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t) / reps


def check_equal(model, path: Path, shape) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(2, *shape)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"x": x.numpy()})[0]
    return float(np.abs(out - ref).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--widths", nargs="*", default=[], help="random-weight variants, e.g. 48,64,96,128@2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args()
    if args.config:
        from train_overlay import load_cfg
        cfg_for_shape = load_cfg(args.config)
        mark_name = cfg_for_shape["mark"]
    else:
        mark_name = "cian"
    ww, wh = load_mark(mark_name).window
    shape = (7, wh, ww)

    models = []
    if args.checkpoint:
        from train_overlay import load_cfg
        cfg = load_cfg(args.config)
        model = build_model(cfg)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
        models.append((Path(args.checkpoint).name, model, Path(args.out) if args.out else None))
    for wstr in args.widths:
        wpart, _, upart = wstr.partition("@")
        model = OverlayUNet(widths=tuple(int(v) for v in wpart.split(",")), unshuffle=int(upart or 1))
        models.append((wstr, model, None))

    tmp = Path(tempfile.mkdtemp())
    for name, model, out in models:
        model.eval()
        path = out or tmp / f"{name.replace(',', '_').replace('@', 'u')}.onnx"
        export(model, path, shape)
        diff = check_equal(model, path, shape)
        line = [f"{name}: onnx {path.stat().st_size / 1e6:.1f} MB, max|onnx-torch|={diff:.2e}"]
        for b in args.batch:
            tt = time_torch(model, shape, b, args.threads)
            to = time_ort(path, shape, b, args.threads)
            line.append(f"batch {b}: torch {tt / b * 1000:.1f} ms/window, onnxruntime {to / b * 1000:.1f} ms/window")
        print("\n  ".join(line), flush=True)


if __name__ == "__main__":
    main()
