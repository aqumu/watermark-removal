"""Step 9: a trained removal model on the held-out photos, measured like accuracy.py.

For every held-out photo (never used for any fit, and "val" for training) the
leftover mark in the frame is the least-squares amplitude k of the mark's pattern
(the mean real effect, out/realest_*.npy), relative to the input (k_none). Compared:
the input, the exact inverse with the nominal mark (assets/marks/avito, the hint)
and the model. Also writes out/eval_<tag>.jpg: input, inverse, model, x5 contrast.

Usage:
  python eval_model.py --config training/configs/avito_v1.yaml --checkpoint <best.pth> [--tag v1]
  python eval_model.py --onnx model_avito_v1.onnx [--tag v1_onnx]
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from common import OUT, REPO, index, split

sys.path[:0] = [str(REPO), str(REPO / "training")]

from accuracy import FH, FW, _init, amplitude  # noqa: E402
from wm_shared.marks import load_mark  # noqa: E402


def contrast(x: np.ndarray, gain: float = 5.0) -> np.ndarray:
    base = cv2.GaussianBlur(x, (0, 0), 6)
    return np.clip(base + (x - base) * gain, 0, 255)


def load_runner(args, profile):
    """Returns f(batch of window uint8 N x wh x ww x 3) -> cleaned windows (uint8)."""
    import torch
    from src.tasks.overlay.features import build_input_layered
    alpha, a = (torch.from_numpy(m.transpose(2, 0, 1).copy())[None] for m in profile.window_maps())
    if args.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def run(x):
            return torch.from_numpy(np.concatenate([sess.run(None, {name: x[i:i + 1].numpy()})[0]
                                                    for i in range(x.shape[0])]))
    else:
        from src.tasks.overlay.model import build_model
        from train_overlay import load_cfg
        cfg = load_cfg(args.config)
        model = build_model(cfg)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        # the averaged weights (what validation measures) unless --raw
        model.load_state_dict(ckpt["ema"]["shadow"] if "ema" in ckpt and not args.raw else ckpt["model"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev).eval()

        def run(x):
            with torch.no_grad():
                return model(x.to(dev)).float().cpu()

    def clean(wins: np.ndarray) -> np.ndarray:
        wm = torch.from_numpy(np.ascontiguousarray(wins.transpose(0, 3, 1, 2)))
        x = build_input_layered(wm, alpha, a, 8.0)
        pred = (x[:, :3] - run(x)).clamp(-1, 1)
        return ((pred.numpy().transpose(0, 2, 3, 1) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--raw", action="store_true", help="the checkpoint's raw weights instead of the EMA")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    profile = load_mark("avito")
    ww, wh = profile.window
    al, A = profile.alpha, profile.a
    clean = load_runner(args, profile)
    _init(["none"])
    rows = [r for r in index(200, 200) if split(r["sha256"]) == "holdout"]
    rows = rows[:args.n] if args.n else rows
    res, examples = [], []
    for i in range(0, len(rows), 64):
        chunk = [(r, cv2.imread(r["path"], cv2.IMREAD_COLOR)) for r in rows[i:i + 64]]
        chunk = [(r, img[..., ::-1]) for r, img in chunk if img is not None]
        outs = clean(np.stack([img[-wh:, -ww:] for _, img in chunk]))
        for (r, img), out in zip(chunk, outs):
            frame = img[-FH:, -FW:].astype(np.float32)
            inv = np.clip((frame - A) / np.maximum(1 - al, 0.05), 0, 255)
            mod = out[-FH:, -FW:].astype(np.float32)
            free = img[40:40 + FH, 40:40 + FW].astype(np.float32)
            res.append({"sha": r["sha256"], "bg": float(frame.mean()), "none": amplitude(frame)[0],
                        "inverse": amplitude(inv)[0], "model": amplitude(mod)[0], "noise": amplitude(free)[0]})
            if len(examples) < 12 and len(res) % 97 == 1:
                examples.append(np.concatenate([frame, inv, mod, contrast(frame), contrast(inv), contrast(mod)], 1))
        print(f"{len(res)}/{len(rows)}", end="\r", flush=True)
    (OUT / f"eval_{args.tag}.json").write_text(json.dumps(res))
    sheet = np.concatenate(examples, 0)
    cv2.imwrite(str(OUT / f"eval_{args.tag}.jpg"),
                cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)[..., ::-1].astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    k0 = np.array([r["none"] for r in res])
    print(f"\nheld-out photos: {len(res)}; leftover relative to the input (|k / k_none|), median k_none {np.median(k0):.3f}")
    print(f"{'':8s} {'median':>7s} {'p90':>7s} {'p99':>7s} {'>17%':>6s} {'mean k':>7s}")
    for n in ("inverse", "model", "noise"):
        k = np.array([r[n] for r in res]) / np.median(k0)
        rel = np.abs(k)
        print(f"{n:8s} {np.median(rel):7.1%} {np.percentile(rel, 90):7.1%} {np.percentile(rel, 99):7.1%} "
              f"{(rel > 0.17).mean():6.1%} {k.mean():+7.1%}")
    bg = np.array([r["bg"] for r in res])
    for lo, hi in ((0, 80), (80, 170), (170, 256)):
        sel = (bg >= lo) & (bg < hi)
        rel = lambda n: np.median(np.abs(np.array([r[n] for r in res])[sel] / np.median(k0)))  # noqa: E731
        print(f"bg {lo:3d}-{hi:3d} ({sel.sum():4d}): inverse {rel('inverse'):.1%}  model {rel('model'):.1%}")


if __name__ == "__main__":
    main()
