"""Step 4: per-pixel, per-channel estimate of the mark from the real photos.

The mark sits at a fixed pixel position from the bottom-right corner (locate.py:
+-0.1 px over 25k photos, every image size), so no alignment is needed: the frame
is the same pixels of every photo. Per pixel p and channel k:

    out = b + a(p) - alpha(p) * b        (a = alpha*c - darkening; c = overlay level)

b is the background, estimated by inpainting the mark's support; alpha and a are
solved with the frame's overall background level as an instrument for b (see
tools/cian_watermark/fit_halo.py: plain least squares on an estimated background
is biased towards a fake ring).

  --mode real    the real marks (fit split; --split holdout for the held-out photos)
  --mode synth   control: the model in out/model_*.npy stamped on mark-free regions
                 of the same photos (top-left, same JPEG grid phase), re-encoded as JPEG,
                 estimated the same way. A faithful estimator returns what was stamped.
Writes out/est_<mode>_<split>.npz (alpha, a, per half, counts).
"""
from __future__ import annotations

import argparse
import random
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import OUT, index, split

# frame inside the bottom-right corner of the image: rows/cols from the end
FH, FW = 56, 124          # logo at (18, 18)..(42, 112) of the frame; image edge at the frame's right/bottom
REACH = 8                 # support = glyphs dilated by this (covers a shadow / glow)
_S = {}


def frame_of(img):
    return img[-FH:, -FW:]


def support_mask():
    seed = np.load(OUT / "seed_alpha.npy")
    return cv2.dilate((seed > 0.01).astype(np.uint8), np.ones((2 * REACH + 1, 2 * REACH + 1), np.uint8))


def _prep(crop, support):
    keep = (1 - support).astype(np.float32)
    b = np.stack([cv2.inpaint(np.clip(crop[..., k], 0, 255).astype(np.uint8), support, 5, cv2.INPAINT_TELEA)
                  for k in range(3)], -1).astype(np.float32)
    b = np.where(support[..., None] > 0, cv2.GaussianBlur(b, (0, 0), 2.0), crop)
    g = crop.mean(-1)
    r2 = (g - cv2.GaussianBlur(g, (0, 0), 3.0)) ** 2 * keep
    var = cv2.GaussianBlur(r2, (0, 0), 12) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 12), 1e-3)
    w = 1.0 / (var + 1.0)
    valid = (b > 6) & (b < 249) & (crop > 1) & (crop < 254)
    z = (crop * keep[..., None]).sum((0, 1)) / keep.sum()
    return b, np.where(valid, w[..., None], 0).astype(np.float32), z


def _synth(bgr, rng):
    """Stamp the model on a mark-free top-left region with the real frame's JPEG grid phase."""
    h, w = bgr.shape[:2]
    py, px = (h - FH) % 16, (w - FW) % 16
    region = bgr[32:32 + 96, 32:32 + 160, ::-1].astype(np.float32)
    f = region[py:py + FH, px:px + FW]
    stamped = f + _S["a"] - _S["alpha"] * f
    region = region.copy(); region[py:py + FH, px:px + FW] = stamped
    q = 85 if rng.random() < 0.6 else 80
    ok, enc = cv2.imencode(".jpg", np.clip(np.rint(region), 0, 255).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    return dec[py:py + FH, px:px + FW]


def _work(rows):
    support = _S["support"]
    sums = np.zeros((2, 6, FH, FW, 3), np.float64)
    n = 0
    for row in rows:
        bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[0] < 200 or bgr.shape[1] < 200:
            continue
        if _S["mode"] == "real":
            crop = frame_of(bgr)[..., ::-1].astype(np.float32)
        else:
            crop = _synth(bgr, random.Random(zlib.crc32(row["sha256"].encode())))
        b, w, z = _prep(crop, support)
        d = crop - b
        s = sums[int(row["sha256"][-1], 16) & 1]
        s[0] += w; s[1] += w * b; s[2] += w * d; s[3] += w * z; s[4] += w * z * b; s[5] += w * z * d
        n += 1
    return n, sums


def solve(s):
    sw, sb, sd, sz, szb, szd = s
    sw = np.maximum(sw, 1e-9)
    mb, md, mz = sb / sw, sd / sw, sz / sw
    czb = szb / sw - mz * mb
    czd = szd / sw - mz * md
    slope = czd / np.where(np.abs(czb) > 1e-6, czb, 1e-6)
    return -slope, md - slope * mb


def _init(mode, model="model"):
    _S["mode"] = mode
    _S["support"] = support_mask()
    if mode == "synth":
        _S["alpha"] = np.load(OUT / f"{model}_alpha.npy"); _S["a"] = np.load(OUT / f"{model}_a.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="real", choices=["real", "synth"])
    ap.add_argument("--split", default="fit", choices=["fit", "holdout"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--phase", default="", help="H%%16,W%%16 group, e.g. 0,0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="model", help="synth: stamp out/<model>_alpha.npy / _a.npy")
    args = ap.parse_args()
    rows = [r for r in index(200, 200) if split(r["sha256"]) == args.split]
    if args.phase:
        ph = tuple(int(v) for v in args.phase.split(","))
        rows = [r for r in rows if (r["h"] % 16, r["w"] % 16) == ph]
    if args.n:
        random.Random(0).shuffle(rows); rows = rows[:args.n]
    chunks = [rows[i::args.workers * 8] for i in range(args.workers * 8)]
    total = None; n = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.mode, args.model)) as ex:
        for k, s in ex.map(_work, chunks):
            n += k; total = s if total is None else total + s
    alpha, a = solve(total.sum(0))
    halves = [solve(total[i]) for i in range(2)]
    np.savez(OUT / f"est_{args.mode}_{args.split}{args.tag}.npz", alpha=alpha, a=a,
             alpha0=halves[0][0], a0=halves[0][1], alpha1=halves[1][0], a1=halves[1][1], n=n)
    print(f"{args.mode}/{args.split}: {n} photos; alpha max {alpha.max():.4f}, a range [{a.min():.2f}, {a.max():.2f}]")


if __name__ == "__main__":
    main()
