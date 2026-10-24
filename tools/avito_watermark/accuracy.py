"""Step 7: accuracy of a mark model on held-out photos (never used for any fit).

Every held-out photo's mark frame is inverted with the model's per-pixel affine map
(b = (out - A) / (1 - Alpha)); then the leftover mark is measured as a least-squares
amplitude k of the mark's own pattern in the result:
    pattern  E = A - Alpha * b_smooth        (what the mark adds on this background)
    k        = <hp(result), hp(E)> / <hp(E), hp(E)>  over the mark's support (hp = detail layer)
Input (no removal) gives k ~ 1, a perfect model k ~ 0. Noise floor: the same k measured
on a mark-free region (top-left of the same photo) where nothing was stamped.
Also the leftover in levels: rms of k * E over the support, and the mean absolute
error of the recovered background against the photo's own surroundings is NOT used
(no ground truth); k is the measure.

Models: out/<name>_alpha.npy / _a.npy, plus "none" (identity) and "codex" (the guessed
104x26 mark at 11% that the first avito_small_v1 run was trained on).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import OUT, index, render_logo, split

FH, FW = 56, 124
_S = {}


def codex_mark():
    from PIL import Image
    src = np.asarray(Image.open(OUT.parents[2] / "data_gen" / "watermark.png").convert("RGBA"))[..., 3].astype(np.float32) / 255.0
    logo = cv2.resize(src, (104, 26), interpolation=cv2.INTER_AREA) * 0.11
    al = np.zeros((FH, FW), np.float32)
    al[19:45, 16:120] = logo                    # frame (W-112, H-48), logo at +4, +11
    return al[..., None].repeat(3, -1), (al * 255.0)[..., None].repeat(3, -1)


def hp(x):
    return x - cv2.GaussianBlur(x, (0, 0), 2.0)


def _init(names):
    _S["models"] = {}
    for n in names:
        if n == "none":
            _S["models"][n] = None
        elif n == "codex":
            _S["models"][n] = codex_mark()
        else:
            _S["models"][n] = (np.load(OUT / f"{n}_alpha.npy"), np.load(OUT / f"{n}_a.npy"))
    ref_al, ref_a = np.load(OUT / "realest_alpha.npy"), np.load(OUT / "realest_a.npy")   # measuring pattern: the mean real effect
    _S["ref"] = (ref_al, ref_a)
    sup = cv2.dilate((np.load(OUT / "seed_alpha.npy") > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    sup[:, -2:] = False; sup[-2:, :] = False
    _S["sup"] = sup


def amplitude(img):
    al, a = _S["ref"]
    bs = cv2.GaussianBlur(img, (0, 0), 3.0)
    E = a - al * bs
    he, hr = hp(E)[_S["sup"]], hp(img)[_S["sup"]]
    k = float((he * hr).sum() / (he * he).sum())
    lev = float(np.sqrt(((k * E)[_S["sup"]] ** 2).mean()))
    return k, lev


def _one(row):
    bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[0] < 200 or bgr.shape[1] < 200:
        return None
    rgb = bgr[..., ::-1].astype(np.float32)
    frame = rgb[-FH:, -FW:]
    free = rgb[40:40 + FH, 40:40 + FW]
    out = {"sha": row["sha256"], "bg": float(frame.mean()), "noise": amplitude(free)[0]}
    for n, m in _S["models"].items():
        if m is None:
            res = frame
        else:
            al, a = m
            res = np.clip((frame - a) / np.maximum(1 - al, 0.05), 0, 255)
        out[n] = amplitude(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="none,codex,model,ff")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    names = args.models.split(",")
    rows = [r for r in index(200, 200) if split(r["sha256"]) == "holdout"]
    if args.n:
        rows = rows[:args.n]
    with ProcessPoolExecutor(5, initializer=_init, initargs=(names,)) as ex:
        res = [r for r in ex.map(_one, rows, chunksize=32) if r]
    (OUT / "accuracy.json").write_text(json.dumps(res))
    noise = np.abs([r["noise"] for r in res])
    print(f"held-out photos: {len(res)}; noise floor |k| median {np.median(noise):.3f} p90 {np.percentile(noise, 90):.3f}")
    print(f"{'model':8s} {'|k| med':>8s} {'p90':>6s} {'p99':>6s} {'k mean':>7s} {'>0.1':>6s} {'>0.2':>6s} {'levels med':>10s} {'p90':>6s}")
    for n in names:
        k = np.array([r[n][0] for r in res]); lv = np.array([r[n][1] for r in res])
        print(f"{n:8s} {np.median(np.abs(k)):8.3f} {np.percentile(np.abs(k), 90):6.3f} {np.percentile(np.abs(k), 99):6.3f} "
              f"{k.mean():7.3f} {(np.abs(k) > 0.1).mean():6.1%} {(np.abs(k) > 0.2).mean():6.1%} {np.median(lv):10.2f} {np.percentile(lv, 90):6.2f}")


if __name__ == "__main__":
    main()
