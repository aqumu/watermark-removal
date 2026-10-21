"""Step 10: the mark's edge profile on flat backgrounds, real vs synthetic (model-free).

Per aligned mark (fit_halo._crop: real photos, or the current template stamped
on clean regions of the same photos and JPEG-compressed) the mean of
out - background by distance to the glyph edge, kept only for flat, untextured
backgrounds where the inpainted background is reliable. Comparing real with
synthetic profiles cancels what the measurement itself adds (the background
estimate is pulled up next to a bright mark, so both show a slight dark rim).

  collect: python edge_profile.py collect real
           python edge_profile.py collect synth [sharpness k, as in training]
  compare: python edge_profile.py compare
"""
from __future__ import annotations

import csv
import pickle
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

import fit_halo as F

DS = np.arange(-4, 16)          # signed distance bins (px), < 0 inside the glyphs


def _work(rows):
    out = []
    for row in rows:
        got = F._crop(row)
        if got is None:
            continue
        crop, b, _, _, m, style = got
        g, bb = crop.mean(-1), b.mean(-1)
        keep = cv2.dilate(F.glyph_mask(m) | (m > 0.003).astype(np.uint8), np.ones((37, 37), np.uint8)) == 0
        if keep.mean() < 0.2:
            continue
        texture = float(np.median(np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))[keep]))
        gradient = float(np.percentile(np.abs(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 6), cv2.CV_32F, 1, 1)), 90))
        mask = F.glyph_mask(m)
        d = np.where(mask > 0, -cv2.distanceTransform(mask, cv2.DIST_L2, 5) + 0.5,
                     cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5) - 0.5)
        valid = (bb > 6) & (bb < 249) & (g > 1) & (g < 254)
        prof = []
        for k in DS:
            s = valid & (d >= k) & (d < k + 1)
            prof.append((g - bb)[s].mean() if s.sum() > 20 else np.nan)
        out.append((texture, gradient, float(np.median(bb[mask > 0])), style, prof))
    return out


def collect(mode: str, sharp: float) -> None:
    rows = list(csv.DictReader(open(F.REPO / "data_gen/backgrounds/cian_index.csv", encoding="utf-8")))
    random.Random(17).shuffle(rows)
    rows = rows[:16000]
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    res = []
    with Pool(6, initializer=F._init, initargs=(mode, 0.0, 1.0, sharp)) as pool:
        for o in pool.imap_unordered(_work, chunks):
            res += o
    name = mode + (f"{sharp}" if sharp else "")
    pickle.dump(res, open(F.OUT_DIR / f"edge_profile_{name}.pkl", "wb"))
    print(name, len(res), "marks")


def _profile(res, lo, hi, max_texture=1.5, max_gradient=3.0):
    sel = [r for r in res if r[0] <= max_texture and r[1] <= max_gradient and lo <= r[2] < hi]
    P = np.array([r[4] for r in sel])
    return len(sel), np.nanmean(P, 0), np.nanstd(P, 0) / np.sqrt(len(sel))


def compare() -> None:
    files = sorted(F.OUT_DIR.glob("edge_profile_*.pkl"))
    runs = {f.stem.removeprefix("edge_profile_"): pickle.load(open(f, "rb")) for f in files}
    for lo, hi in [(20, 90), (90, 150), (150, 230)]:
        print(f"\nbackground {lo}-{hi}: mean (out - bg) by distance to the glyph edge, "
              f"scaled to the real interior level (d = -4..-3)")
        print("                  " + " ".join(f"{d:+6d}" for d in DS[:12]))
        base = _profile(runs["real"], lo, hi)
        for name, res in runs.items():
            n, P, _ = _profile(res, lo, hi)
            P = P * base[1][:2].mean() / P[:2].mean()
            print(f"  {name:10s} n={n:4d} " + " ".join(f"{v:+6.2f}" for v in P[:12]))
        print(f"  real +-2se        " + " ".join(f"{2 * v:6.2f}" for v in base[2][:12]))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
    else:
        compare()
