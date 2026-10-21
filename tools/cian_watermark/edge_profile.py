"""Step 10: the mark's edge profile on flat backgrounds, real vs synthetic (model-free).

Per aligned mark (fit_halo._crop: real photos, or the current template stamped
on clean regions of the same photos and JPEG-compressed) the mean of
out - background by distance to the glyph edge, kept only for flat, untextured
backgrounds where the inpainted background is reliable. Comparing real with
synthetic profiles cancels what the measurement itself adds (the background
estimate is pulled up next to a bright mark, so both show a slight dark rim).

  collect: python edge_profile.py collect real
           python edge_profile.py collect synth [sharpness k, as in training]
           python edge_profile.py collect synth 0 <rim gain> <rim sigma>
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


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, -cv2.distanceTransform(mask, cv2.DIST_L2, 5) + 0.5,
                    cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5) - 0.5)


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
        valid = (bb > 6) & (bb < 249) & (g > 1) & (g < 254)
        mask = F.glyph_mask(m)
        logo_mask = F.glyph_mask(F._S["logo"])
        text_mask = F.glyph_mask(m - F._S["logo"])
        profs = []
        for gm, other in ((mask, None), (logo_mask, text_mask), (text_mask, logo_mask)):
            d = _signed_distance(gm)
            # logo / text: only pixels closer to their own glyphs than to the other part
            own = np.ones(gm.shape, bool) if other is None else np.abs(d) < np.abs(_signed_distance(other))
            prof = []
            for k in DS:
                s = valid & own & (d >= k) & (d < k + 1)
                prof.append((g - bb)[s].mean() if s.sum() > 20 else np.nan)
            profs.append(prof)
        out.append((texture, gradient, float(np.median(bb[mask > 0])), style, *profs))
    return out


def collect(mode: str, sharp: float, rim: tuple[float, float]) -> None:
    rows = list(csv.DictReader(open(F.REPO / "data_gen/backgrounds/cian_index.csv", encoding="utf-8")))
    random.Random(17).shuffle(rows)
    rows = rows[:12000]
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    res = []
    with Pool(6, initializer=F._init, initargs=(mode, 0.0, 1.0, sharp, rim)) as pool:
        for o in pool.imap_unordered(_work, chunks):
            res += o
    name = mode + (f"{sharp}" if sharp else "") + (f"_rim{rim[0]}s{rim[1]}" if rim[0] else "")
    pickle.dump(res, open(F.OUT_DIR / f"edge_profile_{name}.pkl", "wb"))
    print(name, len(res), "marks")


def _profile(res, lo, hi, part=4, max_texture=1.5, max_gradient=3.0):
    sel = [r for r in res if r[0] <= max_texture and r[1] <= max_gradient and lo <= r[2] < hi]
    P = np.array([r[part] for r in sel])
    return len(sel), np.nanmean(P, 0), np.nanstd(P, 0) / np.sqrt(len(sel))


def compare(part: int = 4) -> None:
    """part: 4 whole mark, 5 logo, 6 text line."""
    files = sorted(F.OUT_DIR.glob("edge_profile_*.pkl"))
    runs = {f.stem.removeprefix("edge_profile_"): pickle.load(open(f, "rb")) for f in files}
    runs = {k: v for k, v in runs.items() if v and len(v[0]) > part}
    print({4: "whole mark", 5: "logo", 6: "text line"}[part])
    for lo, hi in [(20, 90), (90, 150), (150, 230)]:
        print(f"\nbackground {lo}-{hi}: mean (out - bg) by distance to the glyph edge, scaled to the real "
              f"interior level (d = -4..-3) | squared error vs real, d = -3..+3")
        print("                      " + " ".join(f"{d:+6d}" for d in DS[:12]))
        base = _profile(runs["real"], lo, hi, part)
        for name, res in runs.items():
            n, P, _ = _profile(res, lo, hi, part)
            P = P * np.nanmean(base[1][:2]) / np.nanmean(P[:2])
            err = np.nansum((P - base[1])[1:8] ** 2)
            print(f"  {name:14s} n={n:4d} " + " ".join(f"{v:+6.2f}" for v in P[:12]) + f" | {err:6.2f}")
        print("  real +-2se            " + " ".join(f"{2 * v:6.2f}" for v in base[2][:12]))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        a = [float(v) for v in sys.argv[3:]] + [0.0, 0.0, 1.0][len(sys.argv) - 3:]
        collect(sys.argv[2], a[0], (a[1], a[2]))
    else:
        compare(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
