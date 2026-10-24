"""Step 3: locate the mark in every photo (position jitter, presence, scale vs image size).

NCC of the corner's detail layer with the artwork's detail layer at the seed
width, searched +-16 px around the seed position, sub-pixel by a parabola.
Then a per-photo width search (0.94..1.06) on the strong matches.
Writes out/positions.json.
"""
from __future__ import annotations

import json
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import CH, CW, OUT, detail, index, load_rgb, luma, render_logo

SEED = json.loads((OUT / "seed.json").read_text())
PAD = 16


@lru_cache(None)
def _tmpl(width, fx=0.0, fy=0.0):
    h = width * 2032 / 8001
    tw, th = int(np.ceil(width)) + 8, int(np.ceil(h)) + 8
    t = render_logo(width, 4 + fx, 4 + fy, frame=(tw, th))
    return detail(t)


def _peak(score, y, x):
    def par(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dy = par(score[y - 1, x], score[y, x], score[y + 1, x]) if 0 < y < score.shape[0] - 1 else 0
    dx = par(score[y, x - 1], score[y, x], score[y, x + 1]) if 0 < x < score.shape[1] - 1 else 0
    return y + dy, x + dx


def locate_one(row):
    try:
        g = luma(load_rgb(row["path"])[-CH:, -CW:])
    except Exception:
        return None
    d = detail(g).astype(np.float32)
    best = None
    for width in (SEED["width"] * f for f in (0.94, 0.97, 1.0, 1.03, 1.06)):
        t = _tmpl(width)
        sc = cv2.matchTemplate(d, t, cv2.TM_CCOEFF_NORMED)
        # restrict to +-PAD around the seed position
        x0 = int(SEED["x"] - 4 - PAD); y0 = int(SEED["y"] - 4 - PAD)
        sub = sc[max(0, y0):y0 + 2 * PAD + 1, max(0, x0):x0 + 2 * PAD + 1]
        a = np.abs(sub)
        y, x = np.unravel_index(int(a.argmax()), a.shape)
        s = float(sub[y, x])
        if best is None or abs(s) > abs(best[0]):
            py, px = _peak(a, y, x)
            best = (s, width, max(0, x0) + px + 4, max(0, y0) + py + 4)
    s, width, x, y = best
    return {"sha": row["sha256"], "size": [row["w"], row["h"]], "ncc": round(float(s), 4), "width": round(float(width), 3),
            "x": round(float(x), 3), "y": round(float(y), 3)}


def main():
    rows = index()
    with ProcessPoolExecutor(5) as ex:
        res = [r for r in ex.map(locate_one, rows, chunksize=64) if r]
    (OUT / "positions.json").write_text(json.dumps(res))
    nc = np.array([r["ncc"] for r in res])
    print(f"n={len(res)}  ncc pct 5/25/50/75: {np.percentile(nc, [5, 25, 50, 75]).round(3)}  "
          f"<0.2: {(np.abs(nc) < 0.2).mean():.3f}  negative: {(nc < 0).mean():.3f}")
    good = [r for r in res if r["ncc"] > 0.5]
    xs = np.array([r["x"] for r in good]); ys = np.array([r["y"] for r in good]); ws = np.array([r["width"] for r in good])
    print(f"strong (ncc>0.5): {len(good)}; x pct 1/50/99 {np.percentile(xs, [1, 50, 99]).round(2)}  "
          f"y {np.percentile(ys, [1, 50, 99]).round(2)}  width counts {dict(zip(*np.unique(ws, return_counts=True)))}")
    fx = (xs % 1); fy = (ys % 1)
    print("fractional x hist", np.histogram(fx, bins=5, range=(0, 1))[0], "y", np.histogram(fy, bins=5, range=(0, 1))[0])
    by = {}
    for r in good:
        by.setdefault(tuple(r["size"]), []).append(r)
    for sz, lst in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(sz, len(lst), "x med", np.median([r["x"] for r in lst]).round(2), "y med", np.median([r["y"] for r in lst]).round(2),
              "w med", np.median([r["width"] for r in lst]).round(2))


if __name__ == "__main__":
    main()
