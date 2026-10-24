"""Step 2: scale and position of the artwork on the corner mean of all photos.

Grid-searches width and sub-pixel offset so the rendered artwork's detail layer
best matches the detail layer of the mean corner (NCC).
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from common import CH, CW, OUT, detail, index, load_rgb, luma, render_logo


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    rows = index()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), 3000, replace=False)
    acc = np.zeros((CH, CW), np.float64)
    for i in pick:
        acc += luma(load_rgb(rows[i]["path"])[-CH:, -CW:])
    mean = (acc / len(pick)).astype(np.float32)
    np.save(OUT / "corner_mean_luma.npy", mean)
    dm = detail(mean)
    best = (-1, None)
    for width in np.arange(88, 102, 1.0):
        h = width * 2032 / 8001
        for x in np.arange(CW - width - 16, CW - width + 2, 1.0):
            for y in np.arange(CH - h - 22, CH - h - 6, 1.0):
                s = ncc(dm, detail(render_logo(width, x, y)))
                if s > best[0]:
                    best = (s, (width, x, y))
    s, (width, x, y) = best
    for step in (0.5, 0.25, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for d in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                cand = (width + d[0], x + d[1], y + d[2])
                c = ncc(dm, detail(render_logo(*cand)))
                if c > s:
                    s, (width, x, y), improved = c, cand, True
    h = width * 2032 / 8001
    res = {"width": round(width, 3), "height": round(h, 3), "x": round(x, 3), "y": round(y, 3),
           "right_margin": round(CW - x - width, 3), "bottom_margin": round(CH - y - h, 3), "ncc": round(s, 4)}
    print(res)
    (OUT / "seed.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
