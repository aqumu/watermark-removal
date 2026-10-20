"""Step 1: find the watermark in every sample photo.

Matches a detail-layer template of the "циан" wordmark (cut from one seed photo,
or from a previous stacking pass) against the bottom-right corner of every photo.
Writes out/locate.json with the wordmark's top-left corner in image coordinates.
"""
from __future__ import annotations

import json
import sys

from common import *

SEED = ("331572984", "05-1.jpg")
SEED_BOX = (195, 125, 340, 165)  # x0, y0, x1, y1 of the wordmark inside the seed's corner crop


def seed_template() -> np.ndarray:
    g = luma(load_rgb(SAMPLES_DIR / SEED[0] / SEED[1]))
    x0, y0 = corner_origin(g.shape)
    hp = highpass(g, 21)[y0:, x0:]
    bx0, by0, bx1, by1 = SEED_BOX
    return hp[by0:by1, bx0:bx1]


def locate_all(template: np.ndarray) -> list[dict]:
    results = []
    for path in list_samples():
        g = luma(load_rgb(path))
        x0, y0 = corner_origin(g.shape)
        hp = highpass(g, 21)[y0:, x0:]
        score = ncc_map(hp, template)
        # The mark lightens dark backgrounds and darkens near-white ones, so accept either sign.
        absscore = np.abs(score)
        y, x = np.unravel_index(int(absscore.argmax()), absscore.shape)
        fy, fx = subpixel_peak(absscore, int(y), int(x))
        h, w = g.shape
        results.append({
            "file": f"{listing_id(path)}/{path.name}",
            "size": [w, h],
            "x": float(x0 + fx), "y": float(y0 + fy),
            "ncc": float(score[y, x]),
        })
    return results


if __name__ == "__main__":
    template = seed_template() if len(sys.argv) < 2 else np.load(sys.argv[1])
    res = locate_all(template)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "locate.json").write_text(json.dumps(res, indent=1))
    for r in res:
        w, h = r["size"]
        print(f"{r['file']:22s} {w:5d}x{h:<5d} x={r['x']:8.2f} y={r['y']:8.2f} "
              f"right={w - r['x']:7.2f} bottom={h - r['y']:7.2f} ncc={r['ncc']:+.3f}")
