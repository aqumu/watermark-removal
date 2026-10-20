"""Step 10: per-photo scale of the mark.

The mark is not always stamped at the same size: regular-style photos fit the
logo best at ~98% scale. For every registered photo, fit the logo's scale (about
the logo centre) and sub-pixel shift by model fit, and classify the text style
at that scale. Writes out/scale_fit.json and prints the distribution per style.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import *  # noqa: E402
from stack import aligned_crop, smooth_background, flatness_weight  # noqa: E402
from wm_shared.marks import load_mark  # noqa: E402

P = None
SCALES = np.round(np.arange(0.95, 1.0501, 0.005), 3)
CENTRE = (135.0, 70.0)   # scaling centre in frame coordinates (middle of the mark)


def _init():
    global P
    P = load_mark("cian")


def scale_map(m: np.ndarray, s: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    cx, cy = CENTRE
    M = np.array([[s, 0, cx * (1 - s) + dx], [0, s, cy * (1 - s) + dy]], np.float32)
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LANCZOS4)


def fit(q: dict) -> dict | None:
    try:
        o = aligned_crop(luma(load_rgb(Path(q["path"]))), q["x"], q["y"])
    except Exception:
        return None
    logo = P.logo
    sup = cv2.dilate((logo > 0.01).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    band = np.zeros_like(sup); band[96:132, 40:280] = 1
    b = smooth_background(o, np.maximum(sup, band), 5.0)
    w = flatness_weight(o, b, np.maximum(sup, band), 5.0) * sup
    d, a = o - b, P.overlay - b
    best = (np.inf, 1.0, 0.0, 0.0)
    for s in SCALES:
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
                mm = scale_map(logo, s, dx, dy)
                amp = (w * d * mm * a).sum() / max((w * (mm * a) ** 2).sum(), 1e-6)
                cost = float((w * (d - amp * mm * a) ** 2).sum())
                if cost < best[0]:
                    best = (cost, float(s), dx, dy)
    c1 = None
    for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
            mm = scale_map(logo, 1.0, dx, dy)
            amp = (w * d * mm * a).sum() / max((w * (mm * a) ** 2).sum(), 1e-6)
            cost = float((w * (d - amp * mm * a) ** 2).sum())
            c1 = cost if c1 is None else min(c1, cost)
    # style at the fitted scale: which text rendering (scaled the same way) fits the band best
    styles = {}
    for st in P.styles:
        t = scale_map(P.render_text_only(q["listing_id"], st), best[1], best[2], best[3])
        wt = flatness_weight(o, b, np.maximum(sup, band), 5.0) * band
        amp = (wt * d * t * a).sum() / max((wt * (t * a) ** 2).sum(), 1e-6)
        styles[st] = float(1 - (wt * (d - amp * t * a) ** 2).sum() / max((wt * d * d).sum(), 1e-6))
    return {"path": q["path"], "listing_id": q["listing_id"], "size": q["size"], "gain": q["gain"],
            "scale": best[1], "cost_ratio_1": c1 / max(best[0], 1e-6), "styles": styles}


if __name__ == "__main__":
    positions = sorted([q for q in json.loads((OUT_DIR / "jpeg_positions.json").read_text()) if q["gain"] >= 0.3],
                       key=lambda q: -q["gain"])
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6, initializer=_init) as pool:
        res = [r for r in pool.imap_unordered(fit, positions, chunksize=4) if r is not None]
    (OUT_DIR / "scale_fit.json").write_text(json.dumps(res))
    sc = np.array([r["scale"] for r in res])
    print(f"{len(res)} photos; scale p5/25/50/75/95 = {np.percentile(sc, [5, 25, 50, 75, 95]).round(3)}")
    hist = defaultdict(int)
    for s in sc:
        hist[round(float(s), 3)] += 1
    print("histogram:", dict(sorted(hist.items())))
    for st in ("bold", "regular"):
        sel = np.array([r["scale"] for r in res if max(r["styles"], key=r["styles"].get) == st and r["styles"][st] > 0.05])
        if len(sel):
            print(f"style {st}: n={len(sel)} scale median {np.median(sel):.3f} p10/p90 {np.percentile(sel, [10, 90]).round(3)}")
    by_size = defaultdict(list)
    for r in res:
        by_size[r["size"][0]].append(r["scale"])
    print("median scale by image width (n>=15):", {k: round(float(np.median(v)), 3) for k, v in sorted(by_size.items()) if len(v) >= 15})
