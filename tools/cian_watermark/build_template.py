"""Step 5: locate, register and stack the mark over thousands of parser-stored JPEGs.

Uses out/cian_index.csv (collect.py) and a seed template (refine.py's
out/m_refined.npy). For each sampled photo: coarse NCC search of the logo detail
layer around the anchor prior, then sub-pixel registration by model fit. The
photos with the strongest evidence are restacked into a new logo estimate.

Writes out/jpeg_positions.json and out/m_jpeg.npy.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from multiprocessing import Pool

from common import *
from relocate import expected_detail, logo_mask
from stack import FRAME_X0, FRAME_Y0, FRAME_W, FRAME_H, aligned_crop, estimate, smooth_background, \
    flatness_weight, save_vis

C = 234.0
_T = {}  # per-worker template state


def anchor_prior(w: int, h: int) -> tuple[float, float]:
    """Expected wordmark top-left (fit on the WebP samples)."""
    return w - (0.1002 * w + 157.15), h - (0.0863 * h + 66.80)


def shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    mat = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _init(m: np.ndarray) -> None:
    mask, _, _ = logo_mask(m)
    support = cv2.dilate((cv2.GaussianBlur(m, (0, 0), 1.0) > 0.015).astype(np.uint8),
                         np.ones((5, 5), np.uint8)).astype(np.float32)
    # only the shared logo takes part in registration (digits differ per listing)
    _T.update(m=m * (mask > 0), support=support * (mask > 0), detail=expected_detail(m, C) * mask)


def _fit_cost(w, d, a, m, dx, dy) -> float:
    return float((w * np.abs(d - shift(m, dx, dy) * a)).sum())


def process(row: dict) -> dict | None:
    try:
        g = luma(load_rgb(Path(row["path"])))
    except Exception:
        return None
    h, w = g.shape
    xg, yg = anchor_prior(w, h)
    radius = 30
    ox, oy = int(round(xg)) + FRAME_X0 - radius, int(round(yg)) + FRAME_Y0 - radius
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + FRAME_W + 2 * radius), min(h, oy + FRAME_H + 2 * radius)
    region = highpass(g[y0:y1, x0:x1], 21)
    if region.shape[0] < FRAME_H or region.shape[1] < FRAME_W:
        return None
    score = cv2.matchTemplate(region, _T["detail"].astype(np.float32), cv2.TM_CCOEFF_NORMED)
    yy, xx = np.unravel_index(int(np.abs(score).argmax()), score.shape)
    xi, yi = x0 + int(xx) - FRAME_X0, y0 + int(yy) - FRAME_Y0
    # registration by model fit around the integer peak
    o = aligned_crop(g, xi, yi)
    sup = _T["support"]
    b = smooth_background(o, sup, 5.0)
    wt = flatness_weight(o, b, sup, 5.0) * sup
    d, a = o - b, C - b
    base = float((wt * np.abs(d)).sum())
    best = (np.inf, 0.0, 0.0)
    for iy in range(-2, 3):
        for ix in range(-2, 3):
            best = min(best, (_fit_cost(wt, d, a, _T["m"], ix, iy), ix, iy))
    for step in (0.25, 0.05):
        _, bx, by = best
        for sy in np.arange(-4, 5) * step:
            for sx in np.arange(-4, 5) * step:
                best = min(best, (_fit_cost(wt, d, a, _T["m"], bx + sx, by + sy), bx + sx, by + sy))
    cost, dx, dy = best
    return {"path": row["path"], "listing_id": row["listing_id"], "size": [w, h],
            "x": float(xi + dx), "y": float(yi + dy), "ncc": float(score[yy, xx]),
            "gain": 1.0 - cost / max(base, 1e-6)}


def sample_rows(n: int, per_listing: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(OUT_DIR / "cian_index.csv", encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    taken: dict[str, int] = {}
    out = []
    for r in rows:
        if taken.get(r["listing_id"], 0) >= per_listing:
            continue
        taken[r["listing_id"]] = taken.get(r["listing_id"], 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def restack(results: list[dict], min_gain: float, limit: int) -> tuple[np.ndarray, int]:
    good = sorted([r for r in results if r["gain"] >= min_gain], key=lambda r: -r["gain"])[:limit]
    crops = []
    for r in good:
        crops.append(aligned_crop(luma(load_rgb(Path(r["path"]))), r["x"], r["y"]))
    m, _, _ = estimate(crops, c_fixed=C)
    return m, len(good)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--per-listing", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.2)
    ap.add_argument("--stack-limit", type=int, default=1500)
    ap.add_argument("--seed-template", default=str(OUT_DIR / "m_refined.npy"))
    args = ap.parse_args()

    rows = sample_rows(args.n, args.per_listing, seed=0)
    m = np.load(args.seed_template)
    for it in range(args.iters):
        with Pool(12, initializer=_init, initargs=(m,)) as pool:
            results = [r for r in pool.imap_unordered(process, rows, chunksize=8) if r is not None]
        gains = np.array([r["gain"] for r in results])
        print(f"iter {it}: {len(results)} photos, gain percentiles 10/50/90 = "
              f"{np.percentile(gains, [10, 50, 90]).round(3)}, >= {args.min_gain}: {(gains >= args.min_gain).sum()}")
        m_new, used = restack(results, args.min_gain, args.stack_limit)
        # keep the listing-specific digit area from the previous estimate out of the logo
        m = m_new
        print(f"  restacked from {used} photos")
        np.save(OUT_DIR / f"m_jpeg_{it}.npy", m)
    np.save(OUT_DIR / "m_jpeg.npy", m)
    save_vis("m_jpeg.png", m)
    (OUT_DIR / "jpeg_positions.json").write_text(json.dumps(results))
