"""Step 2: estimate the watermark's per-pixel opacity from many aligned photos.

Model for every photo i and pixel p of the mark frame (luma, before compression):
    O_i(p) = B_i(p) + m(p) * (c - B_i(p))
m = effective opacity (alpha * global opacity), c = overlay grey level.

B_i is estimated by normalised Gaussian smoothing of O_i over pixels outside the
current mark support. Then, per pixel, O_i - B_i = u - m * B_i is a linear
least-squares problem over photos (u = m * c), weighted by how flat each photo's
background is around that pixel. Support, backgrounds and m are refined in a few
passes, and c is pooled from the strong pixels.
"""
from __future__ import annotations

import json
import sys

from common import *

# Mark frame relative to the "циан" wordmark's top-left corner found by locate.py.
FRAME_X0, FRAME_Y0 = -90, -60
FRAME_W, FRAME_H = 270, 150


def aligned_crop(gray: np.ndarray, x: float, y: float) -> np.ndarray:
    """Crop the mark frame with sub-pixel alignment (Lanczos) at (x, y) = wordmark top-left."""
    m = np.array([[1, 0, -(x + FRAME_X0)], [0, 1, -(y + FRAME_Y0)]], np.float32)
    return cv2.warpAffine(gray, m, (FRAME_W, FRAME_H), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def smooth_background(o: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    keep = 1.0 - support
    num = cv2.GaussianBlur(o * keep, (0, 0), sigma)
    den = cv2.GaussianBlur(keep, (0, 0), sigma)
    filled = num / np.maximum(den, 1e-3)
    # outside the support keep the (lightly smoothed) observation itself
    return np.where(support > 0, filled, cv2.GaussianBlur(o, (0, 0), 0.7))


def flatness_weight(o: np.ndarray, b: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    keep = 1.0 - support
    r2 = (o - b) ** 2 * keep
    var = cv2.GaussianBlur(r2, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(keep, (0, 0), sigma), 1e-3)
    return 1.0 / (var + 2.0)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-pixel weighted median along axis 0 (L1 regression through the origin)."""
    order = np.argsort(values, axis=0)
    v = np.take_along_axis(values, order, 0)
    w = np.take_along_axis(weights, order, 0)
    cw = np.cumsum(w, axis=0)
    idx = (cw < 0.5 * cw[-1:]).sum(0, keepdims=True)
    return np.take_along_axis(v, np.minimum(idx, len(v) - 1), 0)[0]


def fit_m(D: np.ndarray, B: np.ndarray, W: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    A = c - B
    usable = np.abs(A) > 12  # backgrounds too close to the overlay colour carry no signal
    ratio = np.where(usable, D / np.where(usable, A, 1), 0)
    weight = np.where(usable, W * A * A, 0) + 1e-9
    m = weighted_median(ratio, weight)
    resid = (W * np.abs(D - m * A)).sum(0)
    return m, resid


def estimate(crops: list[np.ndarray], iters: int = 4, c_fixed: float | None = None, sigma: float = 5.0,
             c_grid=np.arange(196.0, 256.0, 2.0)):
    O = np.stack(crops)  # N x H x W
    support = np.zeros(O.shape[1:], np.float32)
    B = np.stack([cv2.medianBlur(np.clip(o, 0, 255).astype(np.uint8), 21).astype(np.float32) for o in O])
    W = np.ones_like(O)
    c = 230.0 if c_fixed is None else c_fixed
    m = fit_m(O - B, B, W, c)[0]
    for it in range(iters):
        support = (cv2.GaussianBlur(m, (0, 0), 1.0) > 0.015).astype(np.uint8)
        support = cv2.dilate(support, np.ones((5, 5), np.uint8)).astype(np.float32)
        B = np.stack([smooth_background(o, support, sigma) for o in O])
        W = np.stack([flatness_weight(o, b, support, sigma) for o, b in zip(O, B)])
        D = O - B
        if c_fixed is None:
            mask = support > 0
            costs = [fit_m(D, B, W, cc)[1][mask].sum() for cc in c_grid]
            c = float(c_grid[int(np.argmin(costs))])
        m, _ = fit_m(D, B, W, c)
        print(f"  pass {it}: c={c:.1f} support={int(support.sum())} m_p99={np.percentile(m, 99.5):.3f}")
    return m.astype(np.float32), c, support


def select(entries: list[dict], min_ncc: float) -> list[dict]:
    good = []
    for e in entries:
        w, h = e["size"]
        right, bottom = w - e["x"], h - e["y"]
        if abs(e["ncc"]) < min_ncc or not (120 < bottom < 175):
            continue
        if abs(right - (0.093 * w + 163)) > 35:
            continue
        good.append(e)
    return good


def load_crops(entries: list[dict]) -> list[np.ndarray]:
    return [aligned_crop(luma(load_rgb(SAMPLES_DIR / e["file"])), e["x"], e["y"]) for e in entries]


def save_vis(name: str, m: np.ndarray, scale: float = 6.0) -> None:
    vis = np.clip(m * 255 * scale, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / name), cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    min_ncc = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    entries = select(json.loads((OUT_DIR / "locate.json").read_text()), min_ncc)
    print(f"{len(entries)} photos used")
    m, c, support = estimate(load_crops(entries))
    np.save(OUT_DIR / "m_all.npy", m)
    save_vis("m_all.png", m)
    by_listing: dict[str, list[dict]] = {}
    for e in entries:
        by_listing.setdefault(e["file"].split("/")[0], []).append(e)
    rows = []
    for lid, es in sorted(by_listing.items()):
        print(f"listing {lid}: {len(es)} photos")
        ml, _, _ = estimate(load_crops(es), c_fixed=c)
        np.save(OUT_DIR / f"m_{lid}.npy", ml)
        rows.append(ml)
    save_vis("m_listings.png", np.concatenate(rows, axis=0))
    (OUT_DIR / "stack_meta.json").write_text(json.dumps({"c": c, "n": len(entries)}))
