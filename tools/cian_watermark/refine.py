"""Step 4: register every photo to the current template by model fit, then restack.

For a photo crop O (taken at an integer-rounded position, so the photo itself is
never resampled), the template m is shifted by a sub-pixel offset s and the cost
    sum_p W(p) * |O(p) - B(p) - m_s(p) * (c - B(p))|
is minimised over s on a 0.1 px grid. The gain over "no mark" (m = 0) tells
whether the photo carries usable evidence. Photos with enough gain are restacked
at their refined positions, and the loop repeats.
"""
from __future__ import annotations

import json

from common import *
from stack import FRAME_W, FRAME_H, aligned_crop, estimate, smooth_background, flatness_weight, save_vis

C = 234.0
SEARCH = np.round(np.arange(-1.5, 1.51, 0.1), 2)


def shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    mat = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register(o: np.ndarray, m: np.ndarray, support: np.ndarray, coarse: int = 3) -> tuple[float, float, float]:
    """Best sub-pixel template shift for one crop and the relative cost gain over no mark."""
    b = smooth_background(o, support, 5.0)
    w = flatness_weight(o, b, support, 5.0) * support
    d = o - b
    a = C - b
    base = float((w * np.abs(d)).sum())
    best = (np.inf, 0.0, 0.0)
    # integer search first (the crop may be off by a pixel or two), then sub-pixel around it
    for iy in range(-coarse, coarse + 1):
        for ix in range(-coarse, coarse + 1):
            cost = float((w * np.abs(d - shift(m, ix, iy) * a)).sum())
            if cost < best[0]:
                best = (cost, ix, iy)
    _, bx, by = best
    for sy in SEARCH:
        for sx in SEARCH:
            cost = float((w * np.abs(d - shift(m, bx + sx, by + sy) * a)).sum())
            if cost < best[0]:
                best = (cost, bx + sx, by + sy)
    cost, dx, dy = best
    return dx, dy, 1.0 - cost / max(base, 1e-6)


if __name__ == "__main__":
    entries = json.loads((OUT_DIR / "locate2.json").read_text())
    first = {e["file"]: e for e in json.loads((OUT_DIR / "locate.json").read_text())}
    grays = {e["file"]: luma(load_rgb(SAMPLES_DIR / e["file"])) for e in entries}
    m = np.load(OUT_DIR / "m_subpixel.npy")
    pos = {e["file"]: (e["x"], e["y"]) for e in entries}
    for it in range(3):
        support = cv2.dilate((cv2.GaussianBlur(m, (0, 0), 1.0) > 0.015).astype(np.uint8),
                             np.ones((5, 5), np.uint8)).astype(np.float32)
        results = []
        for e in entries:
            x, y = pos[e["file"]]
            xi, yi = round(x), round(y)
            o = aligned_crop(grays[e["file"]], xi, yi)
            dx, dy, gain = register(o, m, support)
            # template shifted by +dx means the mark sits at xi + dx in the photo
            results.append((e["file"], xi + dx, yi + dy, gain))
        results.sort(key=lambda r: -r[3])
        used = [r for r in results if r[3] > 0.08 and r[0] != "321068459/00-1.jpg"]
        for f, x, y, g in results:
            pos[f] = (x, y)
        print(f"iter {it}: {len(used)} photos with gain > 0.08; gains "
              f"{[round(r[3], 2) for r in results[:5]]} ... {[round(r[3], 2) for r in results[-5:]]}")
        crops = [aligned_crop(grays[f], x, y) for f, x, y, g in used]
        m, c, _ = estimate(crops, c_fixed=C)
    np.save(OUT_DIR / "m_refined.npy", m)
    save_vis("m_refined.png", m)
    out = [{"file": f, "x": float(x), "y": float(y), "gain": float(g),
            "size": first[f]["size"]} for f, x, y, g in results]
    (OUT_DIR / "locate3.json").write_text(json.dumps(out, indent=1))
    fx = np.array([r["x"] % 1 for r in out if r["gain"] > 0.15])
    fy = np.array([r["y"] % 1 for r in out if r["gain"] > 0.15])
    print("fractional x (gain>0.15):", np.round(np.sort(fx), 1))
    print("fractional y (gain>0.15):", np.round(np.sort(fy), 1))
