"""Step 3: re-locate every photo against the stacked logo estimate and test sub-pixel placement.

The template is the expected detail layer of the logo part (pin, roof, wordmark,
"ID"), which is the same on every photo. Matching uses the photo's detail layer.
Writes out/locate2.json in the same format as locate.json (x, y = wordmark top-left).
"""
from __future__ import annotations

import json

from common import *
from stack import FRAME_X0, FRAME_Y0, FRAME_W, FRAME_H, aligned_crop, estimate, select, save_vis

# Columns of the mark frame that hold the listing-specific digits (excluded from the logo template).
DIGITS_X0 = None  # filled from the stack: everything right of "ID " on the ID line


def logo_mask(m: np.ndarray) -> np.ndarray:
    """Pixels of the shared logo: strong m, excluding the digit run of the ID line."""
    support = (cv2.GaussianBlur(m, (0, 0), 1.0) > 0.02).astype(np.uint8)
    rows = np.where(support.any(1))[0]
    # the ID line is the lowest text row: find its top by scanning upwards for an empty gap
    prof = support.sum(1)
    bottom = rows.max()
    y = bottom
    while y > 0 and prof[y] > 0:
        y -= 1
    id_top = y + 1
    id_cols = np.where(support[id_top:bottom + 1].any(0))[0]
    # "ID" is the first two glyphs; digits start after the first wide gap in the ID line
    gaps = np.where(np.diff(id_cols) > 4)[0]
    digits_start = id_cols[gaps[0] + 1] - 2 if len(gaps) else id_cols.max()
    mask = cv2.dilate(support, np.ones((7, 7), np.uint8)).astype(np.float32)
    mask[id_top - 3:, digits_start:] = 0
    return mask, id_top, digits_start


def match(gray: np.ndarray, x_guess: float, y_guess: float, template: np.ndarray, mask: np.ndarray,
          radius: int = 24) -> tuple[float, float, float]:
    """Locate the template near a guess. Returns refined (x, y) of the wordmark top-left and NCC."""
    ox = int(round(x_guess)) + FRAME_X0 - radius
    oy = int(round(y_guess)) + FRAME_Y0 - radius
    h, w = gray.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + FRAME_W + 2 * radius), min(h, oy + FRAME_H + 2 * radius)
    region = highpass(gray[y0:y1, x0:x1], 21)
    if region.shape[0] < FRAME_H or region.shape[1] < FRAME_W:
        return x_guess, y_guess, 0.0
    t = template * mask
    score = cv2.matchTemplate(region, t.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    absscore = np.abs(score)
    yy, xx = np.unravel_index(int(absscore.argmax()), absscore.shape)
    fy, fx = subpixel_peak(absscore, int(yy), int(xx))
    return x0 + fx - FRAME_X0, y0 + fy - FRAME_Y0, float(score[yy, xx])


def expected_detail(m: np.ndarray, c: float, bg: float = 110.0) -> np.ndarray:
    """Detail layer the logo would produce on a flat background of level bg."""
    o = bg + m * (c - bg)
    return highpass(o.astype(np.float32), 21)


if __name__ == "__main__":
    m = np.load(OUT_DIR / "m_all.npy")
    c = json.loads((OUT_DIR / "stack_meta.json").read_text())["c"]
    mask, id_top, digits_start = logo_mask(m)
    print(f"ID line top={id_top} digits start x={digits_start} (frame coords)")
    template = expected_detail(m, c)
    first = {e["file"]: e for e in json.loads((OUT_DIR / "locate.json").read_text())}
    out = []
    for path in list_samples():
        key = f"{listing_id(path)}/{path.name}"
        e = first[key]
        g = luma(load_rgb(path))
        h, w = g.shape
        # prior position from the anchor model when the first pass was unreliable
        xg, yg = e["x"], e["y"]
        if abs(e["ncc"]) < 0.35:
            xg, yg = w - (0.093 * w + 163), h - 148
        x, y, s = match(g, xg, yg, template, mask)
        out.append({"file": key, "size": [w, h], "x": float(x), "y": float(y), "ncc": float(s)})
        print(f"{key:22s} {w}x{h}  x={x:8.2f} y={y:8.2f} right={w - x:7.2f} bottom={h - y:7.2f} ncc={s:+.3f}")
    (OUT_DIR / "locate2.json").write_text(json.dumps(out, indent=1))
    np.save(OUT_DIR / "logo_mask.npy", mask)
    (OUT_DIR / "frame_meta.json").write_text(json.dumps({"id_top": int(id_top), "digits_start": int(digits_start)}))
