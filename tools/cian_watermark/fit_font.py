"""Step 6: fit a font rendering of the ID line to per-listing stacks.

For each candidate (font, size), render "ID <listing id>" supersampled 8x and
box-filtered down (anti-aliased, unhinted), find the best sub-pixel placement by
cross-correlation against the stack's ID-line band, fit the opacity by least
squares and report the correlation. The best (size, position) per style tells
how to render any ID.
"""
from __future__ import annotations

import json
import sys

from PIL import Image, ImageDraw, ImageFont

from common import *

FONTS = Path(__file__).resolve().parents[2] / "assets" / "marks" / "cian" / "fonts"
BAND = (96, 136, 40, 270)   # y0, y1, x0, x1 of the ID line in frame coordinates
SS = 8


def render_text(text: str, font_file: str, size: float, ss: int = SS, pad: int = 4) -> np.ndarray:
    """Coverage of text at `size` px, anti-aliased by ss-x supersampling. Origin = left of the ink box."""
    font = ImageFont.truetype(str(FONTS / font_file), size * ss)
    l, t, r, b = font.getbbox(text, anchor="ls")
    w, h = r - l + 2 * pad * ss, b - t + 2 * pad * ss
    w, h = (w + ss - 1) // ss * ss, (h + ss - 1) // ss * ss
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad * ss - l, pad * ss - t), text, fill=255, font=font, anchor="ls")
    a = np.asarray(img, np.float32) / 255.0
    return a.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))


def fit(stack: np.ndarray, text: str, font_file: str, sizes) -> list[tuple]:
    y0, y1, x0, x1 = BAND
    band = stack[y0:y1, x0:x1].astype(np.float32)
    out = []
    for s in sizes:
        t = render_text(text, font_file, s)
        if t.shape[0] >= band.shape[0] or t.shape[1] >= band.shape[1]:
            continue
        score = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED)
        yy, xx = np.unravel_index(int(score.argmax()), score.shape)
        fy, fx = subpixel_peak(score, int(yy), int(xx))
        patch = band[yy:yy + t.shape[0], xx:xx + t.shape[1]]
        opacity = float((patch * t).sum() / (t * t).sum())
        out.append((float(score[yy, xx]), s, x0 + fx, y0 + fy, opacity, t.shape[1]))
    return sorted(out, reverse=True)


if __name__ == "__main__":
    lids = sys.argv[1:] or ["325371418", "333382852", "318479656", "312255687",
                            "326545824", "333743056", "333159316", "333808851"]
    sizes = np.arange(20.0, 30.01, 0.25)
    for lid in lids:
        stack = np.load(OUT_DIR / f"mj_{lid}.npy")
        for font_file in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            best = fit(stack, f"ID {lid}", font_file, sizes)[0]
            ncc, s, x, y, op, w = best
            print(f"{lid} {font_file:20s} ncc={ncc:.3f} size={s:5.2f} x={x:7.2f} y={y:7.2f} "
                  f"center={x + w / 2:7.2f} opacity={op:.3f}")
