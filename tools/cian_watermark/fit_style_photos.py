"""Fit a text style jointly over many single photos (each with its own listing ID).

Used for variants that have no large per-listing stack. For a set of photos,
background / weights / contrast in the text band are computed once; then font,
size, centre, top and opacity are tuned by coordinate ascent on the summed fit
gain  1 - sum w|d - op*t*a| / sum w|d|  over the photos.

Usage: python fit_style_photos.py <font file> [--which mismatch|regular|bold]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import *  # noqa: E402
from stack import aligned_crop, smooth_background, flatness_weight, estimate, save_vis  # noqa: E402
from wm_shared.marks import load_mark  # noqa: E402
from wm_shared.marks.profile import TextStyle, _paste_add, text_coverage, text_ink_width  # noqa: E402

C = 234.0
BAND = (92, 132, 40, 280)


def prepare(photos: list[dict], logo: np.ndarray):
    y0, y1, x0, x1 = BAND
    data = []
    for p in photos:
        o = aligned_crop(luma(load_rgb(Path(p["path"]))), p["x"], p["y"])
        support = cv2.dilate((logo > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8))
        support[y0:y1, x0:x1] = 1
        support = support.astype(np.float32)
        b = smooth_background(o, support, 5.0)
        w = flatness_weight(o, b, support, 5.0)
        d = (o - b - logo * (C - b))[y0:y1, x0:x1]   # remove the logo's own contribution
        data.append((p["listing_id"], d, (C - b)[y0:y1, x0:x1], w[y0:y1, x0:x1], o))
    return data


def band_text(text: str, st: TextStyle, shape) -> np.ndarray:
    canvas = np.zeros(shape, np.float32)
    cov, ix, iy = text_coverage(text, st, st.center_x - text_ink_width(text, st) / 2, st.top)
    _paste_add(canvas, cov, ix, iy)
    y0, y1, x0, x1 = BAND
    return canvas[y0:y1, x0:x1]


def total_gain(data, st: TextStyle, frame_shape) -> float:
    g = 0.0
    for lid, d, a, w, _ in data:
        t = band_text("ID " + lid, st, frame_shape) * st.opacity
        base = (w * np.abs(d)).sum()
        g += 1 - (w * np.abs(d - t * a)).sum() / max(base, 1e-6)
    return g / len(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("font")
    ap.add_argument("--which", default="mismatch")
    ap.add_argument("--size", type=float, default=26.0)
    ap.add_argument("--top", type=float, default=99.0)
    ap.add_argument("--max", type=int, default=120)
    args = ap.parse_args()
    prof = load_mark("cian")
    stats = json.loads((OUT_DIR / "style_stats.json").read_text())
    pos = {(p["listing_id"], round(p["gain"], 6)): p for p in json.loads((OUT_DIR / "jpeg_positions.json").read_text())}
    sel = []
    for r in stats["photos"]:
        cls = "mismatch" if r["gains"][r["best"]] < 0.1 else r["best"]
        if cls in args.which.split(","):
            sel.append(pos[(r["listing_id"], round(r["gain_logo"], 6))])
    sel = sorted(sel, key=lambda p: -p["gain"])[:args.max]
    print(f"{len(sel)} photos")
    data = prepare(sel, prof.logo)
    fw, fh = prof.frame_size
    st = TextStyle(font=str(REPO / "assets/marks/cian/fonts" / args.font), size=args.size, opacity=0.12,
                   center_x=159.0, top=args.top)
    best = total_gain(data, st, (fh, fw))
    print(f"start gain {best:.4f}")
    for step in (2.0, 1.0, 0.5, 0.25, 0.1):
        improved = True
        while improved:
            improved = False
            for attr, d in (("center_x", step), ("top", step), ("size", step / 2), ("opacity", step / 50)):
                for sign in (-1, 1):
                    trial = TextStyle(**{**st.__dict__, attr: getattr(st, attr) + sign * d})
                    s = total_gain(data, trial, (fh, fw))
                    if s > best + 1e-5:
                        st, best, improved = trial, s, True
        print(f"step {step}: gain {best:.4f} size={st.size:.3f} cx={st.center_x:.2f} top={st.top:.2f} op={st.opacity:.4f}")
    # logo check: does this group share the main logo?
    m, _, _ = estimate([o for *_, o in data], c_fixed=C, iters=3)
    mask = prof.logo > 0.02
    print("logo corr with main logo:", np.corrcoef(m[:95][mask[:95]], prof.logo[:95][mask[:95]])[0, 1].round(4))
    np.save(OUT_DIR / f"m_group_{args.which}.npy", m)
    save_vis(f"m_group_{args.which}.png", m)
