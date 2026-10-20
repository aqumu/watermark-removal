"""Step 9: re-fit the text-line geometry relative to the logo, per style.

The first text fits used photos registered with a template whose ID area held
averaged (mostly bold) digits, which pulled regular-style photos off by ~1.5 px.
Here every photo is registered on the logo alone (identical in all variants),
then each style's size / centre / top is fitted on the photos of that style,
and the share of each style (incl. marks without any text line) is counted.

Writes out/text_refit.json: {"styles": {...}, "probs": {...}, "positions": [...]}.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import *  # noqa: E402
from stack import FRAME_X0, FRAME_Y0, aligned_crop, smooth_background, flatness_weight  # noqa: E402
from wm_shared.marks import load_mark, shift_image  # noqa: E402
from wm_shared.marks.profile import TextStyle, _paste_add, text_coverage, text_ink_width  # noqa: E402

P = None
BAND = (96, 130, 40, 280)


def _init():
    global P
    P = load_mark("cian")


def register_logo(path: str, x: float, y: float) -> dict | None:
    """Sub-pixel model-fit registration of the logo alone around a prior wordmark position."""
    g = luma(load_rgb(Path(path)))
    o = aligned_crop(g, round(x), round(y))
    logo = P.logo
    sup = cv2.dilate((logo > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    band = np.zeros_like(sup); band[BAND[0]:BAND[1], BAND[2]:BAND[3]] = 1
    b = smooth_background(o, np.maximum(sup, band), 5.0)
    w = flatness_weight(o, b, np.maximum(sup, band), 5.0) * sup
    d, a = o - b, P.overlay - b
    base = float((w * np.abs(d)).sum())
    best = (np.inf, 0.0, 0.0)
    for step, rng in ((1.0, 3), (0.25, 4), (0.05, 5)):
        _, bx, by = best
        for iy in range(-rng, rng + 1):
            for ix in range(-rng, rng + 1):
                dx, dy = bx + ix * step, by + iy * step
                cost = float((w * np.abs(d - shift_image(logo, dx, dy) * a)).sum())
                if cost < best[0]:
                    best = (cost, dx, dy)
    cost, dx, dy = best
    gain = 1 - cost / max(base, 1e-6)
    # keep the frame crop re-aligned on the logo for the text fits
    o2 = aligned_crop(g, round(x) + dx, round(y) + dy)
    b2 = smooth_background(o2, np.maximum(sup, band), 5.0)
    w2 = flatness_weight(o2, b2, np.maximum(sup, band), 5.0)
    y0, y1, x0, x1 = BAND
    return {"path": path, "listing_id": pos_id(path), "x": round(x) + dx, "y": round(y) + dy, "gain": gain,
            "d": (o2 - b2)[y0:y1, x0:x1].astype(np.float32), "a": (P.overlay - b2)[y0:y1, x0:x1].astype(np.float32),
            "w": w2[y0:y1, x0:x1].astype(np.float32)}


_IDS: dict = {}


def pos_id(path):
    return _IDS.get(path)


def text_band(text: str, st: TextStyle, frame_shape) -> np.ndarray:
    canvas = np.zeros(frame_shape, np.float32)
    cov, ix, iy = text_coverage(text, st, st.center_x - text_ink_width(text, st) / 2, st.top)
    _paste_add(canvas, cov * st.opacity, ix, iy)
    y0, y1, x0, x1 = BAND
    return canvas[y0:y1, x0:x1]


def gain(item, st: TextStyle | None, shape) -> float:
    d, a, w = item["d"], item["a"], item["w"]
    base = float((w * np.abs(d)).sum())
    if st is None:
        return 0.0
    t = text_band("ID " + item["listing_id"], st, shape)
    return 1 - float((w * np.abs(d - t * a)).sum()) / max(base, 1e-6)


def fit_geometry(items, st: TextStyle, shape) -> tuple[TextStyle, float]:
    def total(s):
        return float(np.mean([gain(i, s, shape) for i in items]))
    best = total(st)
    for step in (0.5, 0.2, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for attr, dlt in (("center_x", step), ("top", step), ("size", step / 2)):
                for sign in (-1, 1):
                    trial = TextStyle(**{**st.__dict__, attr: getattr(st, attr) + sign * dlt})
                    sc = total(trial)
                    if sc > best + 1e-5:
                        st, best, improved = trial, sc, True
    return st, best


if __name__ == "__main__":
    positions = [q for q in json.loads((OUT_DIR / "jpeg_positions.json").read_text()) if q["gain"] >= 0.25]
    positions = sorted(positions, key=lambda q: -q["gain"])[:900]
    for q in positions:
        _IDS[q["path"]] = q["listing_id"]
    with Pool(4, initializer=_init) as pool:
        regs = [r for r in pool.starmap(register_logo, [(q["path"], q["x"], q["y"]) for q in positions], chunksize=8)
                if r is not None]
    for r in regs:
        r["listing_id"] = _IDS[r["path"]]
    _init()
    shape = (P.frame_size[1], P.frame_size[0])
    regs = [r for r in regs if r["gain"] >= 0.25]
    print(f"{len(regs)} photos registered on the logo")
    # classify with the current styles, then refit each style on its photos, twice
    styles = dict(P.styles)
    for rnd in range(2):
        groups = {k: [] for k in styles}
        none = []
        for r in regs:
            g = {k: gain(r, s, shape) for k, s in styles.items()}
            k = max(g, key=g.get)
            # text present only if it clearly beats random IDs of the same style
            rng = np.random.default_rng(int(r["listing_id"]) % 2 ** 32)
            rand = max(gain({**r, "listing_id": str(rng.integers(10 ** 8, 10 ** 9))}, styles[k], shape) for _ in range(4))
            if g[k] - rand > 0.02:
                groups[k].append(r)
            elif r["gain"] >= 0.4 and g[k] < 0.01 and max(rand, 0) < 0.01:
                none.append(r)
        print(f"round {rnd}: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()) + f", no text={len(none)}, "
              f"undecided={len(regs) - sum(len(v) for v in groups.values()) - len(none)}")
        for k in styles:
            if len(groups[k]) >= 20:
                styles[k], sc = fit_geometry(groups[k][:250], styles[k], shape)
                print(f"  {k}: size {styles[k].size:.3f} centre {styles[k].center_x:.2f} top {styles[k].top:.2f} gain {sc:.4f}")
    n = sum(len(v) for v in groups.values()) + len(none)
    probs = {k: len(v) / n for k, v in groups.items()}
    probs["none"] = len(none) / n
    out = {"styles": {k: {"size": round(s.size, 3), "center_x": round(s.center_x, 2), "top": round(s.top, 2)}
                      for k, s in styles.items()}, "probs": probs}
    (OUT_DIR / "text_refit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
