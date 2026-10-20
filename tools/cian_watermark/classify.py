"""Step 8: per photo, which text style is stamped, and does its ID match the listing?

For every registered photo, fit the text band with "ID <listing id>" rendered in
each style (and with no text at all). The gain of the best style over "no text"
classifies the style; photos whose logo fits well but whose text does not are
counted as ID mismatches (re-used photos carry another listing's ID).
Writes out/style_stats.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import *  # noqa: E402
from stack import FRAME_X0, FRAME_Y0, aligned_crop, smooth_background, flatness_weight  # noqa: E402
from wm_shared.marks import load_mark  # noqa: E402

C = 234.0
P = None


def _init():
    global P
    P = load_mark("cian")


def classify(pos: dict) -> dict | None:
    try:
        g = luma(load_rgb(Path(pos["path"])))
    except Exception:
        return None
    o = aligned_crop(g, pos["x"], pos["y"])   # frame-aligned at the registered position
    fw, fh = P.frame_size
    ms = {s: P.render(pos["listing_id"], style=s) for s in P.styles}
    text_union = np.zeros((fh, fw), np.float32)
    for s, m in ms.items():
        text_union = np.maximum(text_union, m - P.logo)
    support = cv2.dilate(((P.logo > 0.01) | (text_union > 0.01)).astype(np.uint8), np.ones((5, 5), np.uint8))
    band = np.zeros_like(support); band[100:128, 40:280] = 1
    support = np.maximum(support, band).astype(np.float32)
    b = smooth_background(o, support, 5.0)
    w = flatness_weight(o, b, support, 5.0) * band
    d, a = o - b, C - b
    base = float((w * np.abs(d)).sum())

    def gain(m):
        return 1 - float((w * np.abs(d - (m - P.logo) * a)).sum()) / max(base, 1e-6)

    gains = {s: gain(m) for s, m in ms.items()}
    best = max(gains, key=gains.get)
    # the same style with random IDs of the same length: does the real ID fit clearly better?
    rng = np.random.default_rng(int(pos["listing_id"]) % 2**32)
    n = len(pos["listing_id"])
    rand = [gain(P.render(str(rng.integers(10 ** (n - 1), 10 ** n)), style=best)) for _ in range(6)]
    return {"listing_id": pos["listing_id"], "gain_logo": pos["gain"], "gains": gains, "best": best,
            "rand_max": max(rand), "rand_mean": float(np.mean(rand))}


if __name__ == "__main__":
    positions = [p for p in json.loads((OUT_DIR / "jpeg_positions.json").read_text()) if p["gain"] >= 0.3]
    with Pool(12, initializer=_init) as pool:
        res = [r for r in pool.imap_unordered(classify, positions, chunksize=8) if r is not None]
    counts = Counter()
    for r in res:
        margin = r["gains"][r["best"]] - r["rand_max"]
        if margin > 0.02:
            counts[r["best"]] += 1
        elif r["gains"][r["best"]] - r["rand_mean"] > 0.01:
            counts["uncertain"] += 1
        else:
            counts["mismatch"] += 1
    n_ok = counts["bold"] + counts["regular"]
    probs = {"bold": counts["bold"] / max(n_ok, 1), "regular": counts["regular"] / max(n_ok, 1)}
    print(f"{len(res)} photos with a clear logo: {dict(counts)}")
    print(f"style probabilities: {probs}; mismatch rate {counts['mismatch'] / len(res):.3f}")
    (OUT_DIR / "style_stats.json").write_text(json.dumps({"counts": dict(counts), "probs": probs,
                                                          "photos": res}, indent=1))
