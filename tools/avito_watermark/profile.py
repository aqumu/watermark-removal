"""Edge profiles of an estimate by signed distance to the artwork's glyph edge (px; negative = inside)."""
import sys

import cv2
import numpy as np

from common import OUT, render_logo

FH, FW = 56, 124


def signed_distance():
    hi = render_logo(94.0925 * 8, 17.9432 * 8, 18.2831 * 8, frame=(FW * 8, FH * 8))
    m = (hi > 0.5).astype(np.uint8)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5); din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    sd = (dout - din) / 8.0
    return cv2.resize(sd, (FW, FH), interpolation=cv2.INTER_AREA)


BINS = [(-6, -3), (-3, -2), (-2, -1.25), (-1.25, -0.75), (-0.75, -0.25), (-0.25, 0.25), (0.25, 0.75), (0.75, 1.25),
        (1.25, 2), (2, 3), (3, 4), (4, 5), (5, 6.5), (6.5, 8), (8, 10), (10, 13)]


def profile(est, sd, sup):
    out = []
    for lo, hi in BINS:
        m = (sd >= lo) & (sd < hi) & sup
        out.append((lo, hi, int(m.sum()), est["alpha"][m].mean(0), est["a"][m].mean(0)))
    return out


if __name__ == "__main__":
    sd = signed_distance()
    np.save(OUT / "signed_distance.npy", sd)
    sup = np.ones_like(sd, bool); sup[:, -2:] = False; sup[-2:, :] = False
    ests = {name: np.load(OUT / name) for name in sys.argv[1:]}
    profs = {k: profile(v, sd, sup) for k, v in ests.items()}
    hdr = "dist        n  " + "  ".join(f"{k[4:24]:>34s}" for k in ests)
    print(hdr)
    for i, (lo, hi, n, _, _) in enumerate(next(iter(profs.values()))):
        cells = []
        for k in ests:
            _, _, _, al, a = profs[k][i]
            cells.append(f"al {al.mean():.4f} a {a.mean():6.2f} a/al {a.mean()/max(al.mean(),1e-4):6.0f}")
        print(f"{lo:5.2f}..{hi:5.2f} {n:4d}  " + "  ".join(cells))
