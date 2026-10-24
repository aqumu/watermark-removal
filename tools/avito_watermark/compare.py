"""Real vs synthetic estimates per JPEG grid phase, against the estimator's own noise.

For each phase group: rms over the mark's support of (real - synth) in levels on
backgrounds 40/128/220, and the noise of that comparison (split-half differences,
scaled to the full set: rms(half0 - half1) / 2 for each of real and synth).
"""
import sys

import cv2
import numpy as np

from common import OUT

sup = cv2.dilate((np.load(OUT / "seed_alpha.npy") > 0.01).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
sup[:, -2:] = False; sup[-2:, :] = False


def lv(al, a, b):
    return a - al * b


for tag in sys.argv[1:]:
    R = np.load(OUT / f"est_real_fit{tag}.npz"); S = np.load(OUT / f"est_synth_fit{tag}.npz")
    cells = []
    for b in (40.0, 128.0, 220.0):
        diff = lv(R["alpha"], R["a"], b) - lv(S["alpha"], S["a"], b)
        nr = (lv(R["alpha0"], R["a0"], b) - lv(R["alpha1"], R["a1"], b)) / 2
        ns = (lv(S["alpha0"], S["a0"], b) - lv(S["alpha1"], S["a1"], b)) / 2
        rms = np.sqrt((diff[sup] ** 2).mean()); noise = np.sqrt((nr[sup] ** 2).mean() + (ns[sup] ** 2).mean())
        cells.append(f"bg{int(b)}: {rms:.2f} (noise {noise:.2f})")
    print(f"{tag or 'all':8s} n={int(R['n']):5d}  " + "  ".join(cells))
