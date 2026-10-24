"""Step 6: free-form per-pixel correction on top of the parametric model.

The parametric model leaves ~2 levels rms of systematic difference between the real
and the synthetic estimates. Part of it changes with the JPEG grid phase (encoder
differences), part is common to all phases (the real mark's own raster). The
correction is real - synth averaged with equal weight over TRAIN_PHASES, so
phase-specific JPEG effects average out instead of being baked into the mark;
TEST_PHASES are left out and show whether the correction is a property of the mark.
Writes out/ff_alpha.npy / ff_a.npy (Alpha / A maps of the corrected mark).
"""
import subprocess
import sys

import cv2
import numpy as np

from common import OUT

TRAIN_PHASES = ["0,0", "5,0", "0,12", "0,15", "15,0", "0,14", "0,10", "14,0", "14,8", "0,2", "6,0", "1,0"]
TEST_PHASES = ["0,11", "4,0"]
PY = sys.executable


def est(mode, ph, model="model"):
    tag = f"_ph{ph.replace(',', '_')}" + ("" if mode == "real" else f"_{model}")
    if mode == "real" and (OUT / f"est_real_fit{tag}.npz").exists():
        return np.load(OUT / f"est_real_fit{tag}.npz")
    subprocess.run([PY, "estimate.py", "--mode", mode, "--phase", ph, "--tag", tag, "--model", model],
                   check=True, stdout=subprocess.DEVNULL)
    return np.load(OUT / f"est_{mode}_fit{tag}.npz")


def score(R, S, sup):
    return [float(np.sqrt(((((R["a"] - R["alpha"] * b) - (S["a"] - S["alpha"] * b)))[sup] ** 2).mean())) for b in (40.0, 128.0, 220.0)]


def main():
    sup = cv2.dilate((np.load(OUT / "seed_alpha.npy") > 0.01).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    sup[:, -2:] = False; sup[-2:, :] = False
    taper = cv2.GaussianBlur(sup.astype(np.float32), (0, 0), 1.0)[..., None]
    al = np.load(OUT / "model_alpha.npy"); a = np.load(OUT / "model_a.npy")
    np.save(OUT / "ff_alpha.npy", al); np.save(OUT / "ff_a.npy", a)
    real = {ph: est("real", ph) for ph in TRAIN_PHASES + TEST_PHASES}
    for it in range(4):
        syn = {ph: est("synth", ph, "ff") for ph in TRAIN_PHASES + TEST_PHASES}
        tr = np.mean([score(real[p], syn[p], sup) for p in TRAIN_PHASES], 0)
        te = np.mean([score(real[p], syn[p], sup) for p in TEST_PHASES], 0)
        print(f"iter {it}: train phases rms bg40/128/220 {tr.round(2)}   held-out phases {te.round(2)}", flush=True)
        if it == 3:
            break
        d_al = np.mean([real[p]["alpha"] - syn[p]["alpha"] for p in TRAIN_PHASES], 0)
        d_a = np.mean([real[p]["a"] - syn[p]["a"] for p in TRAIN_PHASES], 0)
        al = al + d_al * taper; a = a + d_a * taper
        np.save(OUT / "ff_alpha.npy", al.astype(np.float32)); np.save(OUT / "ff_a.npy", a.astype(np.float32))


if __name__ == "__main__":
    main()
