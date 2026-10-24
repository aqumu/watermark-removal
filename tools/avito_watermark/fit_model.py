"""Step 5: layered parametric model of the mark, fitted to the per-pixel estimate.

Layers, bottom to top (all rendered from the artwork, data_gen/watermark.png):
  shadow   black, alpha s = so * blur_ss(artwork shifted by (sdx, sdy)) * (1 - coverage)  (a drop shadow,
           knocked out under the logo: inside the glyphs the mark is a plain white overlay)
  darken   background-independent darkening d = dk * rgb_d * blur_ds(artwork shifted by (sdx, sdy))
  logo     white(ish) colour c, alpha = o * blur_sigma(artwork)
  out = (b * (1 - s) - d) * (1 - alpha) + alpha * c
As a per-pixel affine map of the background: out - b = A - Alpha * b with
  Alpha = 1 - (1 - s)(1 - alpha),   A = alpha * c - d * (1 - alpha)
which is what estimate.py measures.

JPEG changes what the estimator sees near edges, so the model is fitted against
  real estimate - (synthetic estimate of the previous model - previous model)
i.e. the real estimate with the JPEG bias of the previous round removed (--bias).
Writes out/model.json, out/model_alpha.npy, out/model_a.npy (the Alpha / A maps).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from common import OUT, render_logo

FH, FW = 56, 124
NAMES = ["width", "x", "y", "sigma", "o", "oB", "cR", "cG", "cB",
         "so", "sdx", "sdy", "ss", "dk", "ds", "dG", "dB"]
P0 = [94.0, 17.99, 18.24, 0.3, 0.5, 0.99, 255, 255, 255, 0.15, 0.0, 1.0, 1.5, 1.0, 4.0, 1.1, 1.05]
LO = [93, 17, 17, 0.0, 0.3, 0.9, 200, 200, 200, 0.0, -3, -3, 0.2, 0.0, 0.5, 0.5, 0.5]
HI = [95, 19, 19, 1.5, 0.7, 1.1, 300, 300, 300, 0.6, 3, 3, 5.0, 10, 10, 2, 2]


def layers(p):
    q = dict(zip(NAMES, p))
    logo = render_logo(q["width"], q["x"], q["y"], frame=(FW, FH), blur=max(q["sigma"], 1e-3))
    alpha = logo[..., None] * np.array([q["o"], q["o"], q["o"] * q["oB"]], np.float32)
    sh = render_logo(q["width"], q["x"] + q["sdx"], q["y"] + q["sdy"], frame=(FW, FH), blur=q["ss"])
    cover = render_logo(q["width"], q["x"], q["y"], frame=(FW, FH))   # knock-out: no shadow under the logo
    s = (q["so"] * sh * (1 - cover))[..., None] * np.ones(3, np.float32)
    dk = render_logo(q["width"], q["x"] + q["sdx"], q["y"] + q["sdy"], frame=(FW, FH), blur=q["ds"])
    d = (q["dk"] * dk * (1 - cover))[..., None] * np.array([1.0, q["dG"], q["dB"]], np.float32)
    c = np.array([q["cR"], q["cG"], q["cB"]], np.float32)
    return alpha, s, d, c


def affine(p):
    alpha, s, d, c = layers(p)
    Alpha = 1 - (1 - s) * (1 - alpha)
    A = alpha * c - d * (1 - alpha)
    return Alpha, A


def levels_error(Al, A, al_ref, a_ref, sup, bgs=(40.0, 128.0, 220.0)):
    return np.concatenate([((A - Al * b) - (a_ref - al_ref * b))[sup].ravel() for b in bgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", default="est_real_fit_ph0_0.npz")
    ap.add_argument("--bias", default="", help="synthetic estimate of the previous model (npz)")
    ap.add_argument("--prev", default="", help="previous model maps prefix, e.g. model (model_alpha.npy/model_a.npy)")
    args = ap.parse_args()
    e = np.load(OUT / args.est)
    al, a = e["alpha"].copy(), e["a"].copy()
    if args.bias:
        s = np.load(OUT / args.bias)
        pa = np.load(OUT / f"{args.prev}_alpha.npy"); pA = np.load(OUT / f"{args.prev}_a.npy")
        al -= s["alpha"] - pa; a -= s["a"] - pA
    sup = cv2.dilate((np.load(OUT / "seed_alpha.npy") > 0.01).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    sup[:, -2:] = False; sup[-2:, :] = False
    p0 = P0
    prev_json = OUT / "model.json"
    if args.bias and prev_json.exists():
        prev = json.loads(prev_json.read_text()); p0 = [prev[k] for k in NAMES]
    r = least_squares(lambda p: levels_error(*affine(p), al, a, sup), p0, bounds=(LO, HI), x_scale="jac", diff_step=1e-3)
    Al, A = affine(r.x)
    res = dict(zip(NAMES, [round(float(v), 4) for v in r.x]))
    for b in (40.0, 128.0, 220.0):
        err = (A - Al * b) - (a - al * b)
        res[f"rms_bg{int(b)}"] = round(float(np.sqrt((err[sup] ** 2).mean())), 3)
    print(json.dumps(res))
    (OUT / "model.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / "model_alpha.npy", Al.astype(np.float32)); np.save(OUT / "model_a.npy", A.astype(np.float32))


if __name__ == "__main__":
    main()
