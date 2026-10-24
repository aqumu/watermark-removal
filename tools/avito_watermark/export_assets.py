"""Step 8: write assets/marks/avito/{mark.json, mark.npz} for training and the service.

- nominal mark (the hint): the free-form corrected Alpha / A maps (out/ff_*.npy), in the
  frame of the last FH rows x FW columns of the image;
- the parametric layers (out/model.json) and a bank of the artwork's coverage at
  sub-pixel offsets, so training can re-render the mark with small deviations
  (opacity, colour, shadow, darkening, position) without the 8001-px artwork;
- codec and the size distribution of the stored photos (the JPEG grid phase);
- data_gen/backgrounds/avito_index.csv: the photos as training backgrounds (regions
  away from their own mark), split like the accuracy checks (holdout -> "val").
"""
from __future__ import annotations

import collections
import csv
import json

import numpy as np

from common import OUT, REPO, index, render_logo, split
from fit_model import FH, FW, affine

ASSET_DIR = REPO / "assets" / "marks" / "avito"
STEP = 0.125
BANK_DX = np.arange(-0.25, 0.25 + 1e-6, STEP)
BANK_DY = np.arange(-0.25, 1.375 + 1e-6, STEP)     # the shadow / darkening sit ~0.93 px below the logo


def main():
    params = {k: v for k, v in json.loads((OUT / "model.json").read_text()).items() if not k.startswith("rms")}
    bank = np.stack([np.stack([render_logo(params["width"], params["x"] + dx, params["y"] + dy, frame=(FW, FH))
                               for dx in BANK_DX]) for dy in BANK_DY]).astype(np.float32)
    al_model, a_model = affine([params[k] for k in
                                ["width", "x", "y", "sigma", "o", "oB", "cR", "cG", "cB",
                                 "so", "sdx", "sdy", "ss", "dk", "ds", "dG", "dB"]])
    sizes = collections.Counter((r["w"], r["h"]) for r in index(160, 80))
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ASSET_DIR / "mark.npz",
                        alpha=np.load(OUT / "ff_alpha.npy").astype(np.float32),
                        a=np.load(OUT / "ff_a.npy").astype(np.float32),
                        model_alpha=al_model.astype(np.float32), model_a=a_model.astype(np.float32),
                        bank=bank, bank_dx=BANK_DX.astype(np.float32), bank_dy=BANK_DY.astype(np.float32))
    meta = {
        "type": "fixed_layered",
        "description": "Small Avito mark: the Avito logo artwork 94 px wide, top-left at (W - 106.03, H - 37.78) "
                       "on every image size; white logo at 51% over a black drop shadow and a faint "
                       "background-independent darkening. Per pixel and channel out = b * (1 - Alpha) + A. "
                       "Measured in tools/avito_watermark (2026-10-24); served as JPEG q85 / q80 4:2:0.",
        "frame": [FW, FH],
        "window": [128, 64],
        "grid": 16,
        "codec": {"jpeg_quality": {"85": 0.6, "80": 0.4}, "subsampling": "4:2:0"},
        "params": params,
        "sizes": [[w, h, n] for (w, h), n in sizes.most_common()],
    }
    (ASSET_DIR / "mark.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    bg_csv = REPO / "data_gen" / "backgrounds" / "avito_index.csv"
    with open(bg_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["listing_id", "path", "split"])
        for r in index(320, 240):
            wr.writerow([r["sha256"][:16], r["path"], "val" if split(r["sha256"]) == "holdout" else "train"])
    err = np.abs(bank[int(np.argmin(np.abs(BANK_DY))), int(np.argmin(np.abs(BANK_DX)))]
                 - render_logo(params["width"], params["x"], params["y"], frame=(FW, FH))).max()
    print(f"{ASSET_DIR}: bank {bank.shape}, {len(sizes)} sizes, centre check {err:.1e}")


if __name__ == "__main__":
    main()
