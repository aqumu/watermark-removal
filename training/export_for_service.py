"""
export_for_service.py  package a trained overlay model for watermark-service
----------------------------------------------------------------------------
Writes, into --out (the service's models/ directory or a staging folder):

  model_<mark>_<name>.onnx the model (input: 7xHxW from build_input, output: residual)
  <mark>/mark.json         mark geometry (Cian: text styles, overlay level, anchor), window
  <mark>/mark.npz          logo raster (Cian) / nominal Alpha and A maps (Avito)
  <mark>/fonts/*           fonts referenced by mark.json (+ licence), text marks only

The service's mode "cian" picks the newest model_cian_*.onnx and models/cian/,
mode "avito" the newest model_avito_*.onnx and models/avito/.

Usage:
  python training/export_for_service.py --checkpoint student.pth --config cfg.yaml --name c_v2 --out ../watermark-service/models
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from export_overlay_onnx import check_equal, export
from src.tasks.overlay.model import build_model
from train_overlay import load_cfg
from wm_shared.marks.profile import ASSETS_DIR, load_mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="suffix of model_<mark>_<name>.onnx")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    mark = cfg["mark"]
    profile = load_mark(mark)
    if float(cfg["data"].get("hint_scale", 8.0)) != 8.0:
        print(f"note: hint_scale differs from the service default (8.0); set {mark}.hint_scale accordingly")
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"model_{mark}_{args.name}.onnx"
    ww, wh = profile.window
    export(model, onnx_path, (7, wh, ww))
    print(f"{onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), max |onnx - torch| = "
          f"{check_equal(model, onnx_path, (7, wh, ww)):.1e}")

    src = ASSETS_DIR / mark
    dst = out / mark
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("mark.json", "mark.npz"):
        shutil.copy2(src / name, dst / name)
    fonts = ""
    if (src / "fonts").is_dir():   # text marks (Cian); the fixed Avito mark has none
        (dst / "fonts").mkdir(exist_ok=True)
        for f in (src / "fonts").iterdir():
            shutil.copy2(f, dst / "fonts" / f.name)
        fonts = f", fonts/ ({len(list((dst / 'fonts').iterdir()))} files)"
    print(f"{dst}: mark.json, mark.npz{fonts}")


if __name__ == "__main__":
    main()
