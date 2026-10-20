"""
bench_overlay_cpu.py  CPU latency/throughput of overlay-removal model variants
------------------------------------------------------------------------------
Times the model forward pass on the profile's inference window (the whole
per-image model cost; localisation and I/O are measured separately).

Usage:
  python training/bench_overlay_cpu.py [--threads 4] [--batch 1 8] [--widths 24,48,64,96 16,32,48,64]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tasks.overlay.model import OverlayUNet, count_params
from wm_shared.marks import load_mark


def macs(model: torch.nn.Module, shape) -> float:
    """Multiply-accumulates of conv layers for one input of `shape` (C, H, W)."""
    total = 0

    def hook(mod, inp, out):
        nonlocal total
        k = mod.kernel_size[0] * mod.kernel_size[1]
        cin = mod.in_channels // mod.groups
        total += out.numel() // out.shape[0] * cin * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    for h in hs:
        h.remove()
    return total


def bench(model, batch: int, shape, reps: int = 20) -> float:
    x = torch.randn(batch, *shape)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--widths", nargs="+", default=["24,48,64,96", "32,64,96,128", "16,32,48,64"])
    args = ap.parse_args()
    ww, wh = load_mark("cian").window
    shape = (7, wh, ww)
    for wstr in args.widths:
        widths = tuple(int(v) for v in wstr.split(","))
        model = OverlayUNet(widths=widths).eval()
        print(f"widths {widths}: {count_params(model):,} params, {macs(model, shape) / 1e9:.2f} GMAC per window")
        for th in args.threads:
            torch.set_num_threads(th)
            for b in args.batch:
                dt = bench(model, b, shape)
                print(f"  threads={th} batch={b}: {dt * 1000:7.1f} ms/batch  {dt / b * 1000:6.1f} ms/window  "
                      f"{b / dt:6.1f} windows/s")


if __name__ == "__main__":
    main()
