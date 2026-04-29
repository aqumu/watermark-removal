"""
OOM benchmark -- finds the maximum batch size for a full training step
(forward + backward + optimizer update + AMP) without running out of VRAM.

Run from the training/ directory:
    python benchmark_batch.py
"""

from __future__ import annotations

import gc
import sys
import pathlib

# Make repo root importable
_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def probe_step(model, optimizer, scaler, in_ch, out_ch, h, w, batch, device):
    """Run one full training step. Returns True on success, False on OOM."""
    x = y = loss = out = None
    try:
        x = torch.randn(batch, in_ch, h, w, device=device)
        y = torch.randn(batch, out_ch, h, w, device=device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = nn.functional.mse_loss(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()

        return True

    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del x, y, out, loss
        free_memory()


def find_max_batch(label, model, optimizer, scaler, in_ch, out_ch, h, w, device):
    """Try powers of 2 until OOM, then report."""
    print(f"  {label}")
    last_good = 0
    last_peak = 0.0

    batch = 1
    while batch <= 512:
        ok = probe_step(model, optimizer, scaler, in_ch, out_ch, h, w, batch, device)
        if ok:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"    batch={batch:3d}  peak={peak:.0f} MB  OK")
            last_good = batch
            last_peak = peak
            batch *= 2
        else:
            print(f"    batch={batch:3d}  OOM")
            break

    print(f"  -> max batch = {last_good}  (peak VRAM {last_peak:.0f} MB)\n")
    return last_good, last_peak


def make_removal_model(use_checkpoint):
    from training.src.tasks.removal.model import MaskedUNet
    return MaskedUNet(base_channels=32, depth=4, use_checkpoint=use_checkpoint)


def make_removal_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)


def make_seg_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def make_seg_optimizer(model):
    encoder_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in encoder_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    return torch.optim.Adam([
        {"params": encoder_params, "lr": 1e-5},
        {"params": dec_params,     "lr": 1e-4},
    ], weight_decay=1e-4)


def run_config(label, model_fn, opt_fn, in_ch, out_ch, h, w, device):
    free_memory()
    model = opt = scaler = None
    try:
        model = model_fn().to(device)
        model.train()
        opt = opt_fn(model)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        return find_max_batch(label, model, opt, scaler, in_ch, out_ch, h, w, device)
    finally:
        del model, opt, scaler
        free_memory()


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"Device : {props.name}")
    print(f"VRAM   : {props.total_memory / 1024**2:.0f} MB\n")

    results = []

    for ckpt in (True, False):
        label = f"Removal 256x128 ckpt={ckpt}"
        mb, peak = run_config(label, lambda c=ckpt: make_removal_model(c),
                              make_removal_optimizer, 5, 3, 128, 256, device)
        results.append((label, mb, peak))

    for ckpt in (True, False):
        label = f"Removal 512x256 ckpt={ckpt}"
        mb, peak = run_config(label, lambda c=ckpt: make_removal_model(c),
                              make_removal_optimizer, 5, 3, 256, 512, device)
        results.append((label, mb, peak))

    label = "Seg 256x256"
    mb, peak = run_config(label, make_seg_model, make_seg_optimizer,
                          3, 1, 256, 256, device)
    results.append((label, mb, peak))

    label = "Seg 512x512"
    mb, peak = run_config(label, make_seg_model, make_seg_optimizer,
                          3, 1, 512, 512, device)
    results.append((label, mb, peak))

    print("=" * 55)
    print(f"{'Configuration':<38} {'Max batch':>9}  {'Peak MB':>6}")
    print("-" * 55)
    for lbl, mb, peak in results:
        print(f"{lbl:<38} {mb:>9}  {peak:>6.0f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
