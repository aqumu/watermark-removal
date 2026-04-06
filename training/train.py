"""
train.py  –  main entry point
------------------------------
Usage:
  python train.py                          # use configs/train.yaml
  python train.py --config path/to/cfg.yaml
  python train.py --resume checkpoints/epoch_0010.pth
"""

import argparse
import random
import threading

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset   import make_splits
from src.losses    import CombinedLoss
from src.model     import build_model, count_params
from src.trainer   import Trainer
from src.live_plot import LivePlotter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    loss_cfg = cfg.get("loss", {})
    
    tr_ds, va_ds = make_splits(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        train_frac=ds_cfg["train_split"],
        seed=cfg["seed"],
        loss_mask_blur_pct=loss_cfg.get("loss_mask_blur_pct", 0.0),
        max_samples=ds_cfg.get("max_samples"),
    )

    nw = ds_cfg["num_workers"]
    pf = ds_cfg.get("prefetch_factor", 2)
    use_cuda = cfg.get("device", "cpu") == "cuda" and torch.cuda.is_available()
    train_loader = DataLoader(
        tr_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=pf if nw > 0 else None,
    )
    val_loader = DataLoader(
        va_ds, batch_size=1,
        shuffle=False, num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=pf if nw > 0 else None,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from (full state)")
    parser.add_argument("--load-weights", default=None,
                        help="path to checkpoint to load weights from (fresh start)")
    parser.add_argument("--reset-upsamplers", action="store_true",
                        help="re-apply ICNR to decoder upsampling convs after loading "
                             "checkpoint; fixes checkerboard artifacts without full "
                             "retraining — use together with --resume or --load-weights")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Manually override the starting epoch (useful for resuming weights with new LR).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])

    if cfg.get("device", "cpu") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {cfg['device']}")

    n_threads = cfg.get("torch_threads", 0)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        print(f"PyTorch intra-op threads: {torch.get_num_threads()}")

    model = build_model(cfg)
    n_params = count_params(model)
    print(f"Model: {n_params:,} trainable parameters")

    train_loader, val_loader = make_loaders(cfg)

    from pathlib import Path

    # Pick one random sample to use as the live visualisation probe.
    # Same sample is kept for the whole session so progress is comparable.
    _ds_root    = Path(cfg["dataset"]["root"])
    _candidates = sorted([
        d for d in _ds_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "clean.png").exists()
        and (d / "mask.png").exists()
    ])
    sample_dir = random.choice(_candidates) if _candidates else None
    if sample_dir:
        print(f"Visualisation sample: {sample_dir.name}")

    plotter = LivePlotter(
        log_dir=Path(cfg["logging"]["dir"]),
        total_epochs=cfg["training"]["epochs"],
        loss_cfg=cfg.get("loss", {}),
        log_every=cfg["logging"].get("log_every", 10),
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
        load_weights=args.load_weights,
        start_epoch=args.start_epoch,
        plotter=plotter,
    )

    trainer._sample_dir = sample_dir

    if args.reset_upsamplers:
        if not (args.resume or args.load_weights):
            print("[WARNING] --reset-upsamplers has no effect without "
                  "--resume or --load-weights; ignoring")
        elif hasattr(model, "reset_decoder_upsamplers"):
            model.reset_decoder_upsamplers()
        else:
            print("[WARNING] --reset-upsamplers is only supported for MaskedUNet; ignoring")

    def _train_worker(trainer, plotter):
        try:
            trainer.train()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            plotter._queue.put(("stop",))

    thread = threading.Thread(target=_train_worker, args=(trainer, plotter), daemon=True)
    thread.start()
    plotter.run_event_loop()   # blocks main thread; owns the GUI event loop
    thread.join()
    plotter.save()


if __name__ == "__main__":
    main()
