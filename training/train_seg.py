"""
train_seg.py  –  train the watermark segmentation model
---------------------------------------------------------
Usage:
  python train_seg.py                          # use configs/seg.yaml
  python train_seg.py --config path/to/cfg.yaml
  python train_seg.py --resume checkpoints_seg/epoch_0010.pth
  python train_seg.py --load-weights checkpoints_seg/best.pth
"""

import argparse
import random
import threading
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.seg_dataset import WatermarkSegDataset
from src.seg_model   import build_seg_model
from src.model       import count_params
from src.seg_trainer import SegTrainer
from src.live_plot   import SegLivePlotter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg: dict):
    ds_cfg    = cfg["dataset"]
    use_cuda  = cfg.get("device", "cpu") == "cuda"

    full_ds = WatermarkSegDataset(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        augment=True,
    )

    n = len(full_ds)
    max_samples = ds_cfg.get("max_samples") or n
    max_samples = min(max_samples, n)

    rng     = torch.Generator().manual_seed(cfg["seed"])
    all_idx = torch.randperm(n, generator=rng).tolist()[:max_samples]

    n_train = int(max_samples * ds_cfg["train_split"])
    tr_ds   = Subset(full_ds, all_idx[:n_train])

    val_full = WatermarkSegDataset(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        augment=False,
    )
    va_ds = Subset(val_full, all_idx[n_train:])

    nw = ds_cfg["num_workers"]
    train_loader = DataLoader(
        tr_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        va_ds, batch_size=1,
        shuffle=False, num_workers=nw, pin_memory=use_cuda,
    )
    return train_loader, val_loader, val_full, all_idx, n_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seg.yaml")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from (full state)")
    parser.add_argument("--load-weights", default=None,
                        help="path to checkpoint to load weights from (fresh start)")
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

    model    = build_seg_model(cfg)
    n_params = count_params(model)
    print(f"Model: {n_params:,} trainable parameters")

    train_loader, val_loader, val_full, all_idx, n_train = make_loaders(cfg)

    # Pick a fixed sample for the live visualisation probe — from the val subset
    # so the visualised output reflects the same distribution as the IoU metric.
    val_sample_dirs = [val_full.samples[i] for i in all_idx[n_train:]]
    sample_dir = random.choice(val_sample_dirs) if val_sample_dirs else None
    if sample_dir:
        print(f"Visualisation sample: {sample_dir.name}")

    plotter = SegLivePlotter(
        log_dir=Path(cfg["logging"]["dir"]),
        total_epochs=cfg["training"]["epochs"],
    )

    trainer = SegTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
        load_weights=args.load_weights,
        plotter=plotter,
    )
    trainer._sample_dir = sample_dir

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
