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

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import WatermarkDataset
from src.model   import build_model, count_params
from src.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg: dict):
    ds_cfg = cfg["dataset"]

    full_ds = WatermarkDataset(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        augment=True,
    )

    n = len(full_ds)
    max_samples = ds_cfg.get("max_samples") or n
    max_samples = min(max_samples, n)

    rng = torch.Generator().manual_seed(cfg["seed"])
    all_idx = torch.randperm(n, generator=rng).tolist()[:max_samples]

    n_train = int(max_samples * ds_cfg["train_split"])
    tr_ds = Subset(full_ds, all_idx[:n_train])
    va_ds_idx = all_idx[n_train:]

    # val set should not augment – we wrap a fresh no-aug dataset
    val_full = WatermarkDataset(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        augment=False,
    )
    va_ds = Subset(val_full, va_ds_idx)

    nw = ds_cfg["num_workers"]
    train_loader = DataLoader(
        tr_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=nw, pin_memory=False,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        va_ds, batch_size=1,
        shuffle=False, num_workers=nw, pin_memory=False,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from")
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

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
