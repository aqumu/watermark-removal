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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import make_splits
from src.losses  import CombinedLoss
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
    loss_cfg = cfg.get("loss", {})
    
    tr_ds, va_ds = make_splits(
        root=ds_cfg["root"],
        image_size=ds_cfg["image_size"],
        train_frac=ds_cfg["train_split"],
        seed=cfg["seed"],
        loss_mask_blur_pct=loss_cfg.get("loss_mask_blur_pct", 0.0),
        loss_mask_dilate_pct=loss_cfg.get("loss_mask_dilate_pct", 0.0),
        max_samples=ds_cfg.get("max_samples")
    )

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
        load_weights=args.load_weights,
    )
    trainer.train()


if __name__ == "__main__":
    main()
