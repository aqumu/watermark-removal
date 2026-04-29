"""
train_seg.py  train the watermark segmentation model
---------------------------------------------------------
Usage:
  python train_seg.py                          # use configs/seg.yaml
  python train_seg.py --config path/to/cfg.yaml
  python train_seg.py --resume artifacts/checkpoints/segmentation/epoch_0010.pth
  python train_seg.py --load-weights artifacts/checkpoints/segmentation/best.pth
"""

import argparse
import random
import threading
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tasks.segmentation.dataset import WatermarkSegDataset
from src.tasks.segmentation.model import build_seg_model
from src.tasks.removal.model import count_params
from src.common.dashboard_runtime import maybe_start_dashboard
from src.common.run_context import prepare_run_context
from src.common.training_control import TrainingPaused
from src.tasks.segmentation.trainer import SegTrainer
from wm_shared.config import load_yaml_config
from wm_shared.experiment import ExperimentSession


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    return load_yaml_config(path)


def make_loaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    use_cuda = cfg.get("device", "cpu") == "cuda"

    full_ds = WatermarkSegDataset(
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
    parser.add_argument("--force-continue", action="store_true",
                        help="Force continue in the same folder even if config changed (dangerous!).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_context = prepare_run_context(
        task_name="segmentation",
        cfg=cfg,
        config_path=args.config,
        resume=args.resume,
        load_weights=args.load_weights,
        repo_root=REPO_ROOT,
        force_continue=args.force_continue,
    )
    dashboard_runtime = maybe_start_dashboard(cfg=cfg, manifest=run_context.manifest)
    dashboard_sink = dashboard_runtime.create_sink() if dashboard_runtime else None
    set_seed(cfg["seed"])

    if cfg.get("device", "cpu") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {cfg['device']}")
    print(f"Run ID: {run_context.manifest.identity.run_id}")

    n_threads = cfg.get("torch_threads", 0)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        print(f"PyTorch intra-op threads: {torch.get_num_threads()}")

    model = build_seg_model(cfg)
    n_params = count_params(model)
    print(f"Model: {n_params:,} trainable parameters")

    train_loader, val_loader, val_full, all_idx, n_train = make_loaders(cfg)

    val_sample_dirs = [val_full.samples[i] for i in all_idx[n_train:]]
    sample_dir = random.choice(val_sample_dirs) if val_sample_dirs else None
    if sample_dir:
        print(f"Visualisation sample: {sample_dir.name}")

    experiment = ExperimentSession(task_name="segmentation", cfg=cfg, manifest=run_context.manifest, dashboard=dashboard_sink)
    experiment.log_model_overview(
        model_name=type(model).__name__,
        parameter_count=n_params,
        optimizer_name="AdamW",
        scheduler_name=cfg["training"].get("lr_scheduler", "none"),
        extra={
            "image_size": cfg["dataset"]["image_size"],
            "batch_size": cfg["training"]["batch_size"],
            "ema_decay": cfg["training"].get("ema_decay", 0.0),
        },
    )

    trainer = SegTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
        load_weights=args.load_weights,
        experiment=experiment,
        dashboard=dashboard_sink,
    )
    trainer._sample_dir = sample_dir

    def _train_worker(trainer):
        try:
            trainer.train()
            experiment.set_status("completed")
        except TrainingPaused:
            experiment.set_status("paused")
        except Exception:
            experiment.set_status("failed")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_train_worker, args=(trainer,), daemon=True)
    thread.start()
    thread.join()
    if dashboard_runtime is not None:
        dashboard_runtime.close()


if __name__ == "__main__":
    main()
