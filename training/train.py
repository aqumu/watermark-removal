"""
train.py  main entry point
------------------------------
Usage:
  python train.py                          # use configs/train_512.yaml
  python train.py --config path/to/cfg.yaml
  python train.py --resume artifacts/checkpoints/removal/epoch_0010.pth
"""

import argparse
import random
import threading
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tasks.removal.dataset import make_splits
from src.tasks.removal.model import build_model, count_params
from src.common.dashboard_runtime import maybe_start_dashboard
from src.common.restoration import iter_sample_dirs
from src.common.run_context import prepare_run_context
from src.tasks.removal.store_cli import clear_store, load_removal_cfg, rebuild_store, resolve_store_context
from src.tasks.removal.trainer import Trainer
from wm_shared.experiment import ExperimentSession
from src.common.training_control import TrainingPaused


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(path: str) -> dict:
    return load_removal_cfg(path)


def make_loaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    loss_cfg = cfg.get("loss", {})
    store_ctx = resolve_store_context(cfg)

    tr_ds, va_ds = make_splits(
        root=ds_cfg["root"],
        image_width=ds_cfg["image_width"],
        image_height=ds_cfg["image_height"],
        train_frac=ds_cfg["train_split"],
        seed=cfg["seed"],
        loss_mask_blur_pct=loss_cfg.get("loss_mask_blur_pct", 0.0),
        max_samples=ds_cfg.get("max_samples"),
        crop_aspect_ratio=ds_cfg.get("crop_aspect_ratio", 3.54),
        crop_margin_ratio=ds_cfg.get("crop_margin_ratio", 0.10),
        crop_min_width_ratio=ds_cfg.get("crop_min_width_ratio", 0.50),
        preprocessed_store_root=store_ctx["store_root"],
        preprocessed_store_signature=store_ctx["signature"],
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
    parser.add_argument("--config", default="configs/train_256.yaml")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume from (full state)")
    parser.add_argument("--load-weights", default=None,
                        help="path to checkpoint to load weights from (fresh start)")
    parser.add_argument("--reset-upsamplers", action="store_true",
                        help="re-apply ICNR to decoder upsampling convs after loading "
                             "checkpoint; fixes checkerboard artifacts without full "
                             "retraining use together with --resume or --load-weights")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Manually override the starting epoch (useful for resuming weights with new LR).")
    parser.add_argument("--clear-preprocessed-store", action="store_true",
                        help="delete the current removal preprocessed store namespace before training")
    parser.add_argument("--rebuild-preprocessed-store", action="store_true",
                        help="delete and fully regenerate the current removal preprocessed store namespace before training")
    parser.add_argument("--force-continue", action="store_true",
                        help="Force continue in the same folder even if config changed (dangerous!).")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_context = prepare_run_context(
        task_name="removal",
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

    model = build_model(cfg)
    n_params = count_params(model)
    print(f"Model: {n_params:,} trainable parameters")

    store_ctx = resolve_store_context(cfg)
    print(f"Preprocessed store: {store_ctx['store_root']} [{store_ctx['signature']}]")

    if args.clear_preprocessed_store or args.rebuild_preprocessed_store:
        cleared = clear_store(cfg)
        print(f"Cleared {cleared}")

    if args.rebuild_preprocessed_store:
        total, namespace = rebuild_store(cfg)
        print(f"Prepared {total} preprocessed samples in {namespace}")

    train_loader, val_loader = make_loaders(cfg)

    _candidates = iter_sample_dirs(cfg["dataset"]["root"])
    sample_dir = random.choice(_candidates) if _candidates else None
    if sample_dir:
        print(f"Visualisation sample: {sample_dir.name}")

    experiment = ExperimentSession(task_name="removal", cfg=cfg, manifest=run_context.manifest, dashboard=dashboard_sink)
    experiment.log_model_overview(
        model_name=type(model).__name__,
        parameter_count=n_params,
        optimizer_name="AdamW",
        scheduler_name=cfg["training"].get("lr_scheduler", "none"),
        extra={
            "image_width": cfg["dataset"]["image_width"],
            "image_height": cfg["dataset"]["image_height"],
            "batch_size": cfg["training"]["batch_size"],
            "grad_accum_steps": cfg["training"].get("grad_accum_steps", 1),
            "ema_decay": cfg["training"].get("ema_decay", 0.0),
        },
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
        load_weights=args.load_weights,
        start_epoch=args.start_epoch,
        experiment=experiment,
        dashboard=dashboard_sink,
    )

    trainer._sample_dir = sample_dir

    if args.reset_upsamplers:
        if not (args.resume or args.load_weights):
            print("[WARNING] --reset-upsamplers has no effect without --resume or --load-weights; ignoring")
        elif hasattr(model, "reset_decoder_upsamplers"):
            model.reset_decoder_upsamplers()
        else:
            print("[WARNING] --reset-upsamplers is only supported for MaskedUNet; ignoring")

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
