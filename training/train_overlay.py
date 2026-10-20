"""
train_overlay.py  train removal of an anchored overlay mark (e.g. Cian)
-----------------------------------------------------------------------
Usage:
  python training/train_overlay.py --config training/configs/cian_low.yaml
  python training/train_overlay.py --config ... --resume <checkpoint>

Data paths in the config are relative to the repo root; logging/checkpoint
directories are relative to training/. Ctrl+C saves a checkpoint and pauses.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.common.run_context import prepare_run_context
from src.common.run_log import tee_console_to_run_log
from src.common.training_control import TrainingPaused
from src.tasks.overlay.data import OverlaySynthDataset, is_holdout, load_backgrounds
from src.tasks.overlay.model import build_model, count_params
from src.tasks.overlay.trainer import OverlayTrainer
from wm_shared.experiment import ExperimentSession
from wm_shared.marks import load_mark


def worker_init(_):
    # one thread per data worker: OpenCV / torch thread pools in every worker
    # oversubscribe the CPU and make each sample several times slower
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("logging", "checkpointing"):
        d = Path(cfg[key]["dir"])
        cfg[key]["dir"] = str(d if d.is_absolute() else TRAINING_DIR / d)
    for src in cfg["data"]["backgrounds"]:
        for key in ("csv", "dir"):
            if key in src and not Path(src[key]).is_absolute():
                src[key] = str(REPO_ROOT / src[key])
    dist = cfg.get("distill")
    if dist and not Path(dist["checkpoint"]).is_absolute():
        dist["checkpoint"] = str(REPO_ROOT / dist["checkpoint"])
    prev = cfg["data"].get("preview_csv")
    if prev and not Path(prev).is_absolute():
        cfg["data"]["preview_csv"] = str(REPO_ROOT / prev)
    cfg.setdefault("logging", {}).setdefault("keep_latest_runs", 15)
    return cfg


def preview_photos(cfg: dict) -> list[tuple[np.ndarray, str]]:
    """Real stamped photos from held-out listings, for qualitative previews."""
    path = cfg["data"].get("preview_csv")
    n = int(cfg["data"].get("preview_count", 6))
    if not path or n <= 0:
        return []
    import cv2
    holdout = float(cfg["data"].get("holdout", 0.05))
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if is_holdout(r["listing_id"], holdout)]
    random.Random(1234).shuffle(rows)
    out, seen = [], set()
    for r in rows:
        if r["listing_id"] in seen:
            continue
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        seen.add(r["listing_id"])
        out.append((img[..., ::-1].copy(), r["listing_id"]))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(TRAINING_DIR / "configs" / "cian_low.yaml"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--load-weights", default=None)
    ap.add_argument("--force-continue", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_context = prepare_run_context(task_name="overlay", cfg=cfg, config_path=args.config,
                                      resume=args.resume, load_weights=args.load_weights,
                                      repo_root=REPO_ROOT, force_continue=args.force_continue)
    log_path = tee_console_to_run_log(run_context.manifest.paths.run_dir)
    print(f"Logging to {log_path}")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    print(f"Device: {cfg['device']}  Run ID: {run_context.manifest.identity.run_id}")

    profile = load_mark(cfg["mark"])
    dc = cfg["data"]
    holdout = float(dc.get("holdout", 0.05))
    bg_train = load_backgrounds(dc["backgrounds"], "train", holdout)
    bg_val = load_backgrounds(dc["backgrounds"], "val", holdout)
    print(f"Backgrounds: {len(bg_train)} train / {len(bg_val)} val")

    tc = cfg["training"]
    bs = int(tc["batch_size"])
    train_ds = OverlaySynthDataset(profile, bg_train, tc["steps_per_epoch"] * bs, dc, seed=None)
    val_ds = OverlaySynthDataset(profile, bg_val, int(dc.get("val_samples", 512)), dc, seed=cfg["seed"])
    nw = int(dc.get("num_workers", 8))
    use_cuda = cfg["device"] == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=True,
                              pin_memory=use_cuda, persistent_workers=nw > 0, worker_init_fn=worker_init,
                              prefetch_factor=int(dc.get("prefetch", 2)) if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=use_cuda, persistent_workers=nw > 0, worker_init_fn=worker_init)

    model = build_model(cfg)
    n_params = count_params(model)
    print(f"Model: OverlayUNet {tuple(cfg['model'].get('widths'))}, {n_params:,} parameters")

    experiment = ExperimentSession(task_name="overlay", cfg=cfg, manifest=run_context.manifest)
    experiment.log_model_overview(model_name="OverlayUNet", parameter_count=n_params, optimizer_name="AdamW",
                                  scheduler_name="warmup_cosine",
                                  extra={"window": list(profile.window), "batch_size": bs,
                                         "mark": profile.name})
    trainer = OverlayTrainer(model, cfg, profile, train_loader, val_loader, experiment=experiment,
                             preview_photos=preview_photos(cfg), resume=args.resume,
                             load_weights=args.load_weights)
    try:
        trainer.train()
    except TrainingPaused:
        experiment.set_status("paused")
        print("Paused. Continue with --resume <checkpoint>.")
    except BaseException:
        experiment.set_status("failed")
        raise
    else:
        experiment.set_status("completed")


if __name__ == "__main__":
    main()
