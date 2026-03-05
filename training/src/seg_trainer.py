"""
SegTrainer
----------
Training loop for the watermark segmentation model.

Loss    : Focal + Dice  (soft targets in [0, 1])
          Focal loss addresses class imbalance (watermark pixels are a minority).
Metric  : IoU at threshold 0.5 on the validation set
"""

import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .trainer import CSVLogger, _save_ckpt, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# loss
# ──────────────────────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.8) -> torch.Tensor:
    """
    Binary focal loss on soft [0, 1] targets.
    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive pixels
    Downweights easy negatives so the model focuses on hard foreground pixels.
    """
    bce_elem = F.binary_cross_entropy(pred, target, reduction="none")
    # p_t: predicted probability for the ground-truth class
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce_elem).mean()


def seg_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Focal + Dice on soft [0, 1] targets.

    Using soft targets (not binarized) for both losses lets the model learn
    the feathered alpha gradient at watermark edges rather than forcing it to
    predict hard binary values it was never given.
    """
    focal = focal_loss(pred, target)

    p    = pred.view(-1)
    t    = target.view(-1)
    intersection = (p * t).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)

    total = focal + dice
    return total, {"total": total.item(), "focal": focal.item(), "dice": dice.item()}


# ──────────────────────────────────────────────────────────────────────────────
# metric
# ──────────────────────────────────────────────────────────────────────────────

def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    tgt_bin  = (target > threshold).float()
    intersection = (pred_bin * tgt_bin).sum().item()
    union        = ((pred_bin + tgt_bin) > 0).float().sum().item()
    return intersection / max(union, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# trainer
# ──────────────────────────────────────────────────────────────────────────────

class SegTrainer:
    def __init__(self, model: nn.Module, cfg: dict,
                 train_loader: DataLoader, val_loader: DataLoader,
                 resume: str | None = None):

        self.model        = model
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = torch.device(cfg["device"])
        self.model.to(self.device)

        tc = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
        )
        self.scheduler = self._build_scheduler(tc)
        self.grad_clip = tc.get("grad_clip", 0.0)

        self.epochs      = tc["epochs"]
        self.start_epoch = 1
        self.best_iou    = 0.0
        self.val_every   = tc.get("val_every", 1)

        cc = cfg["checkpointing"]
        self.ckpt_dir   = Path(cc["dir"])
        self.save_every = cc["save_every"]
        self.keep_last  = cc["keep_last"]

        lc = cfg["logging"]
        self.log_every = lc["log_every"]
        run_dir = Path(lc["dir"])
        self.train_logger = CSVLogger(run_dir / "train.csv")
        self.val_logger   = CSVLogger(run_dir / "val.csv")

        if resume:
            epoch, best = load_checkpoint(resume, model, self.optimizer, self.scheduler)
            self.start_epoch = epoch + 1
            self.best_iou    = best
            print(f"Resumed from {resume} (epoch {epoch}, best IoU {best:.4f})")

    def _build_scheduler(self, tc: dict):
        name = tc.get("lr_scheduler", "none")
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc["epochs"], eta_min=1e-6
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        return None

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step = 0

        for batch in self.train_loader:
            inp    = batch["input" ].to(self.device)   # Bx3xHxW
            target = batch["target"].to(self.device)   # Bx1xHxW

            pred = self.model(inp)                     # Bx1xHxW in [0,1]
            loss, breakdown = seg_loss(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                row = {"epoch": epoch, "step": step, "lr": f"{lr:.2e}", **breakdown}
                self.train_logger.log(row)
                print(f"  [epoch {epoch:3d} | step {step:4d}] "
                      f"loss={breakdown['total']:.4f}  "
                      f"focal={breakdown['focal']:.4f}  "
                      f"dice={breakdown['dice']:.4f}  "
                      f"lr={lr:.2e}")

        return total_loss / max(step, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_iou = 0.0
        n = 0

        for batch in self.val_loader:
            inp    = batch["input" ].to(self.device)
            target = batch["target"].to(self.device)

            pred = self.model(inp)

            for i in range(pred.shape[0]):
                total_iou += iou(pred[i], target[i])
                n += 1

        avg_iou = total_iou / max(n, 1)
        self.val_logger.log({"epoch": epoch, "iou": f"{avg_iou:.4f}"})
        return avg_iou

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")
        print(f"  train: {len(self.train_loader.dataset)} samples  "
              f"val: {len(self.val_loader.dataset)} samples")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            avg_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            do_val  = (epoch % self.val_every == 0) or (epoch == self.epochs)
            avg_iou = self._validate(epoch) if do_val else self.best_iou
            elapsed = time.time() - t0

            is_best = do_val and (avg_iou > self.best_iou)
            if is_best:
                self.best_iou = avg_iou
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_iou, self.keep_last)
                marker = " * best"
            elif epoch % self.save_every == 0:
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_iou, self.keep_last)
                marker = ""
            else:
                marker = ""

            iou_str = f"{avg_iou:.4f}" if do_val else "  --  "
            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"loss={avg_loss:.4f}  iou={iou_str}{marker}  "
                  f"({elapsed:.0f}s)")

        print(f"\nDone. Best IoU: {self.best_iou:.4f}")
