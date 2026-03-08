"""
Trainer
-------
Handles the full train / validate loop with:
  - periodic checkpointing (keeps only the last K)
  - rolling loss log to a plain CSV
  - PSNR and SSIM metrics on the validation set
  - cosine or step LR schedule
"""

import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import CombinedLoss


# ──────────────────────────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    """PSNR between two [-1,1] tensors. max_val = 1-(-1) = 2.0"""
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


import tempfile
import shutil

def _safe_delete(path: Path, max_retries: int = 5):
    """Attempt to delete a file with retries for Windows locking issues."""
    for i in range(max_retries):
        try:
            if path.exists():
                path.unlink()
            return True
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    return False

def _save_ckpt(ckpt_dir: Path, epoch: int, model: nn.Module,
               optimizer, scheduler, best_psnr: float, keep_last: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:04d}.pth"
    
    # "Step-Aside" Save: Write to system TEMP first to avoid OneDrive/Defender 
    # reactive locks in the project folder.
    fd, temp_path_str = tempfile.mkstemp(suffix=".pth.tmp")
    os.close(fd) # Close handle, PyTorch will open it
    temp_path = Path(temp_path_str)
    
    try:
        # Save to global temp where background scanners are less aggressive
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "best_psnr":  best_psnr,
        }, temp_path)
        
        # Give OS a moment to settle
        time.sleep(0.2)
        
        # Move to destination (Atomic if on same drive, shutil.move handles cross-drive)
        _safe_delete(path)
        
        # Retry move if locked
        for i in range(10):
            try:
                shutil.move(str(temp_path), str(path))
                break
            except (PermissionError, OSError):
                time.sleep(0.5 * (i + 1))
        
    except Exception as e:
        _safe_delete(temp_path)
        print(f"  [WARNING] Failed to save checkpoint at epoch {epoch}: {e}")
        return None

    # remove old checkpoints
    ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
    while len(ckpts) > keep_last:
        _safe_delete(ckpts[0])
        ckpts = ckpts[1:]

    return path


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# CSV logger
# ──────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fields = None

    def log(self, row: dict):
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self._fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self._fields).writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model: nn.Module, cfg: dict,
                 train_loader: DataLoader, val_loader: DataLoader,
                 resume: str | None = None,
                 load_weights: str | None = None):

        self.model       = model
        self.cfg         = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = torch.device(cfg["device"])
        self.model.to(self.device)

        self.criterion = CombinedLoss(cfg).to(self.device)
        # run VGG only every N steps (1 = every step, 4 = every 4th step, etc.)
        self.perceptual_every = cfg["loss"].get("perceptual_every", 1)
        self.val_every = cfg["training"].get("val_every", 1)

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
        self.best_psnr   = 0.0

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
            self.start_epoch, self.best_psnr = load_checkpoint(
                resume, model, self.optimizer, self.scheduler
            )
            self.start_epoch += 1
            print(f"Resumed from {resume} (epoch {self.start_epoch - 1}, "
                  f"best PSNR {self.best_psnr:.2f})")
        elif load_weights:
            ckpt = torch.load(load_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded weights from {load_weights} (starting fresh from epoch 1)")

    # ── scheduler ─────────────────────────────────────────────────────────

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

    # ── one training epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step = 0

        for batch_idx, batch in enumerate(self.train_loader):
            inp    = batch["input" ].to(self.device)   # Bx4xHxW
            target = batch["target"].to(self.device)   # Bx3xHxW
            mask   = batch["mask"  ].to(self.device)   # Bx1xHxW

            delta      = self.model(inp)                        # residual in [0,1]
            pred_clean = (inp[:, :3] - delta).clamp(-1, 1)    # reconstructed clean
            use_perc = (self.perceptual_every <= 1 or
                        step % self.perceptual_every == 0)
            loss, breakdown = self.criterion(pred_clean, target, mask,
                                             use_perceptual=use_perc)

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
                      f"l1={breakdown['l1_full']:.4f}  "
                      f"ssim={breakdown['ssim']:.4f}  "
                      f"perc={breakdown['perceptual']:.4f}  "
                      f"lr={lr:.2e}")

        return total_loss / max(step, 1)

    # ── validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_psnr = 0.0
        n = 0

        for batch in self.val_loader:
            inp    = batch["input" ].to(self.device)
            target = batch["target"].to(self.device)
            mask   = batch["mask"  ].to(self.device)

            delta      = self.model(inp)
            pred_clean = (inp[:, :3] - delta).clamp(-1, 1)

            for i in range(pred_clean.shape[0]):
                total_psnr += psnr(pred_clean[i], target[i])
                n += 1

        avg_psnr = total_psnr / max(n, 1)
        self.val_logger.log({"epoch": epoch, "psnr": f"{avg_psnr:.4f}"})
        return avg_psnr

    # ── main loop ─────────────────────────────────────────────────────────

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")
        print(f"  train: {len(self.train_loader.dataset)} samples  "
              f"val: {len(self.val_loader.dataset)} samples")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            avg_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            do_val   = (epoch % self.val_every == 0) or (epoch == self.epochs)
            avg_psnr = self._validate(epoch) if do_val else self.best_psnr
            elapsed  = time.time() - t0

            is_best = do_val and (avg_psnr > self.best_psnr)
            if is_best:
                self.best_psnr = avg_psnr
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_psnr, self.keep_last)
                marker = " * best"
            elif epoch % self.save_every == 0:
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_psnr, self.keep_last)
                marker = ""
            else:
                marker = ""

            psnr_str = f"{avg_psnr:.2f} dB" if do_val else "  --   "
            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"loss={avg_loss:.4f}  psnr={psnr_str}{marker}  "
                  f"({elapsed:.0f}s)")

        print(f"\nDone. Best PSNR: {self.best_psnr:.2f} dB")
