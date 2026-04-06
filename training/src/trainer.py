"""
Trainer
-------
Handles the full train / validate loop with:
  - periodic checkpointing (keeps only the last K)
  - rolling loss log to a plain CSV
  - PSNR and SSIM metrics on the validation set
  - cosine LR schedule with linear warmup
  - EMA of model weights (best.pth always saves EMA weights)
"""

import csv
import copy
import math
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    After each optimiser step, call ema.update(model).
    Use ema.apply_to(target_model) to copy EMA weights into a model for eval.

    Decay = 0.9995 means the shadow weights change by ~0.05% per step toward
    the live weights — smooth enough to suppress cosine-annealing oscillations
    without lagging behind genuine improvements.

    Zero inference overhead: EMA weights are ordinary parameter tensors stored
    in CPU memory (shadow dict). They are only copied to GPU when needed for
    validation or best.pth saving.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        # Store EMA weights in CPU fp32 to avoid occupying GPU memory.
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().float().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module):
        """Blend live model weights into the EMA shadow (call after every step)."""
        with torch.no_grad():
            for k, v in model.state_dict().items():
                shadow_v = self.shadow[k]
                live_v   = v.detach().float().cpu()
                self.shadow[k] = shadow_v * self.decay + live_v * (1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        """Copy EMA weights into model (in-place, preserves device)."""
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device)
             for k, v in self.shadow.items()},
            strict=True
        )

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict):
        self.decay  = sd["decay"]
        self.shadow = sd["shadow"]


# ──────────────────────────────────────────────────────────────────────────────
# checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

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
               optimizer, scheduler, best_psnr: float, keep_last: int,
               ema: ModelEMA | None = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:04d}.pth"

    # "Step-Aside" Save: Write to system TEMP first to avoid OneDrive/Defender
    # reactive locks in the project folder.
    fd, temp_path_str = tempfile.mkstemp(suffix=".pth.tmp")
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        payload = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "best_psnr":  best_psnr,
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()

        torch.save(payload, temp_path)
        time.sleep(0.2)
        _safe_delete(path)

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


def _save_ckpt_to(path: Path, epoch: int, model: nn.Module,
                  optimizer, scheduler, best_psnr: float,
                  ema: ModelEMA | None = None):
    """Write a checkpoint to an explicit path with no rotation (used for best.pth).

    If EMA is supplied, the saved model weights are the EMA shadow weights —
    this makes best.pth the smoothest version of the model rather than the
    live (oscillating) weights at the time of best validation PSNR.
    """
    import tempfile, os, shutil
    fd, tmp = tempfile.mkstemp(suffix=".pth.tmp")
    os.close(fd)

    # Use EMA weights if available, otherwise live weights.
    saved_state = (
        {k: v.to(next(model.parameters()).device)
         for k, v in ema.shadow.items()}
        if ema is not None else model.state_dict()
    )

    try:
        torch.save({
            "epoch":     epoch,
            "model":     saved_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_psnr": best_psnr,
        }, tmp)
        time.sleep(0.1)
        _safe_delete(path)
        shutil.move(tmp, str(path))
    except Exception as e:
        _safe_delete(Path(tmp))
        print(f"  [WARNING] Failed to save best checkpoint: {e}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None,
                    ema: ModelEMA | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and ckpt.get("ema"):
        ema.load_state_dict(ckpt["ema"])
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
            # Only write header if the file doesn't exist or is empty
            if not self.path.exists() or self.path.stat().st_size == 0:
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
                 load_weights: str | None = None,
                 start_epoch: int = 1,
                 plotter=None):

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

        use_amp = cfg["training"].get("amp", False) and self.device.type == "cuda"
        # bf16 has fp32 dynamic range — no overflow, no scaler needed
        self.amp_dtype = torch.bfloat16 if use_amp else torch.float32
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)  # not needed for bf16
        if use_amp:
            print("AMP (bfloat16 mixed precision) enabled")

        tc = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
        )

        self.epochs        = tc["epochs"]
        self.start_epoch   = start_epoch
        self.best_psnr     = 0.0
        self.scheduler     = self._build_scheduler(tc)
        self.grad_clip     = tc.get("grad_clip", 0.0)

        # GPU-side Preprocessing Buffers
        self.image_size = cfg["dataset"]["image_size"]
        self.loss_mask_blur_pct = cfg["loss"].get("loss_mask_blur_pct", 0.0)
        
        # Sobel kernels: 1x1x3x3 filters for X and Y gradients
        self.sobel_kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.sobel_ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        # EMA — track smoothed weights for cleaner inference and best.pth
        ema_decay = tc.get("ema_decay", 0.0)
        self.ema  = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
        if self.ema:
            print(f"EMA enabled (decay={ema_decay})")
        else:
            print("EMA disabled")

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
                resume, model, self.optimizer, self.scheduler, self.ema
            )
            self.start_epoch += 1
            print(f"Resumed from {resume} (epoch {self.start_epoch - 1}, "
                  f"best PSNR {self.best_psnr:.2f})")
        elif load_weights:
            ckpt = torch.load(load_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded weights from {load_weights} (starting from epoch {self.start_epoch})")

        self.plotter       = plotter
        # Calculate global step from restored epoch so new logs align to the timeline
        self._global_step  = (self.start_epoch - 1) * len(self.train_loader)
        self._sample_dir   = None   # set from train.py after construction

    # ── scheduler ─────────────────────────────────────────────────────────

    def _build_scheduler(self, tc: dict):
        name = tc.get("lr_scheduler", "none")
        total_epochs = tc["epochs"]

        if name in ("cosine", "cosine_warmup"):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=1e-6
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        return None

    # ── GPU preprocessing ────────────────────────────────────────────────
    
    def _prepare_batch_gpu(self, batch: dict, training: bool):
        """Move logic for Sobel/Dilation from CPU Dataloader to GPU.
        
        Input batch contains:
          wm: Bx3xHxW [-1, 1]
          target: Bx3xHxW [-1, 1]
          mask_loss: Bx1xHxW [0, 1] (blurred)
          mask_raw: Bx1xHxW [0, 1] (binary)
        """
        wm        = batch["wm"].to(self.device)
        target    = batch["target"].to(self.device)
        mask_loss = batch["mask_loss"].to(self.device)
        mask_raw  = batch["mask_raw"].to(self.device)
        
        # 1. Mask Dilation (Model input hint)
        # Matches dilate_mask_input in image_utils.py
        if training:
            # Random jitter
            max_t = max(1, round(4 * self.image_size / 256))
            tx = torch.randint(-max_t, max_t + 1, (1,)).item()
            ty = torch.randint(-max_t, max_t + 1, (1,)).item()
            mask_aug = torch.roll(mask_raw, shifts=(ty, tx), dims=(2, 3))
            
            # Random dilation size
            k = torch.tensor([3, 5, 7, 3, 5, 7, 3, 5, 7, 0])[torch.randint(0, 10, (1,))].item()
            if k > 0:
                mask_hint = F.max_pool2d(mask_aug, kernel_size=k, stride=1, padding=k // 2)
            else:
                mask_hint = mask_aug
        else:
            # Deterministic dilation for inference/validation
            mask_hint = F.max_pool2d(mask_raw, kernel_size=5, stride=1, padding=2)

        # 2. Sobel Gradient (Model input hint)
        # Matches compute_gradient in image_utils.py (roughly)
        with torch.no_grad():
            # Grayscale: [0.299, 0.587, 0.114]
            gray = 0.299 * wm[:, 0:1] + 0.587 * wm[:, 1:2] + 0.114 * wm[:, 2:3]
            # Convert [-1,1] back to [0,1] for gradient math consistency
            gray = (gray + 1.0) / 2.0
            
            gx = F.conv2d(gray, self.sobel_kx, padding=1)
            gy = F.conv2d(gray, self.sobel_ky, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            
            # Normalize to [0, 1] per batch
            m_max = mag.view(mag.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            mag = mag / (m_max + 1e-8)
            
        # 3. Final Concatenation
        inp = torch.cat([wm, mask_hint, mag], dim=1)  # Bx5xHxW
        
        return inp, target, mask_loss, wm

    # ── one training epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step = 0
        skipped_steps = 0
        scale_before = self.scaler.get_scale()

        grad_accum_steps = self.cfg["training"].get("grad_accum_steps", 1)
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            inp, target, mask, wm = self._prepare_batch_gpu(batch, training=True)

            use_perc = (self.perceptual_every <= 1 or
                        self._global_step % self.perceptual_every == 0)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                delta      = self.model(inp)
                pred_clean = (wm - delta).clamp(-1, 1)
                loss, breakdown = self.criterion(pred_clean, target, mask,
                                                 delta=delta,
                                                 wm=wm,
                                                 use_perceptual=use_perc)

            self.scaler.scale(loss / grad_accum_steps).backward()

            is_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))

            if is_step:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip if self.grad_clip > 0 else float("inf")
                ).item()
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scaler.get_scale() < scale_before_step:
                    skipped_steps += 1

                # EMA: blend live weights into shadow after every optimiser step
                if self.ema is not None:
                    self.ema.update(self.model)

                step += 1
                self._global_step += 1

            total_loss += loss.item()

            if is_step and (step % self.log_every == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                clipped = grad_norm > self.grad_clip > 0
                row = {"epoch": epoch, "step": step, "global_step": self._global_step,
                       "lr": f"{lr:.2e}", "grad_norm": f"{grad_norm:.4f}", **breakdown}
                self.train_logger.log(row)
                if self.plotter:
                    self.plotter.log_step(self._global_step, breakdown)
                print(f"  [epoch {epoch:3d} | step {step:4d}] "
                      f"loss={breakdown['total']:.4f}  "
                      f"l1_m={breakdown['l1_masked']:.4f}  "
                      f"perc={breakdown['perceptual']:.4f}  "
                      f"sat={breakdown['saturation']:.4f}  "
                      f"cm={breakdown['color_moment']:.4f}  "
                      f"border={breakdown['border']:.4f}  "
                      f"itv={breakdown['interior_tv']:.4f}  "
                      f"btv={breakdown['bg_tv']:.4f}  "
                      f"bd={breakdown['bg_delta']:.4f}  "
                      f"grad={grad_norm:.3f}{'*' if clipped else ''}  "
                      f"lr={lr:.2e}")

            total_loss += loss.item()

        if self.use_amp and skipped_steps > 0:
            scale_after = self.scaler.get_scale()
            print(f"  [AMP] skipped {skipped_steps}/{step} optimizer steps due to inf/nan gradients  "
                  f"(scale {scale_before:.0f} → {scale_after:.0f})")

        return total_loss / len(self.train_loader)

    # ── validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Validate using EMA weights when available (EMA = best inference quality)."""
        # Temporarily copy EMA weights into the model for evaluation.
        # After validation we restore the live (training) weights.
        live_state = None
        if self.ema is not None:
            live_state = copy.deepcopy(self.model.state_dict())
            self.ema.copy_to(self.model)

        self.model.eval()
        total_psnr        = 0.0
        total_psnr_masked = 0.0
        n = 0

        for batch in self.val_loader:
            inp, target, mask, wm = self._prepare_batch_gpu(batch, training=False)

            delta      = self.model(inp)
            pred_clean = (wm - delta).clamp(-1, 1)

            for i in range(pred_clean.shape[0]):
                total_psnr += psnr(pred_clean[i], target[i])

                # Masked PSNR: measure reconstruction quality inside the
                # watermark region only, where the hard work actually happens.
                m = (mask[i] > 0.5).float()
                n_px = m.sum().item()
                if n_px > 0:
                    mse_masked = ((pred_clean[i] - target[i]) ** 2 * m).sum().item() / (n_px * 3)
                    total_psnr_masked += (10 * math.log10(4.0 / mse_masked)
                                          if mse_masked > 0 else 100.0)
                n += 1

        # Restore live weights
        if live_state is not None:
            self.model.load_state_dict(live_state)
        # Reset training mode
        self.model.train()

        avg_psnr        = total_psnr        / max(n, 1)
        avg_psnr_masked = total_psnr_masked / max(n, 1)
        self.val_logger.log({"epoch": epoch, "global_step": self._global_step,
                             "psnr": f"{avg_psnr:.4f}",
                             "psnr_masked": f"{avg_psnr_masked:.4f}"})
        print(f"  val psnr={avg_psnr:.2f} dB  psnr_masked={avg_psnr_masked:.2f} dB"
              + ("  [EMA]" if self.ema else ""))
        if self.plotter:
            self.plotter.log_val(self._global_step, avg_psnr, avg_psnr_masked)
            if self._sample_dir is not None:
                result = self._infer_sample()
                if result is not None:
                    self.plotter.log_images(*result)
        return avg_psnr_masked

    # ── visualisation sample inference ────────────────────────────────────

    @torch.no_grad()
    def _infer_sample(self, size: int = 256):
        """Run inference on the fixed visualisation sample.

        Matches the infer.py preprocessing pipeline exactly so the three
        returned images correspond to debug_0, debug_3, and debug_6.
        Returns (wm_bgr, pred_bgr, wloss_f32) or None on any failure.
        """
        import cv2
        import numpy as np
        from src.image_utils import compute_gradient, dilate_mask_input, blur_mask_for_loss

        wm    = cv2.imread(str(self._sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask  = cv2.imread(str(self._sample_dir / "mask.png"),        cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(self._sample_dir / "clean.png"),       cv2.IMREAD_COLOR)
        if any(x is None for x in (wm, mask, clean)):
            return None

        wm_r    = cv2.resize(wm,    (size, size), interpolation=cv2.INTER_AREA)
        mask_r  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
        clean_r = cv2.resize(clean, (size, size), interpolation=cv2.INTER_AREA)

        mask_binary = (mask_r > 127).astype(np.float32)
        mask_input  = dilate_mask_input(mask_binary, augment=False)

        rgb_np = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t  = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).to(self.device)
        grad_t = compute_gradient(wm_r).unsqueeze(0).to(self.device)
        inp    = torch.cat([rgb_t, mask_t, grad_t], dim=1)

        was_training = self.model.training
        self.model.eval()
        delta = self.model(inp)
        pred  = (inp[:, :3] - delta).clamp(-1, 1)
        if was_training:
            self.model.train()

        pred_np  = pred.squeeze(0).cpu().float().numpy()
        pred_np  = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        # weighted loss map — same formula as _weighted_loss_map in infer.py
        lc            = self.cfg.get("loss", {})
        soft_mask     = blur_mask_for_loss(mask_binary, lc.get("loss_mask_blur_pct", 0.0), size)
        kernel        = np.ones((3, 3), dtype=np.uint8)
        mask_interior = cv2.erode(soft_mask, kernel, iterations=1)
        border_w      = 4.0 * soft_mask * (1.0 - soft_mask)
        # Rename/Refactor l1_background to bg_delta
        bg_delta_w    = 1.0 - soft_mask
        abs_err       = np.abs(pred_bgr.astype(np.float32) / 127.5 - 1.0
                               - (clean_r.astype(np.float32) / 127.5 - 1.0)).mean(axis=2)
        wloss = (self.criterion.w_l1_masked     * abs_err * mask_interior
               + self.criterion.w_bg_delta      * abs_err * bg_delta_w
               + self.criterion.w_border        * abs_err * border_w).astype(np.float32)

        return wm_r, pred_bgr, wloss

    # ── main loop ─────────────────────────────────────────────────────────

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")
        print(f"  train: {len(self.train_loader.dataset)} samples  "
              f"val: {len(self.val_loader.dataset)} samples")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            progress = (epoch - 1) / max(self.epochs - 1, 1)
            self.criterion.set_progress(progress)
            avg_loss = self._train_epoch(epoch)

            if self.scheduler:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Detected call of", category=UserWarning)
                    self.scheduler.step()

            do_val   = (epoch % self.val_every == 0) or (epoch == self.epochs)
            avg_psnr = self._validate(epoch) if do_val else self.best_psnr
            elapsed  = time.time() - t0

            is_best = do_val and (avg_psnr > self.best_psnr)
            if is_best:
                self.best_psnr = avg_psnr
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_psnr, self.keep_last, self.ema)
                # best.pth always contains EMA weights (smoother inference)
                best_path = self.ckpt_dir / "best.pth"
                _save_ckpt_to(best_path, epoch, self.model,
                              self.optimizer, self.scheduler, self.best_psnr,
                              self.ema)
                marker = " * best"
            elif epoch % self.save_every == 0:
                _save_ckpt(self.ckpt_dir, epoch, self.model,
                           self.optimizer, self.scheduler,
                           self.best_psnr, self.keep_last, self.ema)
                marker = ""
            else:
                marker = ""

            psnr_str = f"{avg_psnr:.2f} dB" if do_val else "  --   "
            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"loss={avg_loss:.4f}  psnr={psnr_str}{marker}  "
                  f"({elapsed:.0f}s)")

        print(f"\nDone. Best PSNR: {self.best_psnr:.2f} dB")
