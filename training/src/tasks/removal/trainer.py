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

import copy
import math
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...common.checkpointing import CSVLogger, _save_ckpt, _save_ckpt_to, load_checkpoint
from ...common.metrics import psnr
from ...common.restoration import (
    blend_back,
    build_store_signature,
    get_store_root,
    load_or_build_preprocessed_sample,
    prepare_roi_input,
)
from ...common.training_control import TrainingPaused, clear_pause_request, pause_requested
from wm_shared.preprocess import compute_gradient, dilate_mask_input
from .inference import run_model, weighted_loss_map
from .losses import CombinedLoss
class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    After each optimiser step, call ema.update(model).
    Use ema.apply_to(target_model) to copy EMA weights into a model for eval.

    Decay = 0.9995 means the shadow weights change by ~0.05% per step toward
    the live weights вЂ” smooth enough to suppress cosine-annealing oscillations
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
class Trainer:
    def __init__(self, model: nn.Module, cfg: dict,
                 train_loader: DataLoader, val_loader: DataLoader,
                 resume: str | None = None,
                 load_weights: str | None = None,
                 start_epoch: int = 1,
                 experiment=None,
                 dashboard=None):

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
        # bf16 has fp32 dynamic range вЂ” no overflow, no scaler needed
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
        self.image_width = cfg["dataset"]["image_width"]
        self.image_height = cfg["dataset"]["image_height"]
        self.loss_mask_blur_pct = cfg["loss"].get("loss_mask_blur_pct", 0.0)
        
        # Sobel kernels: 1x1x3x3 filters for X and Y gradients
        self.sobel_kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.sobel_ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        # EMA вЂ” track smoothed weights for cleaner inference and best.pth
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
        self.run_dir = run_dir
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

        self.experiment    = experiment
        self.dashboard     = dashboard
        # Calculate global step from restored epoch so new logs align to the timeline.
        # _global_step increments once per optimizer step (every grad_accum_steps batches),
        # so steps_per_epoch = ceil(batches / grad_accum_steps), not raw batch count.
        _grad_accum = self.cfg["training"].get("grad_accum_steps", 1)
        _steps_per_epoch = math.ceil(len(self.train_loader) / _grad_accum)
        self._global_step  = (self.start_epoch - 1) * _steps_per_epoch
        self._sample_dir   = None   # set from train.py after construction

    def _pause_requested(self) -> bool:
        return pause_requested(self.run_dir)

    def _checkpoint_and_pause(self, epoch: int) -> None:
        checkpoint_path = _save_ckpt(
            self.ckpt_dir,
            epoch,
            self.model,
            self.optimizer,
            self.scheduler,
            self.best_psnr,
            self.keep_last,
            self.ema,
        )
        if self.experiment is not None and checkpoint_path is not None:
            self.experiment.stage_artifact(checkpoint_path, category="checkpoints")
        clear_pause_request(self.run_dir)
        raise TrainingPaused(f"Pause requested at epoch {epoch}")

    # в”Ђв”Ђ scheduler в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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

    # в”Ђв”Ђ GPU preprocessing в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    
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
        mask_input = batch.get("mask_input")
        if mask_input is not None:
            mask_input = mask_input.to(self.device)
        
        # 1. Mask input hint
        # Precomputed by precompute_aligned_masks.py: seg → xcorr alignment → 5px dilation.
        # Falls back to plain 5px dilation on the GT mask when cache is absent.
        if mask_input is not None:
            mask_hint = mask_input
        else:
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

    # в”Ђв”Ђ one training epoch в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_train_psnr = 0.0
        n_train = 0
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

                loss, breakdown = self.criterion(
                    pred_clean,
                    target,
                    mask,
                    delta=delta,
                    use_perceptual=use_perc,
                )

            # Accumulate training PSNR on live weights (free — pred_clean already computed)
            with torch.no_grad():
                pc_f = pred_clean.detach().float()
                tgt_f = target.float()
                for i in range(pc_f.shape[0]):
                    total_train_psnr += psnr(pc_f[i], tgt_f[i])
                    n_train += 1

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
                if self.dashboard is not None:
                    self.dashboard.log_train_metrics(row)
                print(f"  [epoch {epoch:3d} | step {step:4d}] "
                      f"loss={breakdown['total']:.4f}  "
                      f"l1_m={breakdown['l1_masked']:.4f}  "
                      f"perc={breakdown['perceptual']:.4f}  "
                      f"sat={breakdown['saturation']:.4f}  "
                      f"cm={breakdown['color_moment']:.4f}  "
                      f"border={breakdown['border']:.4f}  "
                      f"ecoh={breakdown['edge_coherence']:.4f}  "
                      f"btv={breakdown['bg_tv']:.4f}  "
                      f"bd={breakdown['bg_delta']:.4f}  "
                      f"grad={grad_norm:.3f}{'*' if clipped else ''}  "
                      f"lr={lr:.2e}")

            if is_step and self._pause_requested():
                self._checkpoint_and_pause(epoch)

        if self.use_amp and skipped_steps > 0:
            scale_after = self.scaler.get_scale()
            print(f"  [AMP] skipped {skipped_steps}/{step} optimizer steps due to inf/nan gradients  "
                  f"(scale {scale_before:.0f} в†’ {scale_after:.0f})")

        return total_loss / len(self.train_loader), total_train_psnr / max(n_train, 1)

    #в”Ђв”Ђ validation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    @torch.no_grad()
    def _validate(self, epoch: int, train_psnr: float | None = None) -> float:
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
        val_row = {"epoch": epoch, "global_step": self._global_step,
                   "psnr": f"{avg_psnr:.4f}",
                   "psnr_masked": f"{avg_psnr_masked:.4f}"}
        if train_psnr is not None:
            val_row["train_psnr"] = f"{train_psnr:.4f}"
        self.val_logger.log(val_row)
        print(f"  val psnr={avg_psnr:.2f} dB  psnr_masked={avg_psnr_masked:.2f} dB"
              + (f"  train_psnr={train_psnr:.2f} dB" if train_psnr is not None else "")
              + ("  [EMA]" if self.ema else ""))
        if self.dashboard is not None:
            self.dashboard.log_val_metrics(val_row)
        if self._sample_dir is not None:
            result = self._infer_sample()
            if result is not None:
                if self.experiment is not None:
                    blended_bgr, pred_roi_bgr, wloss = result
                    self.experiment.log_preview_set(
                        epoch,
                        {
                            "blended_output": blended_bgr,
                            "raw_model_output": pred_roi_bgr,
                            "weighted_loss": wloss,
                        },
                    )
        return avg_psnr_masked

    # в”Ђв”Ђ visualisation sample inference в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    @torch.no_grad()
    def _infer_sample(self, size: int | None = None):
        """Run inference on the fixed visualisation sample.

        Matches the infer.py preprocessing pipeline exactly so the three
        returned images correspond to the blended clean prediction at source
        resolution, the raw model ROI output at configured resolution, and the
        ROI weighted loss heatmap.
        Returns (blended_bgr, pred_roi_bgr, wloss_f32) or None on any failure.
        """
        if size is None:
            size = (self.image_width, self.image_height)

        model_width, model_height = size
        wm    = cv2.imread(str(self._sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask  = cv2.imread(str(self._sample_dir / "mask.png"),        cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (wm, mask)):
            return None

        ds_cfg = self.cfg["dataset"]
        store_root = get_store_root(ds_cfg["root"], ds_cfg.get("preprocessed_store_dir"))
        store_signature = build_store_signature(
            image_width=model_width,
            image_height=model_height,
            crop_aspect_ratio=ds_cfg.get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=ds_cfg.get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=ds_cfg.get("crop_min_width_ratio", 0.50),
            use_augmented_mask=False,
        )
        wm_r, clean_r, _, mask_input = load_or_build_preprocessed_sample(
            store_root=store_root,
            dataset_root=ds_cfg["root"],
            sample_dir=self._sample_dir,
            signature=store_signature,
            image_width=model_width,
            image_height=model_height,
            crop_aspect_ratio=ds_cfg.get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=ds_cfg.get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=ds_cfg.get("crop_min_width_ratio", 0.50),
            use_augmented_mask=False,
        )

        # Use the exact cached tensors that training uses so the preview heatmap
        # reflects model progress rather than differences between two
        # preprocessing pipelines.
        rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        mask_input_t = torch.from_numpy(mask_input.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        grad_t = compute_gradient(wm_r).unsqueeze(0)
        inp = torch.cat([rgb_t, mask_input_t, grad_t], dim=1)

        _, _, _, _, mask_binary, roi = prepare_roi_input(
            wm,
            mask,
            model_width,
            model_height,
            dilate=False,
            crop_aspect_ratio=self.cfg["dataset"].get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=self.cfg["dataset"].get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=self.cfg["dataset"].get("crop_min_width_ratio", 0.50),
        )

        was_training = self.model.training
        self.model.eval()
        pred_roi_bgr, _, delta = run_model(self.model, inp, self.device)
        if was_training:
            self.model.train()

        blend_mask = (dilate_mask_input(mask.astype(np.float32) / 255.0, image_size=max(model_width, model_height)) * 255.0).astype(np.uint8)
        pred_bgr = blend_back(pred_roi_bgr, wm, blend_mask, roi, feather=9, mask_expand=0)

        delta_np = delta.squeeze(0).permute(1, 2, 0).cpu().numpy()
        wloss = weighted_loss_map(
            pred_roi_bgr,
            clean_r,
            mask_binary,
            self.cfg,
            (model_width, model_height),
            delta=delta_np,
        )

        return pred_bgr, pred_roi_bgr, wloss

    # в”Ђв”Ђ main loop в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")
        print(f"  train: {len(self.train_loader.dataset)} samples  "
              f"val: {len(self.val_loader.dataset)} samples")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            progress = (epoch - 1) / max(self.epochs - 1, 1)
            self.criterion.set_progress(progress)
            avg_loss, avg_train_psnr = self._train_epoch(epoch)

            if self._pause_requested():
                self._checkpoint_and_pause(epoch)

            if self.scheduler:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Detected call of", category=UserWarning)
                    self.scheduler.step()

            do_val   = (epoch % self.val_every == 0) or (epoch == self.epochs)
            avg_psnr = self._validate(epoch, train_psnr=avg_train_psnr) if do_val else self.best_psnr
            elapsed  = time.time() - t0

            is_best = do_val and (avg_psnr > self.best_psnr)
            if is_best:
                self.best_psnr = avg_psnr
                best_epoch_path = _save_ckpt(self.ckpt_dir, epoch, self.model,
                                             self.optimizer, self.scheduler,
                                             self.best_psnr, self.keep_last, self.ema)
                # best.pth always contains EMA weights (smoother inference)
                best_path = self.ckpt_dir / "best.pth"
                _save_ckpt_to(best_path, epoch, self.model,
                              self.optimizer, self.scheduler, self.best_psnr,
                              self.ema)
                if self.experiment is not None:
                    if best_epoch_path is not None:
                        self.experiment.stage_artifact(best_epoch_path, category="checkpoints")
                    self.experiment.stage_artifact(best_path, category="checkpoints")
                marker = " * best"
            elif epoch % self.save_every == 0:
                saved_path = _save_ckpt(self.ckpt_dir, epoch, self.model,
                                        self.optimizer, self.scheduler,
                                        self.best_psnr, self.keep_last, self.ema)
                if self.experiment is not None and saved_path is not None:
                    self.experiment.stage_artifact(saved_path, category="checkpoints")
                marker = ""
            else:
                marker = ""

            psnr_str = f"{avg_psnr:.2f} dB" if do_val else "  --   "
            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"loss={avg_loss:.4f}  psnr={psnr_str}{marker}  "
                  f"({elapsed:.0f}s)")

        print(f"\nDone. Best PSNR: {self.best_psnr:.2f} dB")

