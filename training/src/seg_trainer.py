"""
SegTrainer
----------
Training loop for the watermark segmentation model.

Loss    : BCE + Focal + L1 + Dice  (soft targets in [0, 1])
          BCE provides a stable baseline; Focal concentrates gradient on hard
          pixels (boundaries, subtle marks); L1 gives direct per-pixel delta
          regression; Dice enforces region overlap.
Metric  : IoU at threshold 0.5 on the validation set
"""

import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .trainer import CSVLogger, _save_ckpt, _save_ckpt_to, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# loss
# ──────────────────────────────────────────────────────────────────────────────

def _multiscale_l1(prob: torch.Tensor, target: torch.Tensor,
                   scales: tuple[int, ...] = (4, 8, 16)) -> torch.Tensor:
    """
    Average-pool pred and target at several coarse scales, compute L1 at each.

    An outline-only prediction averages to near-zero when pooled (the filled
    interior is missing), while a solid mask stays high.  This directly
    penalises hollow predictions and encourages the model to fill interiors.
    """
    loss = torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
    for k in scales:
        p_down = F.avg_pool2d(prob,   k, k)
        t_down = F.avg_pool2d(target, k, k)
        loss = loss + F.l1_loss(p_down, t_down)
    return loss / len(scales)


def seg_loss(logits: torch.Tensor, target: torch.Tensor,
             pos_weight: float = 3.0, focal_gamma: float = 2.0,
             w_bce: float = 1.0, w_focal: float = 1.0,
             w_l1: float = 1.0, w_dice: float = 1.0,
             w_ms: float = 1.0):
    """
    w_bce*BCE + w_focal*Focal + w_l1*L1 + w_dice*Dice + w_ms*MS on soft targets.

    MS (multi-scale) pools pred & target at 4×/8×/16× and computes L1.
    Outline-only predictions collapse at coarse scales; filled masks don't.
    Each term can be scaled (or disabled with weight=0) via config.
    """
    prob  = torch.sigmoid(logits)

    # --- BCE with pos_weight ---
    pw  = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

    # --- focal loss (numerically stable, computed from logits) ---
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
    )
    weight = target * (pw - 1.0) + 1.0          # pos_weight for fg, 1 for bg
    bce_per_pixel = bce_per_pixel * weight
    p_t   = prob * target + (1.0 - prob) * (1.0 - target)  # prob of correct class
    focal = ((1.0 - p_t) ** focal_gamma * bce_per_pixel).mean()

    # --- per-pixel L1 (delta) loss ---
    l1 = F.l1_loss(prob, target)

    # --- multi-scale L1 (fill interiors) ---
    ms = _multiscale_l1(prob, target)

    # --- dice loss ---
    p     = prob.view(-1)
    t     = target.view(-1)
    intersection = (p * t).sum()
    dice  = 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)

    total = w_bce * bce + w_focal * focal + w_l1 * l1 + w_ms * ms + w_dice * dice
    return total, {"total": total.item(),
                   "bce": (w_bce * bce).item(),
                   "focal": (w_focal * focal).item(),
                   "l1": (w_l1 * l1).item(),
                   "ms": (w_ms * ms).item(),
                   "dice": (w_dice * dice).item()}


# ──────────────────────────────────────────────────────────────────────────────
# metric
# ──────────────────────────────────────────────────────────────────────────────

def iou(logits: torch.Tensor, target_bin: torch.Tensor) -> float:
    pred_bin = (logits > 0).float()  # equivalent to sigmoid > 0.5, no extra op
    intersection = (pred_bin * target_bin).sum().item()
    union        = ((pred_bin + target_bin) > 0).float().sum().item()
    return intersection / max(union, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# trainer
# ──────────────────────────────────────────────────────────────────────────────

class SegTrainer:
    def __init__(self, model: nn.Module, cfg: dict,
                 train_loader: DataLoader, val_loader: DataLoader,
                 resume: str | None = None,
                 load_weights: str | None = None,
                 plotter=None):

        self.model        = model
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = torch.device(cfg["device"])
        self.model.to(self.device)

        tc = cfg["training"]
        use_amp = tc.get("amp", False) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if use_amp else torch.float32
        self.use_amp   = use_amp
        if use_amp:
            print("AMP (bfloat16 mixed precision) enabled")

        decoder_lr  = tc["lr"]
        encoder_lr  = tc.get("encoder_lr", decoder_lr)
        has_encoder = hasattr(model, "encoder")
        if has_encoder and encoder_lr != decoder_lr:
            encoder_params  = list(model.encoder.parameters())
            encoder_ids     = {id(p) for p in encoder_params}
            decoder_params  = [p for p in model.parameters() if id(p) not in encoder_ids]
            param_groups    = [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": decoder_params, "lr": decoder_lr},
            ]
            print(f"Differential LR: encoder={encoder_lr:.1e}  decoder={decoder_lr:.1e}")
        else:
            param_groups = model.parameters()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=decoder_lr,
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
        elif load_weights:
            ckpt = torch.load(load_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded weights from {load_weights} (starting fresh from epoch 1)")

        self.plotter      = plotter
        self._global_step = 0
        self._sample_dir  = None  # set from train_seg.py after construction

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

        tc = self.cfg["training"]
        pos_weight  = tc.get("pos_weight", 3.0)
        focal_gamma = tc.get("focal_gamma", 2.0)
        lw = tc.get("loss_weights", {})
        w_bce   = lw.get("bce",   1.0)
        w_focal = lw.get("focal", 1.0)
        w_l1    = lw.get("l1",    1.0)
        w_ms    = lw.get("ms",    1.0)
        w_dice  = lw.get("dice",  1.0)

        for batch in self.train_loader:
            inp    = batch["input" ].to(self.device)          # Bx3xHxW
            target = batch["target"].to(self.device)          # Bx1xHxW soft [0,1]

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                logits = self.model(inp)                       # Bx1xHxW logits
            loss, breakdown = seg_loss(logits.float(), target, pos_weight, focal_gamma,
                                       w_bce, w_focal, w_l1, w_dice, w_ms)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip if self.grad_clip > 0 else float("inf"),
            ).item()
            self.optimizer.step()

            total_loss += loss.item()
            step += 1
            self._global_step += 1

            if step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                clipped = grad_norm > self.grad_clip > 0
                row = {"epoch": epoch, "step": step, "lr": f"{lr:.2e}",
                       "grad_norm": f"{grad_norm:.4f}", **breakdown}
                self.train_logger.log(row)
                if self.plotter:
                    self.plotter.log_step(self._global_step, breakdown)
                print(f"  [epoch {epoch:3d} | step {step:4d}] "
                      f"loss={breakdown['total']:.4f}  "
                      f"bce={breakdown['bce']:.4f}  "
                      f"focal={breakdown['focal']:.4f}  "
                      f"l1={breakdown['l1']:.4f}  "
                      f"ms={breakdown['ms']:.4f}  "
                      f"dice={breakdown['dice']:.4f}  "
                      f"grad={grad_norm:.3f}{'*' if clipped else ''}  "
                      f"lr={lr:.2e}")

            preview_every = self.cfg["logging"].get("preview_every", 0)
            if (preview_every > 0 and self.plotter and self._sample_dir is not None
                    and self._global_step % preview_every == 0):
                result = self._infer_sample()
                if result is not None:
                    self.plotter.log_images(*result)

        return total_loss / max(step, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_iou = 0.0
        n = 0

        for batch in self.val_loader:
            inp    = batch["input" ].to(self.device)
            target = batch["target"].to(self.device)
            target_bin = (target > 0.5).float()          # binarise for IoU metric only

            logits = self.model(inp)

            for i in range(logits.shape[0]):
                total_iou += iou(logits[i], target_bin[i])
                n += 1

        avg_iou = total_iou / max(n, 1)
        self.val_logger.log({"epoch": epoch, "iou": f"{avg_iou:.4f}"})
        print(f"  val iou={avg_iou:.4f}")
        if self.plotter:
            self.plotter.log_val(self._global_step, avg_iou)
            if self._sample_dir is not None:
                result = self._infer_sample()
                if result is not None:
                    self.plotter.log_images(*result)
        return avg_iou

    # ── visualisation sample inference ────────────────────────────────────

    @torch.no_grad()
    def _infer_sample(self, size: int = 0):
        """Run inference on the fixed visualisation sample.
        Returns (wm_bgr, pred_mask_f32, gt_mask_f32) or None on failure.
        """
        import cv2
        import numpy as np

        wm   = cv2.imread(str(self._sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self._sample_dir / "mask.png"),        cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (wm, mask)):
            return None

        if size == 0:
            size = self.cfg["dataset"]["image_size"]
        wm_r   = cv2.resize(wm,   (size, size), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)

        _MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_np = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_np = (rgb_np - _MEAN) / _STD
        inp_t  = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)  # 1x3xHxW

        was_training = self.model.training
        self.model.eval()
        logits_t = self.model(inp_t)                                    # 1x1xHxW logits
        if was_training:
            self.model.train()

        pred_np = torch.sigmoid(logits_t).squeeze().cpu().float().numpy()  # HxW in (0,1)
        gt_np   = mask_r.astype(np.float32) / 255.0                        # HxW in [0,1]

        return wm_r, pred_np, gt_np

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")
        print(f"  train: {len(self.train_loader.dataset)} samples  "
              f"val: {len(self.val_loader.dataset)} samples")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            avg_loss = self._train_epoch(epoch)

            if self.scheduler:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Detected call of", category=UserWarning)
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
                best_path = self.ckpt_dir / "best.pth"
                _save_ckpt_to(best_path, epoch, self.model,
                              self.optimizer, self.scheduler, self.best_iou)
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
