"""
Trainer for overlay removal.

Each epoch is a fixed number of freshly synthesised samples. Validation uses a
fixed synthetic set (held-out backgrounds), and previews include real stamped
photos from held-out listings, run through the same inference path the service
will use, with a contrast-boosted view to expose leftover edges.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ...common.checkpointing import CSVLogger, _save_ckpt, _save_ckpt_to, load_checkpoint
from ...common.training_control import TrainingPaused
from .features import build_input, to_unit
from .inference import remove_mark
from .losses import OverlayLoss


class EMA:
    """Exponential moving average of the weights, kept on the training device."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.float(), 1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in self.shadow.items()})

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = {k: v.to(next(iter(self.shadow.values())).device) for k, v in sd["shadow"].items()}


def psnr(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample PSNR for [-1, 1] images, optionally over a weighted region."""
    se = (a - b) ** 2
    if weight is None:
        mse = se.mean(dim=(1, 2, 3))
    else:
        mse = (se * weight).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) * a.shape[1]).clamp(min=1)
    return 10 * torch.log10(4.0 / mse.clamp(min=1e-10))


def contrast_view(rgb: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Local contrast boost: amplify deviations from a smoothed copy, to reveal faint leftovers."""
    f = rgb.astype(np.float32)
    base = cv2.GaussianBlur(f, (0, 0), 12)
    return np.clip(base + (f - base) * gain, 0, 255).astype(np.uint8)


class OverlayTrainer:
    def __init__(self, model: nn.Module, cfg: dict, profile, train_loader, val_loader,
                 experiment=None, preview_photos: list[tuple[np.ndarray, str]] | None = None,
                 resume: str | None = None, load_weights: str | None = None):
        self.cfg = cfg
        self.profile = profile
        self.device = torch.device(cfg["device"])
        torch.backends.cudnn.benchmark = True   # fixed input size: let cuDNN pick the fastest kernels
        self.model = model.to(self.device, memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment = experiment
        self.preview_photos = preview_photos or []

        tc = cfg["training"]
        self.epochs = tc["epochs"]
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup = int(tc.get("warmup_steps", 500))
        self.base_lr = float(tc["lr"])
        self.min_lr = float(tc.get("min_lr", self.base_lr * 0.02))
        self.grad_clip = float(tc.get("grad_clip", 1.0))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.base_lr,
                                           weight_decay=float(tc.get("weight_decay", 1e-4)),
                                           betas=(0.9, 0.99))
        self.amp_dtype = torch.bfloat16 if tc.get("amp", True) and self.device.type == "cuda" else None
        self.criterion = OverlayLoss(cfg)
        self.criterion.overlay = profile.overlay   # the ghost term measures the mark against the profile's overlay
        self.ema = EMA(self.model, float(tc.get("ema_decay", 0.999))) if tc.get("ema_decay", 0) > 0 else None
        self.hint_scale = float(cfg["data"].get("hint_scale", 8.0))
        self.teacher, self.w_distill = self._load_teacher(cfg.get("distill"))

        self.ckpt_dir = Path(cfg["checkpointing"]["dir"])
        self.keep_last = int(cfg["checkpointing"].get("keep_last", 3))
        run_dir = Path(cfg["logging"]["dir"])
        self.log_every = int(cfg["logging"].get("log_every", 50))
        self.train_logger = CSVLogger(run_dir / "train.csv")
        self.val_logger = CSVLogger(run_dir / "val.csv")

        self.start_epoch = 1
        self.best = -math.inf
        if resume:
            epoch, self.best = load_checkpoint(resume, self.model, self.optimizer, None, self.ema)
            self.start_epoch = epoch + 1
            print(f"Resumed from {resume} (epoch {epoch}, best {self.best:.2f})")
        elif load_weights:
            ckpt = torch.load(load_weights, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
            if unexpected or any(not k.startswith("refine.") for k in missing):
                raise RuntimeError(f"weights don't fit the model: missing {missing}, unexpected {unexpected}")
            if missing:
                print(f"  new (zero-initialised) layers: {sorted({k.split('.')[0] for k in missing})}")
            if self.ema is not None:   # start the average from the loaded weights, not the init
                self.ema = EMA(self.model, self.ema.decay)
            print(f"Loaded weights from {load_weights}")
        self.global_step = (self.start_epoch - 1) * self.steps_per_epoch

    def _load_teacher(self, dc: dict | None):
        """Optional distillation: a trained (larger) model whose prediction the student also matches.

        The teacher's output is a less noisy target than the JPEG'd clean image,
        whose own compression noise cannot be predicted from the input.
        """
        if not dc:
            return None, 0.0
        from .model import build_model
        teacher = build_model({"model": dc["model"]})
        teacher.load_state_dict(torch.load(dc["checkpoint"], map_location="cpu", weights_only=True)["model"])
        teacher.to(self.device, memory_format=torch.channels_last).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Distilling from {dc['checkpoint']} (weight {dc.get('weight', 1.0)})")
        return teacher, float(dc.get("weight", 1.0))

    # ── schedule ─────────────────────────────────────────────────────────────

    def _lr(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        t = (step - self.warmup) / max(1, self.total_steps - self.warmup)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))

    # ── steps ────────────────────────────────────────────────────────────────

    def _forward(self, batch: dict, model: nn.Module):
        wm_u8 = batch["wm"].to(self.device, non_blocking=True)
        hint = batch["hint"].to(self.device, non_blocking=True)
        x = build_input(wm_u8, hint, self.profile.overlay, self.hint_scale)
        x = x.contiguous(memory_format=torch.channels_last)
        wm = x[:, :3]
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            delta = model(x)
        pred = (wm - delta.float()).clamp(-1, 1)
        return pred, wm

    def _targets(self, batch: dict):
        target = to_unit(batch["target"].to(self.device, non_blocking=True))
        region = batch["region"].to(self.device, non_blocking=True).float()
        m_true = batch["m_true"].to(self.device, non_blocking=True).float()
        return target, region, m_true

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        t0 = time.time()
        for step, batch in enumerate(self.train_loader, start=1):
            lr = self._lr(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            target, region, m_true = self._targets(batch)
            pred, _ = self._forward(batch, self.model)
            loss, parts = self.criterion(pred, target, region, m_true)
            if self.teacher is not None:
                with torch.no_grad():
                    t_pred, _ = self._forward(batch, self.teacher)
                weight = 1.0 + self.criterion.region_weight * region
                d = pred - t_pred
                distill = (torch.sqrt(d * d + 1e-6) * weight).sum() / (weight.sum() * d.shape[1])
                loss = loss + self.w_distill * distill
                parts["distill"] = distill.detach()
                parts["total"] = loss.detach()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.global_step += 1

            if step % self.log_every == 0 or step == self.steps_per_epoch:
                row = {"epoch": epoch, "step": step, "global_step": self.global_step,
                       "lr": f"{lr:.2e}", "grad_norm": f"{grad_norm:.4f}",
                       **{k: f"{float(v):.6f}" for k, v in parts.items()}}
                self.train_logger.log(row)
                if self.experiment is not None:
                    self.experiment.heartbeat(epoch=epoch, step=step, global_step=self.global_step)
                rate = step * self.train_loader.batch_size / (time.time() - t0)
                print(f"  [epoch {epoch:3d} | step {step:5d}/{self.steps_per_epoch}] "
                      f"loss={float(parts['total']):.5f} pixel={float(parts['pixel']):.5f} "
                      f"grad={float(parts['grad']):.5f} gn={grad_norm:.3f} lr={lr:.2e} ({rate:.0f} samples/s)")

    @torch.no_grad()
    def _eval_model(self) -> nn.Module:
        if self.ema is None:
            return self.model
        if not hasattr(self, "_ema_model"):
            import copy
            self._ema_model = copy.deepcopy(self.model)
        self.ema.copy_to(self._ema_model)
        return self._ema_model.eval()

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        model = self._eval_model()
        model.eval()
        sums = {"psnr": 0.0, "psnr_masked": 0.0, "psnr_input_masked": 0.0, "mae_mark": 0.0,
                "loc_err": 0.0, "style_ok": 0.0, "ghost": 0.0}
        p999 = []
        n = 0
        first = None
        for batch in self.val_loader:
            target, region, m_true = self._targets(batch)
            pred, wm = self._forward(batch, model)
            sums["ghost"] += float(OverlayLoss.ghost_amplitude(pred, target, m_true, self.profile.overlay).abs().sum())
            b = target.shape[0]
            sums["psnr"] += float(psnr(pred, target).sum())
            sums["psnr_masked"] += float(psnr(pred, target, region).sum())
            sums["psnr_input_masked"] += float(psnr(wm, target, region).sum())
            err = (pred - target).abs() * 127.5
            sums["mae_mark"] += float(((err * region).sum(dim=(1, 2, 3)) /
                                       (region.sum(dim=(1, 2, 3)) * 3).clamp(min=1)).sum())
            sel = err.permute(1, 0, 2, 3)[:, region[:, 0] > 0]
            if sel.numel():
                p999.append(torch.quantile(sel.flatten()[:2_000_000].float(), 0.999).item())
            sums["loc_err"] += float(batch["loc_err"].abs().sum(dim=1).sum())
            sums["style_ok"] += float(batch["style_ok"].float().sum())
            n += b
            if first is None:
                first = (wm[:4].cpu(), pred[:4].cpu(), target[:4].cpu())
        row = {k: v / max(n, 1) for k, v in sums.items()}
        row["p999_mark"] = float(np.mean(p999)) if p999 else 0.0
        self.val_logger.log({"epoch": epoch, "global_step": self.global_step,
                             **{k: f"{v:.4f}" for k, v in row.items()}})
        print(f"  val psnr={row['psnr']:.2f} masked={row['psnr_masked']:.2f} "
              f"(input {row['psnr_input_masked']:.2f}) mae_mark={row['mae_mark']:.3f} "
              f"p99.9={row['p999_mark']:.2f} levels, ghost={row['ghost']:.4f}, loc_err={row['loc_err']:.2f}px")
        if self.experiment is not None:
            self.experiment.log_preview_set(epoch, {**self._synthetic_preview(first),
                                                    **self._real_previews(model)})
        return row["psnr_masked"]

    # ── previews ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bgr(t: torch.Tensor) -> np.ndarray:
        return cv2.cvtColor(((t.numpy().transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)

    def _synthetic_preview(self, first) -> dict:
        if first is None:
            return {}
        wm, pred, target = first
        rows = []
        for i in range(wm.shape[0]):
            a, p, t = self._to_bgr(wm[i]), self._to_bgr(pred[i]), self._to_bgr(target[i])
            diff = np.clip(np.abs(p.astype(np.int16) - t.astype(np.int16)) * 16, 0, 255).astype(np.uint8)
            rows.append(np.concatenate([a, p, t, contrast_view(p), diff], axis=1))
        return {"synthetic": np.concatenate(rows, axis=0)}

    def _real_previews(self, model: nn.Module) -> dict:
        rows = []
        for rgb, listing_id in self.preview_photos:
            out, info = remove_mark(rgb, listing_id, model, self.profile, self.device, self.hint_scale)
            x0, y0 = info["window"]
            ww, wh = self.profile.window
            before = cv2.cvtColor(rgb[y0:y0 + wh, x0:x0 + ww], cv2.COLOR_RGB2BGR)
            after = cv2.cvtColor(out[y0:y0 + wh, x0:x0 + ww], cv2.COLOR_RGB2BGR)
            rows.append(np.concatenate([before, after, contrast_view(before), contrast_view(after)], axis=1))
        return {"real": np.concatenate(rows, axis=0)} if rows else {}

    # ── loop ─────────────────────────────────────────────────────────────────

    def _save(self, epoch: int, score: float) -> None:
        path = _save_ckpt(self.ckpt_dir, epoch, self.model, self.optimizer, None, self.best,
                          self.keep_last, self.ema)
        if self.experiment is not None and path is not None:
            self.experiment.stage_artifact(path, category="checkpoints")
        if score > self.best:
            self.best = score
            best_path = self.ckpt_dir / "best.pth"
            _save_ckpt_to(best_path, epoch, self.model, self.optimizer, None, self.best, self.ema)
            if self.experiment is not None:
                self.experiment.stage_artifact(best_path, category="checkpoints")
            print(f"  * new best masked PSNR {score:.2f} dB")

    def train(self) -> None:
        val_every = int(self.cfg["training"].get("val_every", 1))
        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                t0 = time.time()
                self._train_epoch(epoch)
                score = self._validate(epoch) if (epoch % val_every == 0 or epoch == self.epochs) else -math.inf
                self._save(epoch, score)
                print(f"epoch {epoch} done in {time.time() - t0:.0f}s")
        except KeyboardInterrupt:
            path = _save_ckpt(self.ckpt_dir, epoch, self.model, self.optimizer, None, self.best,
                              self.keep_last, self.ema)
            print(f"\nInterrupted: saved {path}")
            raise TrainingPaused(f"Interrupted at epoch {epoch}")
