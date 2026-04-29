import copy
import math

import cv2
import numpy as np
import torch

from ...common.restoration import (
    blend_back,
    build_store_signature,
    get_store_root,
    load_or_build_preprocessed_sample,
    prepare_roi_input,
)
from wm_shared.preprocess import compute_gradient, dilate_mask_input
from ..removal.trainer import Trainer as RemovalTrainer
from .inference import run_model, weighted_loss_map
from .losses import CombinedLoss


class Trainer(RemovalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CombinedLoss(self.cfg).to(self.device)

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
                pred_clean = self.model(inp)
                correction = wm - pred_clean
                loss, breakdown = self.criterion(
                    pred_clean,
                    target,
                    mask,
                    correction=correction,
                    use_perceptual=use_perc,
                )

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
                  f"(scale {scale_before:.0f} -> {scale_after:.0f})")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        live_state = None
        if self.ema is not None:
            live_state = copy.deepcopy(self.model.state_dict())
            self.ema.copy_to(self.model)

        self.model.eval()
        total_psnr = 0.0
        total_psnr_masked = 0.0
        n = 0

        for batch in self.val_loader:
            inp, target, mask, wm = self._prepare_batch_gpu(batch, training=False)
            pred_clean = self.model(inp)

            for i in range(pred_clean.shape[0]):
                total_psnr += self._psnr(pred_clean[i], target[i])
                m = (mask[i] > 0.5).float()
                n_px = m.sum().item()
                if n_px > 0:
                    mse_masked = ((pred_clean[i] - target[i]) ** 2 * m).sum().item() / (n_px * 3)
                    total_psnr_masked += (10 * math.log10(4.0 / mse_masked)
                                          if mse_masked > 0 else 100.0)
                n += 1

        if live_state is not None:
            self.model.load_state_dict(live_state)
        self.model.train()

        avg_psnr = total_psnr / max(n, 1)
        avg_psnr_masked = total_psnr_masked / max(n, 1)
        val_row = {"epoch": epoch, "global_step": self._global_step,
                   "psnr": f"{avg_psnr:.4f}",
                   "psnr_masked": f"{avg_psnr_masked:.4f}"}
        self.val_logger.log(val_row)
        print(f"  val psnr={avg_psnr:.2f} dB  psnr_masked={avg_psnr_masked:.2f} dB"
              + ("  [EMA]" if self.ema else ""))
        if self.dashboard is not None:
            self.dashboard.log_val_metrics(val_row)
        if self._sample_dir is not None:
            result = self._infer_sample()
            if result is not None and self.experiment is not None:
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

    def _psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = torch.mean((pred - target) ** 2).item()
        return 10 * math.log10(4.0 / mse) if mse > 0 else 100.0

    @torch.no_grad()
    def _infer_sample(self, size: int | None = None):
        if size is None:
            size = (self.image_width, self.image_height)

        model_width, model_height = size
        wm = cv2.imread(str(self._sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self._sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
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
        pred_roi_bgr, _, correction = run_model(self.model, inp, self.device)
        if was_training:
            self.model.train()

        blend_mask = (dilate_mask_input(mask.astype(np.float32) / 255.0, image_size=max(model_width, model_height)) * 255.0).astype(np.uint8)
        pred_bgr = blend_back(pred_roi_bgr, wm, blend_mask, roi, feather=9, mask_expand=0)

        correction_np = correction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        wloss = weighted_loss_map(
            pred_roi_bgr,
            clean_r,
            mask_binary,
            self.cfg,
            (model_width, model_height),
            delta=correction_np,
        )

        return pred_bgr, pred_roi_bgr, wloss
