"""
Loss functions for direct clean-image restoration.
-----------------------------------------------
This is a restoration-specific copy of the removal loss stack so the task can
evolve independently.

CombinedLoss = λ_l1_masked        × L1(pred ⊙ mask_interior, target ⊙ mask_interior)
             + λ_perceptual       × Perceptual(pred, target)   [VGG16 4-level features]
             + λ_border           × BorderRing(pred, target, soft_mask)
             + λ_color_moment     × ColorMoment(pred, target, mask_interior)
             + λ_bg_tv            × BackgroundTV(correction, mask)
             + λ_bg_delta         × BackgroundDelta(correction, mask)
             + λ_saturation       × SaturationLoss(pred, target, mask_interior)
             + λ_edge_coherence   × EdgeCoherence(pred, target, soft_mask)

All tensors are in [-1, 1] (normalised RGB).
The perceptual loss remaps to [0, 1] internally before passing to VGG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class PerceptualLoss(nn.Module):
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]
    _LEVEL_WEIGHTS = (0.5, 1.0, 1.0, 0.05)

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        feats = vgg.features
        self.slice0 = nn.Sequential(*list(feats[0:4]))
        self.slice1 = nn.Sequential(*list(feats[4:9]))
        self.slice2 = nn.Sequential(*list(feats[9:16]))
        self.slice3 = nn.Sequential(*list(feats[16:23]))

        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    _VGG_SIZE = 224

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        if x.shape[-1] != self._VGG_SIZE:
            x = F.interpolate(x, size=self._VGG_SIZE, mode="area")
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._normalise(pred)
        t = self._normalise(target)
        loss = pred.new_zeros(1).squeeze()
        for sl, w in zip((self.slice0, self.slice1, self.slice2, self.slice3), self._LEVEL_WEIGHTS):
            p = sl(p)
            t = sl(t)
            loss = loss + w * F.mse_loss(p, t)
        return loss


def color_moment_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum(dim=(2, 3)).clamp(min=1)
    pred_mean = (pred * mask).sum(dim=(2, 3)) / n
    target_mean = (target * mask).sum(dim=(2, 3)) / n
    return F.l1_loss(pred_mean, target_mean)


def _rgb_to_sv(rgb01: torch.Tensor):
    max_c, _ = rgb01.max(dim=1, keepdim=True)
    min_c, _ = rgb01.min(dim=1, keepdim=True)
    diff = max_c - min_c
    denom = max_c.detach().clamp(min=0.05)
    s = torch.where(max_c > 0.05, diff / denom, torch.zeros_like(max_c))
    return s, max_c


class SaturationLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p01 = (pred + 1.0) * 0.5
        t01 = (target + 1.0) * 0.5
        p_s, p_v = _rgb_to_sv(p01)
        t_s, t_v = _rgb_to_sv(t01)
        n = mask.sum().clamp(min=1e-6)
        sat_loss = ((p_s - t_s).abs() * mask).sum() / n
        val_loss = ((p_v - t_v).abs() * mask).sum() / n
        return sat_loss + 0.5 * val_loss


def border_ring_loss(pred: torch.Tensor, target: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
    weight = 4.0 * soft_mask * (1.0 - soft_mask)
    n = weight.sum().clamp(min=1e-6) * pred.shape[1]
    return ((pred - target).abs() * weight).sum() / n


def background_tv_loss(correction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bg = 1.0 - mask
    d = correction * bg
    tv_h = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean()
    tv_w = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def background_delta_penalty(correction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bg = 1.0 - mask
    return (correction * bg).abs().mean()


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, channels: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    if channels is not None and value.shape[1] != weight.shape[1]:
        weight = weight.expand(-1, channels, -1, -1)
    denom = weight.sum().clamp(min=eps)
    return (value * weight).sum() / denom


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def edge_coherence_loss(pred: torch.Tensor, target: torch.Tensor, soft_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weight = 4.0 * soft_mask * (1.0 - soft_mask)
    residual_gray = _rgb_to_gray(pred - target)
    rx, ry = _sobel_xy(residual_gray)
    j11 = F.avg_pool2d(rx * rx, kernel_size=3, stride=1, padding=1)
    j22 = F.avg_pool2d(ry * ry, kernel_size=3, stride=1, padding=1)
    j12 = F.avg_pool2d(rx * ry, kernel_size=3, stride=1, padding=1)
    trace = j11 + j22
    delta = torch.sqrt((j11 - j22) ** 2 + 4.0 * (j12 ** 2) + eps)
    coherence = delta / (trace + eps)
    grad_energy = torch.sqrt(rx * rx + ry * ry + eps)
    return _weighted_mean(coherence * grad_energy, weight)


def _parse_weight(v) -> tuple[float, float]:
    if isinstance(v, (list, tuple)):
        return (float(v[0]), float(v[1]))
    return (float(v), float(v))


def _lerp_weight(w: tuple[float, float], progress: float) -> float:
    return w[0] + (w[1] - w[0]) * progress


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]
        self._w_l1_masked = _parse_weight(lc["l1_masked"])
        self._w_perceptual = _parse_weight(lc["perceptual"])
        self._w_color_moment = _parse_weight(lc.get("color_moment", 0.0))
        self._w_border = _parse_weight(lc.get("border", 0.0))
        self._w_bg_tv = _parse_weight(lc.get("bg_tv", 0.0))
        self._w_bg_delta = _parse_weight(lc.get("bg_delta", 0.0))
        self._w_saturation = _parse_weight(lc.get("saturation", 0.0))
        self._w_edge_coherence = _parse_weight(lc.get("edge_coherence", 0.0))
        self._erosion_kernel = lc.get("erosion_kernel", 3)
        self._progress = 0.0
        self._update_weights()
        self.perceptual = PerceptualLoss() if max(self._w_perceptual) > 0 else None
        self.saturation = SaturationLoss() if max(self._w_saturation) > 0 else None

    def _update_weights(self):
        t = self._progress
        self.w_l1_masked = _lerp_weight(self._w_l1_masked, t)
        self.w_perceptual = _lerp_weight(self._w_perceptual, t)
        self.w_color_moment = _lerp_weight(self._w_color_moment, t)
        self.w_border = _lerp_weight(self._w_border, t)
        self.w_bg_tv = _lerp_weight(self._w_bg_tv, t)
        self.w_bg_delta = _lerp_weight(self._w_bg_delta, t)
        self.w_saturation = _lerp_weight(self._w_saturation, t)
        self.w_edge_coherence = _lerp_weight(self._w_edge_coherence, t)

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))
        self._update_weights()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, correction: torch.Tensor | None = None, use_perceptual: bool = True) -> tuple[torch.Tensor, dict]:
        need_mask_interior = self.w_l1_masked > 0 or self.w_color_moment > 0 or self.w_saturation > 0
        if need_mask_interior:
            k = self._erosion_kernel
            mask_interior = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)
        else:
            mask_interior = None

        if self.w_l1_masked > 0:
            masked_pred = pred * mask_interior
            masked_target = target * mask_interior
            n_masked = mask_interior.sum().clamp(min=1) * 3
            l1_masked = (masked_pred - masked_target).abs().sum() / n_masked
        else:
            l1_masked = pred.new_zeros(1).squeeze()

        perc_loss = self.perceptual(pred, target) if (self.perceptual is not None and self.w_perceptual > 0 and use_perceptual) else pred.new_zeros(1).squeeze()
        cm_loss = color_moment_loss(pred, target, mask_interior) if self.w_color_moment > 0 else pred.new_zeros(1).squeeze()
        sat_loss = self.saturation(pred, target, mask_interior) if (self.saturation is not None and self.w_saturation > 0) else pred.new_zeros(1).squeeze()
        b_loss = border_ring_loss(pred, target, mask) if self.w_border > 0 else pred.new_zeros(1).squeeze()
        bg_tv_loss = background_tv_loss(correction, mask) if (self.w_bg_tv > 0 and correction is not None) else pred.new_zeros(1).squeeze()
        bg_delta_loss = background_delta_penalty(correction, mask) if (self.w_bg_delta > 0 and correction is not None) else pred.new_zeros(1).squeeze()
        edge_coh = edge_coherence_loss(pred, target, mask) if self.w_edge_coherence > 0 else pred.new_zeros(1).squeeze()

        total = (
            self.w_l1_masked * l1_masked
            + self.w_perceptual * perc_loss
            + self.w_color_moment * cm_loss
            + self.w_saturation * sat_loss
            + self.w_border * b_loss
            + self.w_bg_tv * bg_tv_loss
            + self.w_bg_delta * bg_delta_loss
            + self.w_edge_coherence * edge_coh
        )

        breakdown = {
            "l1_masked": l1_masked.item(),
            "perceptual": perc_loss.item() if self.perceptual else 0.0,
            "color_moment": cm_loss.item(),
            "saturation": sat_loss.item(),
            "border": b_loss.item(),
            "bg_tv": bg_tv_loss.item(),
            "bg_delta": bg_delta_loss.item(),
            "edge_coherence": edge_coh.item(),
            "total": total.item(),
            "weight_l1_masked": self.w_l1_masked,
            "weight_perceptual": self.w_perceptual,
            "weight_color_moment": self.w_color_moment,
            "weight_saturation": self.w_saturation,
            "weight_border": self.w_border,
            "weight_bg_tv": self.w_bg_tv,
            "weight_bg_delta": self.w_bg_delta,
            "weight_edge_coherence": self.w_edge_coherence,
        }
        return total, breakdown
