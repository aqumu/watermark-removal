"""
Loss for overlay removal: everything is measured in the window's own pixel
grid against the JPEG'd clean target.

  pixel : Charbonnier over the window, with the macroblocks the mark touches
          weighted up (outside them the target equals the input exactly).
  grad  : L1 between image gradients of prediction and target inside the mark
          region. Leftover mark edges show up as gradient errors long before
          they move the mean pixel error, and edges are what a contrast boost
          reveals.
  ghost : |least-squares amplitude| of the residual along the true mark
          signal m * (overlay - clean), per sample (smoothed at 0, so the pull
          towards zero does not fade as the leftover gets small). A faint leftover mark is
          below the target's own JPEG noise per pixel but coherent over the
          whole mark; this measures exactly that (1 = untouched, 0 = gone).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


class OverlayLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]
        self.region_weight = float(lc.get("region_weight", 10.0))
        self.w_pixel = float(lc.get("pixel", 1.0))
        self.w_grad = float(lc.get("grad", 1.0))
        self.w_ghost = float(lc.get("ghost", 0.0))
        self.overlay = 234.0   # set from the mark profile by the trainer
        self.eps = 1e-3

    @staticmethod
    def _ghost_terms(pred, target, m_true, overlay, sig=None):
        """sig: the mark's own luma signal (Bx1xHxW, /127.5) when it is not m * (overlay - clean)."""
        luma = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        e = ((pred - target) * luma).sum(1, keepdim=True)
        if sig is None:
            clean = ((target + 1) * 127.5 * luma).sum(1, keepdim=True)
            sig = m_true * (overlay - clean) / 127.5
        return (e * sig).sum(dim=(1, 2, 3)), (sig * sig).sum(dim=(1, 2, 3))

    @classmethod
    def ghost_amplitude(cls, pred: torch.Tensor, target: torch.Tensor, m_true: torch.Tensor,
                        overlay: float, sig: torch.Tensor | None = None) -> torch.Tensor:
        """Per-sample amplitude of the mark left in pred (inputs in [-1, 1], m_true Bx1xHxW)."""
        num, den = cls._ghost_terms(pred, target, m_true, overlay, sig)
        return num / den.clamp(min=1e-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor,
                m_true: torch.Tensor | None = None, sig: torch.Tensor | None = None):
        err = pred - target
        weight = 1.0 + self.region_weight * region
        charb = torch.sqrt(err * err + self.eps ** 2)
        pixel = (charb * weight).sum() / (weight.sum() * err.shape[1])

        gx, gy = _grad(err)
        rx, ry = region[..., :, 1:], region[..., 1:, :]
        grad = ((gx.abs() * rx).sum() + (gy.abs() * ry).sum()) / ((rx.sum() + ry.sum()) * err.shape[1] + 1e-6)

        total = self.w_pixel * pixel + self.w_grad * grad
        parts = {"pixel": pixel.detach(), "grad": grad.detach()}
        if self.w_ghost > 0 and m_true is not None:
            # |amplitude|, smoothly down-weighted when the mark signal itself is weak
            # (background close to the overlay level), where the amplitude is noise
            num, den = self._ghost_terms(pred, target, m_true, self.overlay, sig)
            ghost = torch.sqrt(num ** 2 / (den * (den + 10.0)).clamp(min=1e-6) + 1e-6).mean()
            total = total + self.w_ghost * ghost
            parts["ghost"] = ghost.detach()
        return total, {"total": total.detach(), **parts}
