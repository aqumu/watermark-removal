"""
Loss for overlay removal: everything is measured in the window's own pixel
grid against the JPEG'd clean target.

  pixel : Charbonnier over the window, with the macroblocks the mark touches
          weighted up (outside them the target equals the input exactly).
  grad  : L1 between image gradients of prediction and target inside the mark
          region. Leftover mark edges show up as gradient errors long before
          they move the mean pixel error, and edges are what a contrast boost
          reveals.
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
        self.eps = 1e-3

    def forward(self, pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor):
        err = pred - target
        weight = 1.0 + self.region_weight * region
        charb = torch.sqrt(err * err + self.eps ** 2)
        pixel = (charb * weight).sum() / (weight.sum() * err.shape[1])

        gx, gy = _grad(err)
        rx, ry = region[..., :, 1:], region[..., 1:, :]
        grad = ((gx.abs() * rx).sum() + (gy.abs() * ry).sum()) / ((rx.sum() + ry.sum()) * err.shape[1] + 1e-6)

        total = self.w_pixel * pixel + self.w_grad * grad
        return total, {"total": total.detach(), "pixel": pixel.detach(), "grad": grad.detach()}
