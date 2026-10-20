"""
Model input for overlay removal, shared by training and inference.

Built from the stamped window (uint8) and the hint opacity map, on whatever
device the tensors live on, so the training batches stay small (uint8) and
inference uses exactly the same preprocessing.

Channels: stamped RGB, exact inverse for the hint RGB (both scaled to [-1, 1]),
hint opacity * hint_scale.
"""
from __future__ import annotations

import torch


def to_unit(img_u8: torch.Tensor) -> torch.Tensor:
    return img_u8.float() / 127.5 - 1.0


def build_input(wm_u8: torch.Tensor, hint: torch.Tensor, overlay: float, hint_scale: float) -> torch.Tensor:
    """wm_u8: Bx3xHxW uint8, hint: Bx1xHxW opacity. Returns Bx7xHxW float."""
    wm = wm_u8.float()
    m = hint.float()
    inv = ((wm - m * overlay) / (1.0 - m).clamp(min=1e-3)).clamp(-32.0, 287.0)
    return torch.cat([wm / 127.5 - 1.0, inv / 127.5 - 1.0, m * hint_scale], dim=1)
