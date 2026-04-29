import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    """PSNR between two [-1,1] tensors. max_val = 1-(-1) = 2.0."""
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)
