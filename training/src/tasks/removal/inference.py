import cv2
import numpy as np
import torch

from ...common.restoration import blend_back, prepare_roi_input
from wm_shared.preprocess import (
    blur_mask_for_loss,
)


@torch.no_grad()
def run_model(model, inp: torch.Tensor, device: torch.device):
    inp = inp.to(device)
    delta = model(inp)
    pred_float = (inp[:, :3] - delta).clamp(-1, 1)
    pred_np = pred_float.squeeze(0).cpu().numpy()
    pred_np = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return pred_bgr, pred_float.cpu(), delta.cpu()


def weighted_loss_map(pred_bgr: np.ndarray,
                      clean_bgr: np.ndarray,
                      mask_binary: np.ndarray,
                      cfg: dict,
                      image_size: tuple[int, int],
                      delta: np.ndarray | None = None) -> np.ndarray:
    lc = cfg["loss"]
    w_l1_masked = _cfg_weight_end(lc.get("l1_masked", 0.0))
    w_bg_delta = _cfg_weight_end(lc.get("bg_delta", 0.0))
    w_border = _cfg_weight_end(lc.get("border", 0.0))
    blur_pct = lc.get("loss_mask_blur_pct", 0.0)

    pred_f = pred_bgr.astype(np.float32) / 127.5 - 1.0
    target_f = clean_bgr.astype(np.float32) / 127.5 - 1.0
    abs_err = np.abs(pred_f - target_f).mean(axis=2)

    soft_mask = blur_mask_for_loss(mask_binary, blur_pct, image_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_interior = cv2.erode(soft_mask, kernel, iterations=1)
    border_w = 4.0 * soft_mask * (1.0 - soft_mask)
    bg_delta_w = 1.0 - soft_mask
    if delta is None:
        bg_signal = abs_err
    else:
        bg_signal = np.abs(delta.astype(np.float32)).mean(axis=2)

    return (
        w_l1_masked * abs_err * mask_interior
        + w_bg_delta * bg_signal * bg_delta_w
        + w_border * abs_err * border_w
    ).astype(np.float32)


def _cfg_weight_end(v) -> float:
    if isinstance(v, (list, tuple)):
        return float(v[-1])
    return float(v)
