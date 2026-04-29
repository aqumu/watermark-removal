import cv2
import torch

from ...common.restoration import blend_back, prepare_roi_input
from ..removal.inference import weighted_loss_map


@torch.no_grad()
def run_model(model, inp: torch.Tensor, device: torch.device):
    inp = inp.to(device)
    pred_float = model(inp)
    correction = inp[:, :3] - pred_float
    pred_np = pred_float.squeeze(0).cpu().numpy()
    pred_np = ((pred_np + 1) / 2 * 255).clip(0, 255).astype("uint8")
    pred_bgr = cv2.cvtColor(pred_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return pred_bgr, pred_float.cpu(), correction.cpu()
