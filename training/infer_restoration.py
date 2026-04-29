"""
Run the direct clean-image restoration model on a single image + mask.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import load_seg_model, predict_mask, save_debug_frames
from src.tasks.restoration.inference import blend_back, prepare_roi_input, run_model
from src.tasks.restoration.losses import CombinedLoss
from src.tasks.restoration.model import build_model
from src.tasks.restoration.store_cli import load_restoration_cfg
from wm_shared.config import load_yaml_config


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def compute_debug_loss(cfg: dict,
                       pred_float: torch.Tensor,
                       mask_binary,
                       clean_path: str,
                       model_width: int,
                       model_height: int,
                       roi,
                       correction: torch.Tensor | None = None) -> dict | None:
    clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if clean is None:
        print(f"[debug] Cannot read clean image: {clean_path}")
        return None

    from wm_shared.preprocess import crop_by_roi, blur_mask_for_loss

    clean_r = crop_by_roi(clean, roi, model_width, model_height, is_mask=False)
    rgb = cv2.cvtColor(clean_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    target = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    lc = cfg["loss"]
    mask_loss = blur_mask_for_loss(mask_binary, lc.get("loss_mask_blur_pct", 0.0), (model_width, model_height))
    mask_t = torch.from_numpy(mask_loss).unsqueeze(0).unsqueeze(0)

    criterion = CombinedLoss(cfg)
    criterion.set_progress(1.0)
    with torch.no_grad():
        _, breakdown = criterion(pred_float, target, mask_t, correction=correction)
    return breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--watermarked", required=True)
    parser.add_argument("--mask", default=None,
                        help="path to binary mask PNG; omit to auto-generate via --seg-checkpoint")
    parser.add_argument("--seg-checkpoint", default=None,
                        help="segmentation model checkpoint for automatic mask prediction")
    parser.add_argument("--seg-config", default="configs/seg.yaml")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--config", default="configs/train_restoration_512.yaml")
    parser.add_argument("--clean", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--mask-expand", type=int, default=0)
    args = parser.parse_args()

    cfg = load_restoration_cfg(args.config)
    crop_aspect_ratio = cfg["dataset"].get("crop_aspect_ratio", 3.54)
    crop_margin_ratio = cfg["dataset"].get("crop_margin_ratio", 0.10)
    crop_min_width_ratio = cfg["dataset"].get("crop_min_width_ratio", 0.50)
    model_width = args.width or cfg["dataset"]["image_width"]
    model_height = args.height or cfg["dataset"]["image_height"]

    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = load_model(cfg, args.checkpoint, device)

    if args.mask is None and args.seg_checkpoint is None:
        parser.error("provide either --mask or --seg-checkpoint")

    orig_wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
    if orig_wm is None:
        raise FileNotFoundError(f"Cannot read {args.watermarked}")

    if args.mask is not None:
        orig_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if orig_mask is None:
            raise FileNotFoundError(f"Cannot read {args.mask}")
    else:
        seg_cfg = load_yaml_config(Path(args.seg_config).resolve())
        seg_model = load_seg_model(seg_cfg, args.seg_checkpoint, device)
        seg_size = seg_cfg["dataset"]["image_size"]
        orig_mask = predict_mask(seg_model, orig_wm, seg_size, device)
        print(f"Mask predicted by segmentation model (seg size={seg_size})")

    inp, wm_r, mask_r, grad_np, mask_binary, roi = prepare_roi_input(
        orig_wm,
        orig_mask,
        model_width,
        model_height,
        dilate=args.debug,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )

    pred_bgr, pred_float, correction = run_model(model, inp, device)

    loss_breakdown = None
    if args.debug and args.clean:
        loss_breakdown = compute_debug_loss(
            cfg,
            pred_float,
            mask_binary,
            args.clean,
            model_width,
            model_height,
            roi,
            correction=correction,
        )
        save_debug_frames(
            args.output,
            wm_r,
            mask_r,
            grad_np,
            pred_bgr,
            args.clean,
            loss_breakdown,
            mask_binary=mask_binary,
            cfg=cfg,
            roi=roi,
        )

    result = blend_back(
        pred_bgr,
        orig_wm,
        orig_mask,
        roi,
        feather=args.feather,
        mask_expand=args.mask_expand,
    )

    cv2.imwrite(args.output, result)
    print(f"Saved -> {args.output}  (original resolution: {orig_wm.shape[1]}x{orig_wm.shape[0]})")


if __name__ == "__main__":
    main()
