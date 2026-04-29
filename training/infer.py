"""
infer.py  - run the trained model on a single image + mask
----------------------------------------------------------
Usage:
  # provide mask manually
  python infer.py \
      --checkpoint artifacts/checkpoints/removal/epoch_0060.pth \
      --watermarked path/to/watermarked.jpg \
      --mask        path/to/mask.png \
      --output      result.png \
      [--width 512 --height 256]  [--config configs/train_512.yaml]

  # Debug mode: show N random dataset samples in a matplotlib grid
  python infer.py \
      --checkpoint artifacts/checkpoints/removal/epoch_0060.pth \
      --debug-grid [--n-samples 6] \
      [--width 512 --height 256] [--config configs/train_512.yaml]
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wm_shared.config import load_yaml_config, validate_removal_config
from src.common.restoration import (
    build_store_signature,
    get_store_root,
    load_or_build_preprocessed_sample,
)
from src.tasks.removal.inference import (
    blend_back,
    prepare_roi_input as shared_prepare_roi_input,
    run_model as shared_run_model,
    weighted_loss_map as shared_weighted_loss_map,
)
from src.tasks.removal.losses import CombinedLoss
from src.tasks.removal.model import build_model
from src.tasks.segmentation.model import build_seg_model
from wm_shared.preprocess import (
    blur_mask_for_loss,
    compute_gradient,
    crop_by_roi,
)


def load_cfg(path: str) -> dict:
    return validate_removal_config(load_yaml_config(path))


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_seg_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_seg_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(seg_model, wm_bgr: np.ndarray, size: int, device: torch.device) -> np.ndarray:
    """
    Run the segmentation model on a BGR image and return a binary uint8 mask
    (0 / 255) at the original image resolution.
    """
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    h, w = wm_bgr.shape[:2]
    wm_r = cv2.resize(wm_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    inp = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pred = torch.sigmoid(seg_model(inp))
    pred_np = pred.squeeze().cpu().numpy()
    binary = (pred_np > 0.5).astype(np.uint8) * 255
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)


def _prepare_roi_input(wm: np.ndarray,
                       mask_u8: np.ndarray,
                       model_width: int,
                       model_height: int,
                       dilate: bool = False,
                       crop_aspect_ratio: float = 3.54,
                       crop_margin_ratio: float = 0.10,
                       crop_min_width_ratio: float = 0.50,
                       use_augmented_mask: bool = False):
    return shared_prepare_roi_input(
        wm,
        mask_u8,
        model_width,
        model_height,
        dilate=dilate,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=use_augmented_mask,
    )


def preprocess(wm_path: str,
               mask_path: str,
               model_width: int,
               model_height: int,
               dilate: bool = False,
               crop_aspect_ratio: float = 3.54,
               crop_margin_ratio: float = 0.10,
               crop_min_width_ratio: float = 0.50):
    wm = cv2.imread(wm_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Cannot read {wm_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")

    orig_size = (wm.shape[1], wm.shape[0])
    inp, wm_r, mask_r, grad_np, mask_binary, roi = _prepare_roi_input(
        wm,
        mask,
        model_width,
        model_height,
        dilate=dilate,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
    )
    return inp, wm, mask, orig_size, wm_r, mask_r, grad_np, mask_binary, roi


@torch.no_grad()
def run(model, inp: torch.Tensor, device: torch.device):
    """Returns (pred_bgr uint8 at model resolution, pred_float Bx3xHxW in [-1,1], delta Bx3xHxW)."""
    return shared_run_model(model, inp, device)


def compute_debug_loss(cfg: dict,
                       pred_float: torch.Tensor,
                       mask_binary: np.ndarray,
                       clean_path: str,
                       model_width: int,
                       model_height: int,
                       roi: dict,
                       delta: torch.Tensor | None = None) -> dict | None:
    """
    Run CombinedLoss identically to the trainer on the ROI crop.
    """
    clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if clean is None:
        print(f"[debug] Cannot read clean image: {clean_path}")
        return None

    clean_r = crop_by_roi(clean, roi, model_width, model_height, is_mask=False)

    rgb = cv2.cvtColor(clean_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    target = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    lc = cfg["loss"]
    mask_loss = blur_mask_for_loss(mask_binary, lc.get("loss_mask_blur_pct", 0.0), (model_width, model_height))
    mask_t = torch.from_numpy(mask_loss).unsqueeze(0).unsqueeze(0)

    criterion = CombinedLoss(cfg)
    criterion.set_progress(1.0)
    with torch.no_grad():
        _, breakdown = criterion(pred_float, target, mask_t, delta=delta)
    return breakdown


def _save_heatmap(path: str, diff_gray: np.ndarray, label: str) -> None:
    max_diff = diff_gray.max()
    diff_norm = (
        (diff_gray / max_diff * 255).astype(np.uint8)
        if max_diff > 0 else np.zeros_like(diff_gray, dtype=np.uint8)
    )
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    h, w = heatmap.shape[:2]
    bar_w = 40
    bar = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    bar_bgr = cv2.applyColorMap(np.tile(bar, (1, bar_w)), cv2.COLORMAP_JET)

    canvas = np.zeros((h, w + 80, 3), dtype=np.uint8)
    canvas[:, :w] = heatmap
    canvas[:, w + 10:w + 10 + bar_w] = bar_bgr

    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    color = (255, 255, 255)
    cv2.putText(canvas, f"{max_diff:.2f}", (w + 10, 20), font, fs, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, "0.00", (w + 10, h - 10), font, fs, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, label, (5, h - 10), font, fs, color, thick, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def _cfg_weight_end(v) -> float:
    if isinstance(v, (list, tuple)):
        return float(v[-1])
    return float(v)


def _weighted_loss_map(pred_bgr: np.ndarray,
                       clean_bgr: np.ndarray,
                       mask_binary: np.ndarray,
                       cfg: dict,
                       image_size: tuple[int, int]) -> np.ndarray:
    return shared_weighted_loss_map(pred_bgr, clean_bgr, mask_binary, cfg, image_size)


def save_debug_frames(out_path: str,
                      wm_r: np.ndarray,
                      mask_r: np.ndarray,
                      grad_np: np.ndarray,
                      pred_bgr: np.ndarray,
                      clean_path: str = None,
                      loss_breakdown: dict = None,
                      mask_binary: np.ndarray = None,
                      cfg: dict = None,
                      roi: dict = None) -> None:
    import os

    base, ext = os.path.splitext(out_path)
    cv2.imwrite(f"{base}_debug_0_wm_resized{ext}", wm_r)
    cv2.imwrite(f"{base}_debug_1_mask_resized{ext}", (mask_r * 255).astype(np.uint8))
    cv2.imwrite(f"{base}_debug_2_gradient{ext}", grad_np)
    cv2.imwrite(f"{base}_debug_3_model_raw{ext}", pred_bgr)

    delta_gray = np.abs(pred_bgr.astype(np.float32) - wm_r.astype(np.float32)).mean(axis=2)
    _save_heatmap(f"{base}_debug_4_delta_heatmap{ext}", delta_gray, f"delta={delta_gray.mean():.2f}")

    if clean_path and roi is not None:
        clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        if clean is not None:
            clean_r = crop_by_roi(clean, roi, pred_bgr.shape[1], pred_bgr.shape[0], is_mask=False)
            diff_gray = np.abs(pred_bgr.astype(np.float32) - clean_r.astype(np.float32)).mean(axis=2)
            label = f"loss={loss_breakdown['total']:.4f}" if loss_breakdown else f"MAE={diff_gray.mean():.2f}"
            _save_heatmap(f"{base}_debug_5_error_heatmap{ext}", diff_gray, label)

            if mask_binary is not None and cfg is not None:
                wmap = _weighted_loss_map(pred_bgr, clean_r, mask_binary, cfg, (pred_bgr.shape[1], pred_bgr.shape[0]))
                lc = cfg["loss"]
                legend = (
                    f"l1m*{_cfg_weight_end(lc.get('l1_masked', 0))}"
                    f"  bd*{_cfg_weight_end(lc.get('bg_delta', 0))}"
                    f"  brd*{_cfg_weight_end(lc.get('border', 0))}"
                )
                _save_heatmap(f"{base}_debug_6_weighted_loss{ext}", wmap, legend)

    if loss_breakdown:
        bd = loss_breakdown
        print(
            f"  loss={bd['total']:.4f}  "
            f"l1_masked={bd['l1_masked']:.4f}  "
            f"perc={bd['perceptual']:.4f}  "
            f"sat={bd['saturation']:.4f}  "
            f"cm={bd['color_moment']:.4f}  "
            f"border={bd['border']:.4f}  "
            f"btv={bd['bg_tv']:.4f}  "
            f"bd={bd['bg_delta']:.4f}"
        )

    print(f"Saved debug frames to {base}_debug_*")


def _debug_grid(model, cfg, device, n_samples: int, model_width: int, model_height: int, only_pred: bool = False):
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt

    ds_root = Path(cfg["dataset"]["root"])
    store_root = get_store_root(ds_root, cfg["dataset"].get("preprocessed_store_dir"))
    store_signature = build_store_signature(
        image_width=model_width,
        image_height=model_height,
        crop_aspect_ratio=cfg["dataset"].get("crop_aspect_ratio", 3.54),
        crop_margin_ratio=cfg["dataset"].get("crop_margin_ratio", 0.10),
        crop_min_width_ratio=cfg["dataset"].get("crop_min_width_ratio", 0.50),
    )
    candidates = sorted([
        d for d in ds_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "mask.png").exists()
        and (d / "clean.png").exists()
    ])
    if not candidates:
        raise RuntimeError(f"No samples found in {ds_root}")

    samples = random.sample(candidates, min(n_samples, len(candidates)))

    if only_pred:
        cols = int(math.ceil(math.sqrt(len(samples))))
        rows = int(math.ceil(len(samples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    else:
        cols = 3
        rows = len(samples)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

    fig.suptitle("Predicted Clean Results" if only_pred else "Watermarked  |  Predicted Clean  |  Ground Truth", fontsize=10)

    for i, sample_dir in enumerate(samples):
        wm = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        wm_r, clean_r, mask_binary, mask_input = load_or_build_preprocessed_sample(
            store_root=store_root,
            dataset_root=ds_root,
            sample_dir=sample_dir,
            signature=store_signature,
            image_width=model_width,
            image_height=model_height,
            crop_aspect_ratio=cfg["dataset"].get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=cfg["dataset"].get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=cfg["dataset"].get("crop_min_width_ratio", 0.50),
        )

        rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).to(device)
        grad_t = compute_gradient(wm_r).unsqueeze(0).to(device)
        inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)

        with torch.no_grad():
            delta = model(inp)
            pred = (rgb_t - delta).clamp(-1, 1)

        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_np = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        if only_pred:
            ax = axes[i // cols, i % cols]
            ax.imshow(pred_np)
            ax.set_title(sample_dir.name, fontsize=8)
        else:
            axes[i, 0].imshow(cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_ylabel(sample_dir.name, fontsize=8)
            axes[i, 1].imshow(pred_np)
            axes[i, 2].imshow(cv2.cvtColor(clean_r, cv2.COLOR_BGR2RGB))

    for i in range(len(samples), rows * cols):
        axes[i // cols, i % cols].axis("off")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--watermarked", default=None)
    parser.add_argument("--mask", default=None,
                        help="path to binary mask PNG; omit to auto-generate via --seg-checkpoint")
    parser.add_argument("--seg-checkpoint", default=None,
                        help="segmentation model checkpoint for automatic mask prediction")
    parser.add_argument("--seg-config", default="configs/seg.yaml")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--width", type=int, default=None,
                        help="model ROI width override; defaults to dataset.image_width")
    parser.add_argument("--height", type=int, default=None,
                        help="model ROI height override; defaults to dataset.image_height")
    parser.add_argument("--config", default="configs/train_256.yaml")
    parser.add_argument("--clean", default=None,
                        help="optional clean ground truth image for loss heatmap generation in debug mode")
    parser.add_argument("--debug", action="store_true",
                        help="save intermediate resized/model-output images for debugging")
    parser.add_argument("--debug-grid", action="store_true",
                        help="show a grid of N random dataset samples in a window")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--feather", type=int, default=9,
                        help="blend softness radius in px at model resolution (default 9)")
    parser.add_argument("--mask-expand", type=int, default=0,
                        help="extra mask dilation in px at model resolution before feathering; "
                             "use when the input mask underestimates the watermark extent")
    parser.add_argument("--only-pred", action="store_true",
                        help="in debug-grid mode, only show the predicted clean results")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    crop_aspect_ratio = cfg["dataset"].get("crop_aspect_ratio", 3.54)
    crop_margin_ratio = cfg["dataset"].get("crop_margin_ratio", 0.10)
    crop_min_width_ratio = cfg["dataset"].get("crop_min_width_ratio", 0.50)
    model_width = args.width or cfg["dataset"]["image_width"]
    model_height = args.height or cfg["dataset"]["image_height"]

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_cfg)
    model = load_model(cfg, args.checkpoint, device)

    if args.debug_grid:
        _debug_grid(model, cfg, device, args.n_samples, model_width, model_height, only_pred=args.only_pred)
        return

    if not args.watermarked:
        parser.error("--watermarked is required unless --debug-grid is set")
    if args.mask is None and args.seg_checkpoint is None:
        parser.error("provide either --mask or --seg-checkpoint")

    if args.mask is not None:
        inp, orig_wm, orig_mask, orig_size, wm_r, mask_r, grad_np, mask_binary, roi = preprocess(
            args.watermarked,
            args.mask,
            model_width,
            model_height,
            dilate=args.debug,
            crop_aspect_ratio=crop_aspect_ratio,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_width_ratio=crop_min_width_ratio,
        )
    else:
        seg_cfg = load_yaml_config(args.seg_config)
        seg_model = load_seg_model(seg_cfg, args.seg_checkpoint, device)

        orig_wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
        if orig_wm is None:
            raise FileNotFoundError(f"Cannot read {args.watermarked}")
        orig_size = (orig_wm.shape[1], orig_wm.shape[0])

        seg_size = seg_cfg["dataset"]["image_size"]
        orig_mask = predict_mask(seg_model, orig_wm, seg_size, device)
        print(f"Mask predicted by segmentation model (seg size={seg_size})")

        inp, wm_r, mask_r, grad_np, mask_binary, roi = _prepare_roi_input(
            orig_wm,
            orig_mask,
            model_width,
            model_height,
            dilate=args.debug,
            crop_aspect_ratio=crop_aspect_ratio,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_width_ratio=crop_min_width_ratio,
        )

    pred_bgr, pred_float, delta = run(model, inp, device)

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
            delta=delta,
        )

    if args.debug:
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
    print(f"Saved -> {args.output}  (original resolution: {orig_size[0]}x{orig_size[1]})")


if __name__ == "__main__":
    main()
