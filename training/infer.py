"""
infer.py  –  run the trained model on a single image + mask
-------------------------------------------------------------
Usage:
  # provide mask manually
  python infer.py \\
      --checkpoint checkpoints/epoch_0060.pth \\
      --watermarked path/to/watermarked.jpg \\
      --mask        path/to/mask.png \\
      --output      result.png \\
      [--size 512]  [--config configs/train.yaml]

  # Debug mode: show N random dataset samples in a matplotlib grid
  python infer.py \\
      --checkpoint checkpoints/epoch_0060.pth \\
      --debug-grid [--n-samples 6] \\
      [--size 512] [--config configs/train.yaml]
"""

import argparse
import math

import cv2
import numpy as np
import torch
import yaml

from src.model       import build_model
from src.seg_model   import build_seg_model
from src.image_utils import compute_gradient, dilate_mask_input, blur_mask_for_loss
from src.losses      import CombinedLoss


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_seg_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_seg_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(seg_model, wm_bgr: np.ndarray, size: int,
                 device: torch.device) -> np.ndarray:
    """
    Run the segmentation model on a BGR image and return a binary uint8 mask
    (0 / 255) at the original image resolution.
    """
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    h, w = wm_bgr.shape[:2]
    wm_r = cv2.resize(wm_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb  = (rgb - _MEAN) / _STD
    inp  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1x3xHxW

    pred   = torch.sigmoid(seg_model(inp))                       # 1x1xHxW in [0,1]
    pred_np = pred.squeeze().cpu().numpy()                       # HxW
    binary  = (pred_np > 0.5).astype(np.uint8) * 255
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)


def preprocess(wm_path: str, mask_path: str, size: int, dilate: bool = False):
    """
    Returns
    -------
    inp_tensor : torch.Tensor  1x4xHxW  in [-1,1] / {0,1}
    orig_wm    : np.ndarray    HxWx3 uint8 BGR  (original-size, for blending)
    orig_mask  : np.ndarray    HxW  uint8       (original-size, for blending)
    orig_size  : (W, H)
    """
    wm   = cv2.imread(wm_path,   cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Cannot read {wm_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")

    orig_size = (wm.shape[1], wm.shape[0])   # (W, H)

    wm_r      = cv2.resize(wm,   (size, size), interpolation=cv2.INTER_AREA)
    mask_r    = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_r > 127).astype(np.float32)   # raw binary — kept for loss mask
    mask_r = dilate_mask_input(mask_binary) if dilate else mask_binary

    rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)       # 1x3xHxW
    mask_t = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0)          # 1x1xHxW

    grad_t = compute_gradient(wm_r)
    grad_np = (grad_t.squeeze().numpy() * 255).astype(np.uint8)           # save numpy for debug
    grad_t = grad_t.unsqueeze(0)                                          # 1x1xHxW

    inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)   # 1x5xHxW
    return inp, wm, mask, orig_size, wm_r, mask_r, grad_np, mask_binary


@torch.no_grad()
def run(model, inp: torch.Tensor, device: torch.device):
    """Returns (pred_bgr uint8 at model resolution, pred_float Bx3xHxW in [-1,1], delta Bx3xHxW)."""
    inp        = inp.to(device)
    delta      = model(inp)                                # Bx3xHxW, residue in [-2, 2]
    pred_float = (inp[:, :3] - delta).clamp(-1, 1)        # Bx3xHxW, clean [-1,1]
    pred_np    = pred_float.squeeze(0).cpu().numpy()       # 3xHxW
    pred_np    = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    pred_bgr   = cv2.cvtColor(pred_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return pred_bgr, pred_float.cpu(), delta.cpu()


def blend_back(pred_bgr, orig_wm, orig_mask, orig_size,
               feather: int = 9, mask_expand: int = 0):
    """
    Paste the model prediction back at original resolution.

    feather     : blend softness radius (px) at model resolution, scaled to
                  original resolution automatically. Controls how smooth the
                  transition edge looks.
    mask_expand : extra dilation (px) at model resolution applied BEFORE the
                  feather dilation. Use this when the input mask underestimates
                  the watermark extent — expanding it ensures all watermark
                  pixels are covered by pred_up before the feather zone begins,
                  preventing watermark bleed-back through the blend.
    """
    pred_up = cv2.resize(pred_bgr, orig_size, interpolation=cv2.INTER_CUBIC)

    if feather > 0 or mask_expand > 0:
        scale = orig_size[0] / pred_bgr.shape[1]
        feather_scaled    = max(1, round(feather * scale))
        mask_expand_scaled = round(mask_expand * scale)

        working_mask = orig_mask
        # First pass: expand coverage to fully capture approximate masks
        if mask_expand_scaled > 0:
            exp_k = mask_expand_scaled * 2 + 1
            working_mask = cv2.dilate(working_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (exp_k, exp_k)))

        # Second pass: push feather zone into clean background so both sides
        # of the blend are watermark-free (no bleed-back)
        if feather_scaled > 0:
            dil_k = feather_scaled * 2 + 1
            working_mask = cv2.dilate(working_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dil_k, dil_k)))
            working_mask = cv2.GaussianBlur(
                working_mask.astype(np.float32), (dil_k, dil_k), feather_scaled / 2)

        m = (working_mask / 255.0)[:, :, None]
    else:
        m = (orig_mask > 127)[:, :, None].astype(np.float32)

    out = (pred_up * m + orig_wm * (1 - m)).clip(0, 255).astype(np.uint8)
    return out


def compute_debug_loss(cfg: dict, pred_float: torch.Tensor,
                       mask_binary: np.ndarray, clean_path: str, model_size: int,
                       delta: torch.Tensor | None = None) -> dict | None:
    """
    Run CombinedLoss identically to the trainer.
    mask_binary : HxW float32 {0,1} — the raw binary mask BEFORE any dilation,
                  matching mask.png from the dataset. We build mask_loss here
                  the same way WatermarkDataset does (dilate_for_loss → blur_for_loss).
    pred_float  : 1x3xHxW in [-1,1]  (on CPU)
    delta       : 1x3xHxW raw model output (on CPU), used for bg_tv / bg_delta terms.
    """
    clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if clean is None:
        print(f"[debug] Cannot read clean image: {clean_path}")
        return None

    clean_r = cv2.resize(clean, (model_size, model_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(clean_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    target = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)   # 1x3xHxW [-1,1]

    # Build mask_loss exactly as WatermarkDataset does
    lc = cfg["loss"]
    mask_loss = blur_mask_for_loss(mask_binary,
                                   lc.get("loss_mask_blur_pct", 0.0), model_size)
    mask_t = torch.from_numpy(mask_loss).unsqueeze(0).unsqueeze(0)   # 1x1xHxW

    criterion = CombinedLoss(cfg)
    criterion.set_progress(1.0)   # use end-of-training weights for a fully trained model
    with torch.no_grad():
        _, breakdown = criterion(pred_float, target, mask_t, delta=delta)
    return breakdown


def _save_heatmap(path: str, diff_gray: np.ndarray, label: str) -> None:
    """Save a JET heatmap with a colorbar and a text label."""
    max_diff = diff_gray.max()
    diff_norm = (diff_gray / max_diff * 255).astype(np.uint8) if max_diff > 0 \
                else np.zeros_like(diff_gray, dtype=np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    h, w = heatmap.shape[:2]
    bar_w = 40
    bar = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    bar_bgr = cv2.applyColorMap(np.tile(bar, (1, bar_w)), cv2.COLORMAP_JET)

    canvas = np.zeros((h, w + 80, 3), dtype=np.uint8)
    canvas[:, :w] = heatmap
    canvas[:, w+10 : w+10+bar_w] = bar_bgr

    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    color = (255, 255, 255)
    cv2.putText(canvas, f"{max_diff:.2f}", (w + 10, 20),    font, fs, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, "0.00",            (w + 10, h - 10), font, fs, color, thick, cv2.LINE_AA)
    cv2.putText(canvas, label,             (5,      h - 10), font, fs, color, thick, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def _cfg_weight_end(v) -> float:
    """Extract the end-of-training weight from a scalar or [start, end] ramp config value."""
    if isinstance(v, (list, tuple)):
        return float(v[-1])
    return float(v)


def _weighted_loss_map(pred_bgr: np.ndarray, clean_bgr: np.ndarray,
                       mask_binary: np.ndarray, cfg: dict, size: int) -> np.ndarray:
    """
    Per-pixel weighted loss contribution matching CombinedLoss.

    Spatial components included (same logic as trainer):
      l1_masked  — w × |pred − target| × mask_interior  (eroded by 3 px)
      l1_background — w × |pred − target| × (1−soft_mask)
      border     — w × |pred − target| × 4·m·(1−m)      (feathered edge ring)

    SSIM and perceptual are global scalars and cannot be decomposed spatially;
    they appear only in the printed loss_breakdown.

    Ramped weights ([start, end] in config) use the end-of-training value,
    matching a fully-trained checkpoint.

    Returns HxW float32.  The scale is comparable to what the optimiser sees
    (i.e. pixel error already multiplied by the configured λ weights).
    """
    lc = cfg["loss"]
    w_l1_masked     = _cfg_weight_end(lc.get("l1_masked",     0.0))
    w_bg_delta      = _cfg_weight_end(lc.get("bg_delta",      0.0))
    w_border        = _cfg_weight_end(lc.get("border",        0.0))
    blur_pct        = lc.get("loss_mask_blur_pct", 0.0)

    pred_f   = pred_bgr.astype(np.float32) / 127.5 - 1.0
    target_f = clean_bgr.astype(np.float32) / 127.5 - 1.0
    abs_err  = np.abs(pred_f - target_f).mean(axis=2)   # HxW, [0, 2]

    # Match the soft loss mask built by WatermarkDataset
    soft_mask = blur_mask_for_loss(mask_binary, blur_pct, size)  # HxW [0,1]

    # mask_interior: erode by 3 px — mirrors the max_pool trick in CombinedLoss
    kernel        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_interior = cv2.erode(soft_mask, kernel, iterations=1)

    # Border ring weight: 4·m·(1−m), peaks at the feathered midpoint
    border_w = 4.0 * soft_mask * (1.0 - soft_mask)

    # Background weight: complement of soft_mask (represents bg_delta spatially)
    bg_delta_w = 1.0 - soft_mask

    return (w_l1_masked     * abs_err * mask_interior
          + w_bg_delta      * abs_err * bg_delta_w
          + w_border        * abs_err * border_w).astype(np.float32)


def save_debug_frames(out_path: str, wm_r: np.ndarray, mask_r: np.ndarray,
                      grad_np: np.ndarray, pred_bgr: np.ndarray,
                      clean_path: str = None, loss_breakdown: dict = None,
                      mask_binary: np.ndarray = None, cfg: dict = None) -> None:
    """Save intermediate model inputs / outputs for debugging."""
    import os
    base, ext = os.path.splitext(out_path)
    cv2.imwrite(f"{base}_debug_0_wm_resized{ext}", wm_r)
    cv2.imwrite(f"{base}_debug_1_mask_resized{ext}", (mask_r * 255).astype(np.uint8))
    cv2.imwrite(f"{base}_debug_2_gradient{ext}", grad_np)
    cv2.imwrite(f"{base}_debug_3_model_raw{ext}", pred_bgr)

    # Always: heatmap of what the model actually changed (no ground truth needed)
    delta_gray = np.abs(pred_bgr.astype(np.float32) - wm_r.astype(np.float32)).mean(axis=2)
    _save_heatmap(f"{base}_debug_4_delta_heatmap{ext}", delta_gray,
                  f"delta={delta_gray.mean():.2f}")

    # When ground truth is available: raw error + weighted loss heatmaps
    if clean_path:
        clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        if clean is not None:
            size_h, size_w = pred_bgr.shape[:2]
            clean_r = cv2.resize(clean, (size_w, size_h), interpolation=cv2.INTER_AREA)

            # debug_5: unweighted per-pixel MAE vs ground truth
            diff_gray = np.abs(pred_bgr.astype(np.float32) - clean_r.astype(np.float32)).mean(axis=2)
            label = f"loss={loss_breakdown['total']:.4f}" if loss_breakdown \
                    else f"MAE={diff_gray.mean():.2f}"
            _save_heatmap(f"{base}_debug_5_error_heatmap{ext}", diff_gray, label)

            # debug_6: per-pixel loss weighted by training λ values (l1_full,
            # l1_masked, border).  Shows exactly where the optimiser penalises
            # the model — useful for diagnosing outside-mask leakage.
            if mask_binary is not None and cfg is not None:
                wmap = _weighted_loss_map(pred_bgr, clean_r, mask_binary, cfg, size_w)
                lc = cfg["loss"]
                legend = (f"l1m×{_cfg_weight_end(lc.get('l1_masked',0))}"
                          f"  bd×{_cfg_weight_end(lc.get('bg_delta',0))}"
                          f"  brd×{_cfg_weight_end(lc.get('border',0))}")
                _save_heatmap(f"{base}_debug_6_weighted_loss{ext}", wmap, legend)

    if loss_breakdown:
        bd = loss_breakdown
        print(f"  loss={bd['total']:.4f}  "
              f"l1_masked={bd['l1_masked']:.4f}  "
              f"perc={bd['perceptual']:.4f}  "
              f"sat={bd['saturation']:.4f}  "
              f"cm={bd['color_moment']:.4f}  "
              f"border={bd['border']:.4f}  "
              f"itv={bd['interior_tv']:.4f}  "
              f"btv={bd['bg_tv']:.4f}  "
              f"bd={bd['bg_delta']:.4f}")

    print(f"Saved debug frames to {base}_debug_*")


def _debug_grid(model, cfg, device, n_samples: int, size: int, only_pred: bool = False):
    """
    Pick N random samples from the dataset and show a comparison grid:
    Watermarked | Predicted Clean | Ground Truth Clean
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import random

    ds_root = Path(cfg["dataset"]["root"])
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
        cols = 3  # wm | pred | gt
        rows = len(samples)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

    fig.suptitle("Predicted Clean Results" if only_pred else "Watermarked  |  Predicted Clean  |  Ground Truth", fontsize=10)

    for i, sample_dir in enumerate(samples):
        # Load
        wm    = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
        mask  = cv2.imread(str(sample_dir / "mask.png"),        cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(sample_dir / "clean.png"),       cv2.IMREAD_COLOR)

        # Preprocess
        wm_r    = cv2.resize(wm,    (size, size), interpolation=cv2.INTER_AREA)
        mask_r  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_r > 127).astype(np.float32)
        
        rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).to(device)
        
        grad_t = compute_gradient(wm_r).unsqueeze(0).to(device)
        inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)

        # Run
        with torch.no_grad():
            delta = model(inp)
            pred  = (rgb_t - delta).clamp(-1, 1)

        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_np = ((pred_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        if only_pred:
            ax = axes[i // cols, i % cols]
            ax.imshow(pred_np)
            ax.set_title(sample_dir.name, fontsize=8)
        else:
            # Plot All (Row per sample)
            axes[i, 0].imshow(cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_ylabel(sample_dir.name, fontsize=8)
            axes[i, 1].imshow(pred_np)
            axes[i, 2].imshow(cv2.cvtColor(cv2.resize(clean, (size, size)), cv2.COLOR_BGR2RGB))

    for i in range(len(samples), rows * cols):
        axes[i // cols, i % cols].axis("off")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--watermarked",    default=None)
    parser.add_argument("--mask",           default=None,
                        help="path to binary mask PNG; omit to auto-generate via --seg-checkpoint")
    parser.add_argument("--seg-checkpoint", default=None,
                        help="segmentation model checkpoint for automatic mask prediction")
    parser.add_argument("--seg-config",     default="configs/seg.yaml")
    parser.add_argument("--output",         default="result.png")
    parser.add_argument("--size",           type=int, default=512)
    parser.add_argument("--config",         default="configs/train.yaml")
    parser.add_argument("--clean",          default=None,
                        help="optional clean ground truth image for loss heatmap generation in debug mode")
    parser.add_argument("--debug",          action="store_true",
                        help="save intermediate resized/model-output images for debugging")
    parser.add_argument("--debug-grid",     action="store_true",
                        help="show a grid of N random dataset samples in a window")
    parser.add_argument("--n-samples",      type=int, default=4)
    parser.add_argument("--feather",        type=int, default=9,
                        help="blend softness radius in px at model resolution (default 9)")
    parser.add_argument("--mask-expand",    type=int, default=0,
                        help="extra mask dilation in px at model resolution before feathering; "
                             "use when the input mask underestimates the watermark extent")
    parser.add_argument("--only-pred",      action="store_true",
                        help="in debug-grid mode, only show the predicted clean results")
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(cfg.get("device", "auto"))
    if device.type == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, args.checkpoint, device)

    if args.debug_grid:
        _debug_grid(model, cfg, device, args.n_samples, args.size, only_pred=args.only_pred)
        return

    if not args.watermarked:
        parser.error("--watermarked is required unless --debug-grid is set")

    if args.mask is None and args.seg_checkpoint is None:
        parser.error("provide either --mask or --seg-checkpoint")

    if args.mask is not None:
        inp, orig_wm, orig_mask, orig_size, wm_r, mask_r, grad_np, mask_binary = preprocess(
            args.watermarked, args.mask, args.size, dilate=args.debug
        )
    else:
        seg_cfg  = load_cfg(args.seg_config)
        seg_model = load_seg_model(seg_cfg, args.seg_checkpoint, device)

        orig_wm = cv2.imread(args.watermarked, cv2.IMREAD_COLOR)
        if orig_wm is None:
            raise FileNotFoundError(f"Cannot read {args.watermarked}")
        orig_size = (orig_wm.shape[1], orig_wm.shape[0])

        seg_size  = seg_cfg["dataset"]["image_size"]
        orig_mask = predict_mask(seg_model, orig_wm, seg_size, device)
        print(f"Mask predicted by segmentation model (seg size={seg_size})")

        wm_r        = cv2.resize(orig_wm,   (args.size, args.size), interpolation=cv2.INTER_AREA)
        mask_r      = cv2.resize(orig_mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_r > 127).astype(np.float32)
        mask_r      = dilate_mask_input(mask_binary) if args.debug else mask_binary

        rgb    = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb_t  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        mask_t = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0)

        grad_t = compute_gradient(wm_r)
        grad_np = (grad_t.squeeze().numpy() * 255).astype(np.uint8)
        grad_t = grad_t.unsqueeze(0)

        inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)

    pred_bgr, pred_float, delta = run(model, inp, device)

    loss_breakdown = None
    if args.debug and args.clean:
        loss_breakdown = compute_debug_loss(cfg, pred_float, mask_binary,
                                            args.clean, args.size, delta=delta)

    if args.debug:
        save_debug_frames(args.output, wm_r, mask_r, grad_np, pred_bgr,
                          args.clean, loss_breakdown,
                          mask_binary=mask_binary, cfg=cfg)

    result   = blend_back(pred_bgr, orig_wm, orig_mask, orig_size,
                          feather=args.feather, mask_expand=args.mask_expand)

    cv2.imwrite(args.output, result)
    print(f"Saved → {args.output}  (original resolution: {orig_size[0]}×{orig_size[1]})")


if __name__ == "__main__":
    main()
