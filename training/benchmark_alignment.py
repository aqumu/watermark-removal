"""
benchmark_alignment.py  –  Compare watermark alignment algorithms on the synthetic dataset
============================================================================================

For each dataset sample we have:
  - watermarked.jpg      : the input image (with semi-transparent watermark)
  - mask.png             : the ground-truth soft alpha mask
  - meta.json            : {"position": [x, y], "blend_mode": "..."}

The original watermark template lives at:
  data_gen/watermark.png (RGBA)

Each method receives:
  - prob_map  : HxW float32 [0,1]  — raw sigmoid output from the seg model
  - template  : HxW float32 [0,1]  — original watermark alpha at base scale
  - image W,H : image dimensions (for scale computation)

And must return:
  - (tx, ty, scale)  — top-left corner (in original image pixels) + scale fraction

We measure:
  - Position error  : Euclidean distance between predicted (tx,ty) and ground-truth (x,y)
  - Scale error     : |pred_scale - gt_scale| (as fraction of image width)
  - Aligned IoU     : IoU between the placed template and the ground-truth mask

Usage:
    python benchmark_alignment.py \\
        --seg-checkpoint artifacts/checkpoints/segmentation/best.pth \\
        --seg-config     configs/seg.yaml \\
        --watermark      ../data_gen/watermark.png \\
        --dataset        ../data_gen/dataset \\
        --n-samples      200 \\
        [--no-model]     # skip seg model, use ground-truth mask as prob_map

    python benchmark_alignment.py --help
"""

import argparse
import io
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import fftconvolve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class AlignResult:
    tx: float        # predicted top-left X in original image pixels
    ty: float        # predicted top-left Y in original image pixels
    scale: float     # predicted scale fraction (watermark_width / image_width)


@dataclass
class SampleMetrics:
    pos_err: float   # Euclidean distance error in original pixels
    scale_err: float # |pred_scale - gt_scale|
    iou: float       # IoU of aligned template vs ground-truth binary mask


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_meta(sample_dir: Path) -> dict:
    with open(sample_dir / "meta.json") as f:
        return json.load(f)


def gt_transform(meta: dict, wm_rgba: np.ndarray, img_w: int, img_h: int) -> dict:
    """Recover the exact ground-truth placement from meta.json.

    meta["position"] = [x, y]  (top-left of watermark in original image space)
    We recover scale by back-computing from the mask bounding box.
    """
    x, y = meta["position"]
    return {"x": x, "y": y}


def gt_scale_from_mask(mask_bin: np.ndarray, wm_alpha: np.ndarray, img_w: int) -> float:
    """Estimate gt_scale from the ground-truth mask bounding box width."""
    cols = np.any(mask_bin, axis=0)
    if not cols.any():
        return float(wm_alpha.shape[1]) / img_w
    wm_width_px = cols.sum()
    return float(wm_width_px) / float(img_w)


def place_template(template: np.ndarray, tx: int, ty: int,
                   scale: float, img_w: int, img_h: int) -> np.ndarray:
    """Render a binary mask of the placed template at (tx, ty) with given scale."""
    tw = max(1, int(round(img_w * scale)))
    th = max(1, int(round(tw * template.shape[0] / template.shape[1])))
    t_bin = (cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    x0, y0 = int(round(tx)), int(round(ty))
    x1 = min(x0 + tw, img_w)
    y1 = min(y0 + th, img_h)
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    if x1 > max(0, x0) and y1 > max(0, y0):
        canvas[max(0,y0):y1, max(0,x0):x1] = t_bin[sy0:sy0+(y1-max(0,y0)), sx0:sx0+(x1-max(0,x0))]
    return canvas


def iou_masks(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    intersection = float((pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    return intersection / max(union, 1.0)


# ---------------------------------------------------------------------------
# Shared building blocks used by multiple methods
# ---------------------------------------------------------------------------

def render_template(template: np.ndarray, scale: float, img_w: int, img_h: int, seg_w: int, seg_h: int) -> np.ndarray:
    # 1. Size in original image
    tw_orig = max(1, int(round(img_w * scale)))
    th_orig = max(1, int(round(tw_orig * template.shape[0] / template.shape[1])))
    
    # 2. Projected size in squished seg-map space
    tw_seg = max(1, int(round(tw_orig * seg_w / img_w)))
    th_seg = max(1, int(round(th_orig * seg_h / img_h)))
    
    return cv2.resize(template, (tw_seg, th_seg), interpolation=cv2.INTER_AREA)


def confidence_weighted_xcorr(
    confidence: np.ndarray,
    template_scaled: np.ndarray,
    img_w: int,
    img_h: int,
    subpixel: bool = False,
) -> Tuple[float, float, float]:
    """
    Cross-correlate confidence map with template.
    Returns (tx, ty, peak_value) — tx,ty are the top-left coordinates.
    If subpixel=True, refines the peak via parabolic interpolation.
    """
    corr = fftconvolve(confidence, template_scaled[::-1, ::-1], mode="full")
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    th, tw = template_scaled.shape[:2]
    py, px = peak_idx

    if subpixel and 0 < py < corr.shape[0] - 1 and 0 < px < corr.shape[1] - 1:
        # Fit 1-D parabola along each axis through the 3 peak neighbours
        denom_x = corr[py, px - 1] - 2 * corr[py, px] + corr[py, px + 1]
        denom_y = corr[py - 1, px] - 2 * corr[py, px] + corr[py + 1, px]
        sub_dx = 0.5 * (corr[py, px - 1] - corr[py, px + 1]) / (denom_x + 1e-12)
        sub_dy = 0.5 * (corr[py - 1, px] - corr[py + 1, px]) / (denom_y + 1e-12)
        px = px + float(np.clip(sub_dx, -1.0, 1.0))
        py = py + float(np.clip(sub_dy, -1.0, 1.0))

    ty = float(py) - (th - 1)
    tx = float(px) - (tw - 1)
    peak_val = float(corr[peak_idx])
    return tx, ty, peak_val


def ncc_map(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Globally-normalised cross-correlation: zero-mean + unit-variance for both
    signal and kernel before convolving.  Cheaper than sliding-window NCC but
    still removes DC bias and amplitude differences.
    """
    s = signal.astype(np.float64)
    k = kernel.astype(np.float64)
    s = (s - s.mean()) / (s.std() + 1e-8)
    k = (k - k.mean()) / (k.std() + 1e-8)
    return fftconvolve(s, k[::-1, ::-1], mode="full").astype(np.float32)


def scale_search(
    confidence: np.ndarray,
    template: np.ndarray,
    base_scale: float,
    scale_range: float,
    n_scales: int,
    img_w: int,
    img_h: int,
    seg_w: int,
    seg_h: int,
    subpixel: bool = False,
    use_ncc: bool = False,
) -> Tuple[float, float, float]:
    """Search over scale × translation for best fit. Returns (tx, ty, best_scale)."""
    scales = np.linspace(base_scale * (1 - scale_range),
                         base_scale * (1 + scale_range), n_scales)
    best_score = -np.inf
    best_tx, best_ty, best_scale = 0.0, 0.0, base_scale

    for s in scales:
        t_scaled = render_template(template, s, img_w, img_h, seg_w, seg_h)
        if use_ncc:
            corr = ncc_map(confidence, t_scaled)
            peak_idx = np.unravel_index(corr.argmax(), corr.shape)
            th, tw = t_scaled.shape[:2]
            py, px = peak_idx
            if subpixel and 0 < py < corr.shape[0] - 1 and 0 < px < corr.shape[1] - 1:
                ipy, ipx = int(py), int(px)
                denom_x = corr[ipy, ipx - 1] - 2 * corr[ipy, ipx] + corr[ipy, ipx + 1]
                denom_y = corr[ipy - 1, ipx] - 2 * corr[ipy, ipx] + corr[ipy + 1, ipx]
                px = ipx + float(np.clip(0.5 * (corr[ipy, ipx - 1] - corr[ipy, ipx + 1]) / (denom_x + 1e-12), -1, 1))
                py = ipy + float(np.clip(0.5 * (corr[ipy - 1, ipx] - corr[ipy + 1, ipx]) / (denom_y + 1e-12), -1, 1))
            tx = float(px) - (tw - 1)
            ty = float(py) - (th - 1)
            peak = float(corr[peak_idx])
        else:
            tx, ty, peak = confidence_weighted_xcorr(confidence, t_scaled, seg_w, seg_h, subpixel=subpixel)
        if peak > best_score:
            best_score = peak
            best_tx, best_ty, best_scale = tx, ty, s

    return best_tx, best_ty, best_scale


# ---------------------------------------------------------------------------
# METHOD IMPLEMENTATIONS
# ---------------------------------------------------------------------------

# ── 0: ORACLE (perfect knowledge — upper bound) ──────────────────────────────

def method_oracle(prob_map, template, img_w, img_h, gt_x, gt_y, gt_scale, **_):
    return AlignResult(tx=gt_x, ty=gt_y, scale=gt_scale)


# ── 1: Prior only (no alignment at all) ───────────────────────────────────────

def method_prior_only(prob_map, template, img_w, img_h, base_scale=0.79, **_):
    """Place the watermark at the image center with the known base scale."""
    tw = int(round(img_w * base_scale))
    th = int(round(tw * template.shape[0] / template.shape[1]))
    tx = (img_w - tw) / 2.0
    ty = (img_h - th) / 2.0
    return AlignResult(tx=tx, ty=ty, scale=base_scale)


# ── 2: Centroid + bbox moments (binary mask) ──────────────────────────────────

def method_centroid_binary(prob_map, template, img_w, img_h, threshold=0.5, **_):
    """Classic moment-based: binarise → centroid → scale from bbox."""
    mask_bin = (prob_map > threshold).astype(np.float32)
    if mask_bin.sum() < 10:
        return method_prior_only(prob_map, template, img_w, img_h)

    # Scale from bounding box
    rows = np.any(mask_bin > 0, axis=1)
    cols = np.any(mask_bin > 0, axis=0)
    pred_bbox_w = cols.sum()
    pred_bbox_h = rows.sum()

    # Map from seg-model resolution to original image resolution
    scale_x = img_w / prob_map.shape[1]
    scale_y = img_h / prob_map.shape[0]

    pred_bbox_w_orig = pred_bbox_w * scale_x
    pred_bbox_h_orig = pred_bbox_h * scale_y

    # Infer scale from bbox width
    s = pred_bbox_w_orig / img_w

    tw = int(round(img_w * s))
    th = int(round(tw * template.shape[0] / template.shape[1]))

    # Centroid → top-left
    m = cv2.moments(mask_bin)
    if m["m00"] < 1e-6:
        return method_prior_only(prob_map, template, img_w, img_h)
    cx_map = m["m10"] / m["m00"]   # in seg-map pixels
    cy_map = m["m01"] / m["m00"]
    cx_orig = cx_map * scale_x
    cy_orig = cy_map * scale_y

    tx = cx_orig - tw / 2.0
    ty = cy_orig - th / 2.0
    return AlignResult(tx=tx, ty=ty, scale=s)


# ── 3: Soft moments (weighted by probability) ─────────────────────────────────

def method_soft_moments(prob_map, template, img_w, img_h, **_):
    """Use soft probability values as pixel weights for moments (no threshold)."""
    scale_x = img_w / prob_map.shape[1]
    scale_y = img_h / prob_map.shape[0]

    pm = prob_map.astype(np.float64)
    total = pm.sum()
    if total < 1e-6:
        return method_prior_only(prob_map, template, img_w, img_h)

    ys = np.arange(prob_map.shape[0], dtype=np.float64)
    xs = np.arange(prob_map.shape[1], dtype=np.float64)
    cx_map = (pm * xs[None, :]).sum() / total
    cy_map = (pm * ys[:, None]).sum() / total

    # Scale from soft moment of inertia → template size (Hu moment approach)
    mu20 = ((pm * (xs[None, :] - cx_map) ** 2).sum()) / total
    mu02 = ((pm * (ys[:, None] - cy_map) ** 2).sum()) / total
    var_x = mu20 * scale_x ** 2
    var_y = mu02 * scale_y ** 2

    # Template reference moments
    t_alpha = template.astype(np.float64)
    t_total = t_alpha.sum()
    t_xs = np.arange(template.shape[1], dtype=np.float64)
    t_ys = np.arange(template.shape[0], dtype=np.float64)
    t_cx = (t_alpha * t_xs[None, :]).sum() / (t_total + 1e-9)
    t_cy = (t_alpha * t_ys[:, None]).sum() / (t_total + 1e-9)
    t_mu20 = ((t_alpha * (t_xs[None, :] - t_cx) ** 2).sum()) / (t_total + 1e-9)
    t_mu02 = ((t_alpha * (t_ys[:, None] - t_cy) ** 2).sum()) / (t_total + 1e-9)

    # Scale ratio from inertia
    if t_mu20 > 1e-6 and t_mu02 > 1e-6:
        s_from_x = (var_x / t_mu20) ** 0.5
        s_from_y = (var_y / t_mu02) ** 0.5
        # But we need s as fraction of img_w, and t is already in template pixels
        s_x = s_from_x / template.shape[1]
        s_y = s_from_y / template.shape[0]
        s = float(np.clip((s_x + s_y) / 2.0, 0.5, 1.5))
    else:
        s = 0.79

    tw = int(round(img_w * s))
    th = int(round(tw * template.shape[0] / template.shape[1]))
    cx_orig = cx_map * scale_x
    cy_orig = cy_map * scale_y
    tx = cx_orig - tw / 2.0
    ty = cy_orig - th / 2.0
    return AlignResult(tx=tx, ty=ty, scale=s)


# ── 4: Hard correlation (binary mask × binary template) ───────────────────────

def method_binary_xcorr(prob_map, template, img_w, img_h,
                         threshold=0.5, base_scale=0.79,
                         scale_range=0.05, n_scales=11, **_):
    """Cross-correlate binarised mask with template (no confidence weighting)."""
    mask_bin = (prob_map > threshold).astype(np.float32)
    template_bin = (template > 0.5).astype(np.float32)

    # All correlation in seg-map space (256px) — fast
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        mask_bin, template_bin,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    # Upscale result to original image space only at the end
    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 5: Confidence-weighted correlation (prob^1) ───────────────────────────────

def method_conf_xcorr_linear(prob_map, template, img_w, img_h,
                              base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """Cross-correlate prob_map (linear weighting, alpha=1) with template."""
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        prob_map, template,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 6: Confidence-weighted correlation (prob^2) ───────────────────────────────

def method_conf_xcorr_sq(prob_map, template, img_w, img_h,
                          base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """Cross-correlate prob^2 (squaring suppresses the uncertain fringe)."""
    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        confidence, template,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 7: Confidence-weighted correlation (prob^3) ───────────────────────────────

def method_conf_xcorr_cubic(prob_map, template, img_w, img_h,
                            base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """Cross-correlate prob^3 (sharper confidence suppression)."""
    confidence = prob_map ** 3
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        confidence, template,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 8: Erode-then-correlate (binary, erosion-based cleaning) ─────────────────

def method_erode_xcorr(prob_map, template, img_w, img_h,
                        threshold=0.5, erode_px=4, base_scale=0.79,
                        scale_range=0.05, n_scales=11, **_):
    """Erode the binary mask before correlating (removes over-predicted border)."""
    mask_bin = (prob_map > threshold).astype(np.uint8)
    if erode_px > 0:
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        mask_bin = cv2.erode(mask_bin, kernel)
    mask_f = mask_bin.astype(np.float32)
    template_bin = (template > 0.5).astype(np.float32)

    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        mask_f, template_bin,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 9: Two-stage: soft moments → refine with conf^2 xcorr ────────────────────

def method_two_stage(prob_map, template, img_w, img_h,
                     scale_range=0.05, n_scales=11, **_):
    """
    Stage 1: soft moments → coarse (tx, ty, scale)
    Stage 2: conf^2 xcorr in a tight window around Stage-1 estimate
    """
    coarse = method_soft_moments(prob_map, template, img_w, img_h)
    s_coarse = float(np.clip(coarse.scale, 0.5, 1.5))

    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]

    tight_range = min(scale_range * 0.5, 0.025)

    tx_seg, ty_seg, best_scale_seg = scale_search(
        confidence, template,
        s_coarse, tight_range, n_scales,
        img_w, img_h, seg_w, seg_h,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 10: Prior + conf^2 xcorr (no scale search, fixed prior scale) ─────────────

def method_prior_xcorr(prob_map, template, img_w, img_h,
                        base_scale=0.79, **_):
    """Use the exact prior scale, only solve for translation via conf^2 xcorr."""
    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]

    t_scaled = render_template(template, base_scale, img_w, img_h, seg_w, seg_h)
    tx_seg, ty_seg, _ = confidence_weighted_xcorr(confidence, t_scaled, seg_w, seg_h)

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=base_scale,
    )


# ── 11: NCC (normalised cross-correlation, conf^2) ────────────────────────────

def method_ncc_xcorr(prob_map, template, img_w, img_h,
                     base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """
    Like conf_xcorr_p2 but uses globally-normalised cross-correlation.
    Removes the DC / amplitude bias that can fool raw correlation when the
    confidence map has large bright regions unrelated to the watermark position.
    """
    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        confidence, template,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
        use_ncc=True,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 12: Subpixel conf^2 xcorr ─────────────────────────────────────────────────

def method_subpixel_xcorr(prob_map, template, img_w, img_h,
                           base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """
    conf_xcorr_p2 with parabolic subpixel interpolation of the correlation peak.
    Typically buys 0.5–2 px on the position error at essentially zero extra cost.
    """
    confidence = prob_map ** 2
    seg_h, seg_w = prob_map.shape[:2]

    tx_seg, ty_seg, best_scale_seg = scale_search(
        confidence, template,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
        subpixel=True,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 13: Image-edge NCC ────────────────────────────────────────────────────────

def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Return float32 edge-magnitude image (Sobel), normalised to [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = mag.max()
    return mag / (m + 1e-6)


def method_image_edge_xcorr(prob_map, template, img_w, img_h, wm_bgr=None,
                              base_scale=0.79, scale_range=0.05, n_scales=11, **_):
    """
    Instead of correlating the seg probability map with the template alpha, this
    method correlates the *edge map of the actual image* with the edge map of the
    watermark template.

    Motivation: the watermark's internal edges (text strokes, logo outlines) are
    directly visible in the image and carry much finer positional information than
    the blurry seg model output.  The seg model is only used here to set the search
    window (base_scale prior); the correlation itself is image-driven.

    Works in the same 256-px seg-map space as all other methods, but uses
    image-derived signal instead of model-derived signal.
    """
    seg_h, seg_w = prob_map.shape[:2]

    if wm_bgr is None:
        # Fall back to conf^2 xcorr if image not provided
        return method_conf_xcorr_sq(prob_map, template, img_w, img_h,
                                    base_scale=base_scale,
                                    scale_range=scale_range,
                                    n_scales=n_scales)

    # Resize image to seg-map resolution, convert to grayscale, compute edges
    img_small = cv2.resize(wm_bgr, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_edges = _sobel_mag(gray)

    # Edge map of the template alpha (at native template resolution)
    template_u8 = (template * 255).astype(np.uint8)
    template_gray = template_u8.astype(np.float32) / 255.0
    template_edges = _sobel_mag(template_gray)

    # Weight image edges by seg confidence so we suppress background clutter
    # outside the rough watermark region
    img_edges = img_edges * (prob_map ** 1)  # light confidence weighting

    tx_seg, ty_seg, best_scale_seg = scale_search(
        img_edges, template_edges,
        base_scale, scale_range, n_scales,
        img_w, img_h, seg_w, seg_h,
        use_ncc=True,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ── 14: Two-stage: seg soft-moments → image-edge NCC refinement ───────────────

def method_two_stage_image(prob_map, template, img_w, img_h, wm_bgr=None,
                            scale_range=0.05, n_scales=15, **_):
    """
    Stage 1: soft moments on the seg map → coarse (center, scale).
    Stage 2: image-edge NCC in a tight scale window around that estimate.

    This combines the seg model's ability to identify *which region* the watermark
    is in with the image's own fine-grained edge signal for precise sub-pixel
    localisation.
    """
    if wm_bgr is None:
        return method_two_stage(prob_map, template, img_w, img_h,
                                scale_range=scale_range, n_scales=n_scales)

    # Stage 1: coarse estimate via soft moments
    coarse = method_soft_moments(prob_map, template, img_w, img_h)
    s_coarse = float(np.clip(coarse.scale, 0.5, 1.5))

    # Stage 2: image-edge NCC with tight scale window
    tight_range = min(scale_range * 0.5, 0.025)
    seg_h, seg_w = prob_map.shape[:2]

    img_small = cv2.resize(wm_bgr, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_edges = _sobel_mag(gray)

    template_gray = template.astype(np.float32)
    template_edges = _sobel_mag(template_gray)

    # Light confidence weighting to suppress background clutter
    img_edges = img_edges * (prob_map ** 1)

    tx_seg, ty_seg, best_scale_seg = scale_search(
        img_edges, template_edges,
        s_coarse, tight_range, n_scales,
        img_w, img_h, seg_w, seg_h,
        subpixel=True,
        use_ncc=True,
    )

    return AlignResult(
        tx=tx_seg * img_w / seg_w,
        ty=ty_seg * img_h / seg_h,
        scale=best_scale_seg,
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "oracle":              method_oracle,
    "prior_only":          method_prior_only,
    "centroid_binary":     method_centroid_binary,
    "soft_moments":        method_soft_moments,
    "binary_xcorr":        method_binary_xcorr,
    "conf_xcorr_p1":       method_conf_xcorr_linear,
    "conf_xcorr_p2":       method_conf_xcorr_sq,
    "conf_xcorr_p3":       method_conf_xcorr_cubic,
    "erode_xcorr":         method_erode_xcorr,
    "two_stage":           method_two_stage,
    "prior_xcorr":         method_prior_xcorr,
    # ── new methods ──────────────────────────────────────────────────────────
    "ncc_xcorr":           method_ncc_xcorr,           # NCC instead of raw corr
    "subpixel_xcorr":      method_subpixel_xcorr,      # parabolic subpixel peak
    "image_edge_xcorr":    method_image_edge_xcorr,    # image edges as signal
    "two_stage_image":     method_two_stage_image,     # seg coarse + image-edge NCC
}


def apply_spatial_prior(prob_map: np.ndarray) -> np.ndarray:
    """Apply a Tukey window to suppress edge artifacts from the model. 
    A Tukey window is flat (1.0) in the middle and smoothly tapers to 0 at the edges."""
    from scipy.signal.windows import tukey
    h, w = prob_map.shape[:2]
    # alpha=0.15 means 7.5% of each edge is a cosine taper to 0, 85% is exactly 1.0
    win_y = tukey(h, alpha=0.15).astype(np.float32)
    win_x = tukey(w, alpha=0.15).astype(np.float32)
    window = win_y[:, None] * win_x[None, :]
    return prob_map * window


# ---------------------------------------------------------------------------
# Segmentation model inference
# ---------------------------------------------------------------------------

def build_prob_map(seg_model, wm_bgr: np.ndarray, seg_size: int, device) -> np.ndarray:
    """Run segmentation model → return soft probability map (HxW float32 [0,1])."""
    import torch
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    wm_r = cv2.resize(wm_bgr, (seg_size, seg_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    inp = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(seg_model(inp)).squeeze().cpu().numpy()
    return prob.astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample_dir: Path,
    wm_rgba: np.ndarray,
    template: np.ndarray,       # HxW float32 alpha at native resolution
    seg_model,
    seg_size: int,
    device,
    use_gt_mask: bool,
    spatial_prior: bool,
    methods: dict,
) -> Optional[dict]:
    """Run all methods on one sample and return their metrics."""
    wm_bgr = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
    gt_mask_u8 = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if wm_bgr is None or gt_mask_u8 is None:
        return None

    meta = load_meta(sample_dir)
    gt_x, gt_y = meta["position"]  # top-left in original image pixels
    img_h, img_w = wm_bgr.shape[:2]

    # Ground-truth scale from bounding box of mask
    gt_mask_bin = (gt_mask_u8 > 128).astype(np.uint8)
    gt_scale = gt_scale_from_mask(gt_mask_bin, template, img_w)

    # Probability map
    if use_gt_mask:
        # Perfect oracle prob map = ground-truth mask rescaled to float
        prob_map = cv2.resize(gt_mask_u8, (256, 256),
                              interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    else:
        prob_map = build_prob_map(seg_model, wm_bgr, seg_size, device)

    if spatial_prior:
        prob_map = apply_spatial_prior(prob_map)

    results = {}
    kwargs = dict(
        prob_map=prob_map,
        template=template,
        img_w=img_w,
        img_h=img_h,
        gt_x=gt_x,
        gt_y=gt_y,
        gt_scale=gt_scale,
        wm_bgr=wm_bgr,   # passed to image-based methods; ignored via **_ by others
    )

    for name, fn in methods.items():
        result = fn(**kwargs)

        # Position error in original image pixels
        pos_err = float(np.hypot(result.tx - gt_x, result.ty - gt_y))

        # Scale error (absolute difference in fraction-of-width)
        scale_err = abs(result.scale - gt_scale)

        # Aligned IoU
        placed = place_template(template, result.tx, result.ty, result.scale, img_w, img_h)
        iou = iou_masks(placed, gt_mask_bin)

        results[name] = SampleMetrics(pos_err=pos_err, scale_err=scale_err, iou=iou)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate(metrics_list: List[SampleMetrics]) -> dict:
    pos_errs  = [m.pos_err   for m in metrics_list]
    scale_errs = [m.scale_err for m in metrics_list]
    ious       = [m.iou       for m in metrics_list]
    return {
        "pos_err_mean":   float(np.mean(pos_errs)),
        "pos_err_median": float(np.median(pos_errs)),
        "pos_err_p90":    float(np.percentile(pos_errs, 90)),
        "scale_err_mean": float(np.mean(scale_errs)),
        "iou_mean":       float(np.mean(ious)),
        "iou_median":     float(np.median(ious)),
        "iou_p10":        float(np.percentile(ious, 10)),  # worst 10%
        "n":              len(metrics_list),
    }


def print_report(all_metrics: dict):
    col_w = 18
    cols_order = [
        ("pos_err_mean",   "PosErr--(mean)"),
        ("pos_err_median", "PosErr--(med) "),
        ("pos_err_p90",    "PosErr--(P90) "),
        ("scale_err_mean", "ScaleErr--    "),
        ("iou_mean",       "IoU++(mean)   "),
        ("iou_median",     "IoU++(med)    "),
        ("iou_p10",        "IoU++(P10)    "),
    ]

    # Header
    header = f"{'Method':<{col_w}}"
    for _, label in cols_order:
        header += f"  {label:>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Sort by mean IoU descending
    sorted_methods = sorted(all_metrics.items(), key=lambda kv: kv[1]["iou_mean"], reverse=True)
    for name, agg in sorted_methods:
        values = [agg[key] for key, _ in cols_order]
        row = f"{name:<{col_w}}"
        for v, (key, _) in zip(values, cols_order):
            if "iou" in key:
                row += f"  {v:>14.4f}"
            elif "scale" in key:
                row += f"  {v:>14.5f}"
            else:
                row += f"  {v:>14.2f}"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure stdout can handle any character on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Watermark alignment algorithm benchmark")
    parser.add_argument("--seg-checkpoint", default=None,
                        help="Segmentation model checkpoint (omit if --no-model)")
    parser.add_argument("--seg-config", default="configs/seg.yaml")
    parser.add_argument("--watermark", default="../data_gen/watermark.png",
                        help="Path to the original watermark RGBA PNG")
    parser.add_argument("--dataset", default="../data_gen/dataset",
                        help="Dataset root directory")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--no-model", action="store_true",
                        help="Use ground-truth mask as prob_map (oracle noise test)")
    parser.add_argument("--apply-spatial-prior", action="store_true",
                        help="Multiply prob_map by a Tukey window to suppress edge artifacts")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of methods to run (default: all). "
                             f"Available: {list(METHODS.keys())}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load watermark template
    wm_rgba = cv2.imread(args.watermark, cv2.IMREAD_UNCHANGED)
    if wm_rgba is None:
        raise FileNotFoundError(f"Cannot read watermark: {args.watermark}")
    template = wm_rgba[:, :, 3].astype(np.float32) / 255.0  # alpha channel

    # Collect samples
    ds_root = Path(args.dataset)
    sample_dirs = sorted([
        d for d in ds_root.iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "mask.png").exists()
        and (d / "meta.json").exists()
    ])
    if not sample_dirs:
        raise RuntimeError(f"No valid samples found in {ds_root}")

    n = len(sample_dirs) if args.n_samples == 0 else min(args.n_samples, len(sample_dirs))
    sample_dirs = random.sample(sample_dirs, n)
    print(f"Evaluating {n} samples from {ds_root}")

    # Load segmentation model if needed
    seg_model = None
    device = None
    seg_size = 256
    if not args.no_model:
        import torch
        from wm_shared.config import load_yaml_config
        from training.src.tasks.segmentation.model import build_seg_model

        if args.seg_checkpoint is None:
            parser.error("--seg-checkpoint is required unless --no-model is set")

        seg_cfg = load_yaml_config(args.seg_config)
        seg_size = seg_cfg["dataset"]["image_size"]

        if seg_cfg.get("device", "cpu") == "auto":
            seg_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(seg_cfg["device"])

        seg_model = build_seg_model(seg_cfg)
        ckpt = torch.load(args.seg_checkpoint, map_location="cpu", weights_only=True)
        seg_model.load_state_dict(ckpt["model"])
        seg_model.to(device).eval()
        print(f"Loaded seg model from {args.seg_checkpoint}  (device={device})")
    else:
        print("Using ground-truth masks as probability maps (oracle test)")

    # Select methods
    methods_to_run = {}
    if args.methods:
        for m in args.methods:
            if m not in METHODS:
                parser.error(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            methods_to_run[m] = METHODS[m]
    else:
        methods_to_run = dict(METHODS)
    # Always exclude oracle when running with model (it would be cheating in the table)
    if not args.no_model and "oracle" in methods_to_run and len(methods_to_run) > 1:
        pass  # keep oracle as reference upper-bound column

    print(f"Methods: {list(methods_to_run.keys())}\n")

    # Evaluation
    all_sample_metrics: dict[str, list] = {name: [] for name in methods_to_run}
    t0 = time.time()

    for i, sample_dir in enumerate(sample_dirs):
        sample_results = evaluate_sample(
            sample_dir=sample_dir,
            wm_rgba=wm_rgba,
            template=template,
            seg_model=seg_model,
            seg_size=seg_size,
            device=device,
            use_gt_mask=args.no_model,
            spatial_prior=args.apply_spatial_prior,
            methods=methods_to_run,
        )
        if sample_results is None:
            continue

        for name, m in sample_results.items():
            all_sample_metrics[name].append(m)

        if (i + 1) % 20 == 0 or i == len(sample_dirs) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}]  {elapsed:.1f}s elapsed", flush=True)

    # Aggregate and report
    agg = {}
    for name, metrics_list in all_sample_metrics.items():
        if metrics_list:
            agg[name] = aggregate(metrics_list)

    print_report(agg)

    # Per-method timing note
    print(f"\nTotal wall time: {time.time() - t0:.1f}s  ({n} samples)")
    print("\nMetric key:")
    print("  PosErr   = Euclidean distance from GT top-left corner (original image pixels)")
    print("  ScaleErr = |predicted_scale - gt_scale| (fraction of image width)")
    print("  IoU      = Intersection-over-Union of placed template vs GT binary mask")


if __name__ == "__main__":
    main()
