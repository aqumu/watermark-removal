"""
Loss functions for watermark removal
-------------------------------------
CombinedLoss = λ_l1_masked        × L1(pred ⊙ mask_interior, target ⊙ mask_interior)
             + λ_perceptual       × Perceptual(pred, target)   [VGG16 4-level features]
             + λ_border           × BorderRing(pred, target, soft_mask)
             + λ_color_moment     × ColorMoment(pred, target, mask_interior)
             + λ_bg_tv            × BackgroundTV(delta, mask)
             + λ_interior_tv      × InteriorTV(delta, mask_interior)
             + λ_bg_delta         × BackgroundDelta(delta, mask)
             + λ_saturation       × SaturationLoss(pred, target, mask_interior)

All tensors are in [-1, 1] (normalised RGB).
The perceptual loss remaps to [0, 1] internally before passing to VGG.

Each term has a distinct, non-overlapping spatial responsibility:
  - l1_masked         : fully inside the watermark (eroded mask interior)
  - saturation        : HSV saturation + value recovery inside mask
                        (white overlay specifically desaturates colours)
  - border            : feathered edge transition zone  [4·m·(1−m) weight]
  - bg_tv             : total variation of delta outside mask
  - interior_tv       : total variation of delta inside mask
  - bg_delta          : L1 penalty on delta magnitude outside mask
  - perceptual        : global texture / mid-frequency fidelity (4 VGG levels)
  - color_moment      : per-channel mean in the restored region
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ──────────────────────────────────────────────────────────────────────────────
# VGG perceptual loss — 4 feature levels
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss using four VGG16 activation levels:
      relu1_2 (fine edges/details)  weight 0.5
      relu2_2 (texture)             weight 1.0
      relu3_3 (structure)           weight 1.0
      relu4_3 (semantics, light)    weight 0.05

    Adding relu1_2 vs the old relu2_2+relu3_3 pair specifically targets fine
    texture fidelity inside the watermark, which is the primary detail-preservation
    challenge for a low-opacity (0.1–0.2) white overlay.

    All VGG weights are frozen; the model stays in eval mode.
    """
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    mean: torch.Tensor
    std: torch.Tensor

    # Relative weights per feature level
    _LEVEL_WEIGHTS = (0.5, 1.0, 1.0, 0.05)

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        feats = vgg.features

        # VGG16 feature indices:
        #   0-3  : conv1_1 relu1_1 conv1_2 relu1_2
        #   4-8  : pool1 conv2_1 relu2_1 conv2_2 relu2_2
        #   9-15 : pool2 conv3_1 relu3_1 conv3_2 relu3_2 conv3_3 relu3_3
        #  16-22 : pool3 conv4_1 relu4_1 conv4_2 relu4_2 conv4_3 relu4_3
        #
        # Slices are complementary segments so we can chain them in forward
        # without re-running earlier layers (one forward pass through VGG).
        self.slice0 = nn.Sequential(*list(feats[0:4]))    # → relu1_2
        self.slice1 = nn.Sequential(*list(feats[4:9]))    # pool1 → relu2_2
        self.slice2 = nn.Sequential(*list(feats[9:16]))   # pool2 → relu3_3
        self.slice3 = nn.Sequential(*list(feats[16:23]))  # pool3 → relu4_3

        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD ).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    # VGG was designed for 224×224; running it at full training resolution is
    # expensive on CPU. Downscale inputs here — perceptual features are
    # low-frequency enough that 224 captures everything that matters.
    _VGG_SIZE = 224

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Remap [-1,1] → ImageNet-normalised, resized to _VGG_SIZE."""
        x = (x + 1) / 2                          # [0, 1]
        if x.shape[-1] != self._VGG_SIZE:
            x = F.interpolate(x, size=self._VGG_SIZE, mode="area")
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._normalise(pred)
        t = self._normalise(target)

        loss = pred.new_zeros(1).squeeze()
        for sl, w in zip((self.slice0, self.slice1, self.slice2, self.slice3),
                         self._LEVEL_WEIGHTS):
            p = sl(p)
            t = sl(t)
            loss = loss + w * F.mse_loss(p, t)

        return loss


# ──────────────────────────────────────────────────────────────────────────────
# colour-moment loss
# ──────────────────────────────────────────────────────────────────────────────

def color_moment_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """
    Match the per-channel mean of the predicted masked region to the target.

    This is the minimal constraint needed to prevent overall colour drift
    (e.g. the restored area coming out too bright or tinted) without adding
    any real compute cost — it's just a masked mean difference.

    pred, target : Bx3xHxW  [-1,1]
    mask         : Bx1xHxW  {0,1}
    """
    n = mask.sum(dim=(2, 3)).clamp(min=1)          # Bx1
    pred_mean   = (pred   * mask).sum(dim=(2, 3)) / n   # Bx3
    target_mean = (target * mask).sum(dim=(2, 3)) / n   # Bx3
    return F.l1_loss(pred_mean, target_mean)


# ──────────────────────────────────────────────────────────────────────────────
# saturation recovery loss
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_to_sv(rgb01: torch.Tensor):
    """
    Compute HSV saturation and value channels from an RGB tensor in [0, 1].

    Pure PyTorch — no OpenCV dependency.  Hue is intentionally excluded:
      - A white overlay (R=G=B) does not shift hue, only S and V.
      - Hue is numerically unstable for near-achromatic pixels (denominator → 0).
    Penalising hue would add noise without improving colour accuracy.

    rgb01 : Bx3xHxW in [0, 1]
    returns (S, V) each Bx1xHxW in [0, 1]
    Gradient stability notes:
      - The denominator (max_c) is detached: prevents ∂S/∂V from fighting
        the brightness correction that l1_masked is already handling.
      - Pixels with V < 0.05 (~13 on 0-255) are excluded: saturation is
        numerically meaningless there and the large delta is already
        handled by l1_masked. Without this, ∂S/∂RGB → 1e8 on dark pixels.
    """
    max_c, _ = rgb01.max(dim=1, keepdim=True)   # V  =  max(R,G,B)
    min_c, _ = rgb01.min(dim=1, keepdim=True)
    diff      = max_c - min_c
    # Detach denominator: gradient flows only through numerator (chroma span).
    # Threshold 0.05: exclude near-black pixels where S is ill-defined.
    denom = max_c.detach().clamp(min=0.05)
    s = torch.where(max_c > 0.05, diff / denom, torch.zeros_like(max_c))
    return s, max_c   # saturation, value


class SaturationLoss(nn.Module):
    """
    Penalises saturation and value error inside the watermark mask.

    A white overlay pulls all pixels toward white, desaturating colours and
    increasing brightness.  Standard pixel-space L1 loss treats all channels
    equally and can tolerate systematic saturation errors if the mean channel
    values are close.  This term directly measures and penalises the desaturation
    that remains after the model's correction.

    Saturation weight 1.0, Value weight 0.5:
      - Saturation is the primary problem (white overlay always desaturates).
      - Value is secondary (brightness recovery is also constrained by l1_masked).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        pred, target : Bx3xHxW  [-1, 1]
        mask         : Bx1xHxW  [0, 1]  (eroded interior)
        """
        p01 = (pred   + 1.0) * 0.5
        t01 = (target + 1.0) * 0.5

        p_s, p_v = _rgb_to_sv(p01)
        t_s, t_v = _rgb_to_sv(t01)

        n = mask.sum().clamp(min=1e-6)
        sat_loss = ((p_s - t_s).abs() * mask).sum() / n
        val_loss = ((p_v - t_v).abs() * mask).sum() / n
        return sat_loss + 0.5 * val_loss

# ──────────────────────────────────────────────────────────────────────────────
# border ring loss
# ──────────────────────────────────────────────────────────────────────────────

def border_ring_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     soft_mask: torch.Tensor) -> torch.Tensor:
    """
    Extra penalty on the feathered edge of the watermark.

    Weight = 4 × mask × (1 − mask).  This function is zero at mask=0 (clean
    background) and mask=1 (fully inside the watermark), and peaks at mask=0.5
    (the midpoint of the transition zone) — exactly where shadow artefacts form.

    With blur_pct=8.0 the loss mask has a wide (≈80px at 512px) soft gradient
    zone, so this loss fires over a large ring around the watermark boundary.
    This is essential for the "no visible edges" goal on an 80%-width watermark.

    pred, target : Bx3xHxW  [-1,1]
    soft_mask    : Bx1xHxW  [0,1]  (Gaussian-blurred loss mask)
    """
    weight = 4.0 * soft_mask * (1.0 - soft_mask)   # Bx1xHxW, peak = 1.0
    n = weight.sum().clamp(min=1e-6) * pred.shape[1]
    return ((pred - target).abs() * weight).sum() / n


# ──────────────────────────────────────────────────────────────────────────────
# background and interior TV and delta regularisers
# ──────────────────────────────────────────────────────────────────────────────

def interior_tv_loss(delta: torch.Tensor,
                     mask_interior: torch.Tensor) -> torch.Tensor:
    """
    Total variation of the raw model delta inside the eroded mask.

    Penalises high-frequency spatial variation (grids, dots, stripes) in the
    delta inside the watermark mask. Suppresses VGG checkerboard artifacts.

    delta         : Bx3xHxW  (raw model output, before subtraction from input)
    mask_interior : Bx1xHxW  [0,1]
    """
    d = delta * mask_interior                              # zero outside mask
    tv_h = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean()
    tv_w = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def background_tv_loss(delta: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """
    Total variation of the raw model delta in background regions.

    delta : Bx3xHxW
    mask  : Bx1xHxW  [0,1]
    """
    bg = 1.0 - mask                                        # Bx1xHxW
    d = delta * bg                                         # zero inside mask
    tv_h = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean()
    tv_w = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def background_delta_penalty(delta: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """
    L1 penalty on delta magnitude outside the mask.

    The model should predict delta = 0 exactly in background regions.
    L1 (unlike L2) strictly enforces sparsity, brutally zeroing out tiny 
    fuzz or checkerboard artifacts that cause visible seams.

    delta : Bx3xHxW
    mask  : Bx1xHxW  [0,1]
    """
    bg = 1.0 - mask
    return (delta * bg).abs().mean()


# ──────────────────────────────────────────────────────────────────────────────
# edge-selective refinement losses
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(value: torch.Tensor,
                   weight: torch.Tensor,
                   channels: int | None = None,
                   eps: float = 1e-6) -> torch.Tensor:
    """Weighted mean with stable denominator and optional channel expansion."""
    if channels is not None and value.shape[1] != weight.shape[1]:
        weight = weight.expand(-1, channels, -1, -1)
    denom = weight.sum().clamp(min=eps)
    return (value * weight).sum() / denom


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (Bx3xHxW) to luminance (Bx1xHxW)."""
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise Sobel gradients using depthwise grouped convolution."""
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def edge_gradient_loss(pred: torch.Tensor,
                       target: torch.Tensor,
                       soft_mask: torch.Tensor) -> torch.Tensor:
    """
    Gradient-domain L1 inside the feathered edge band (4*m*(1-m)).
    Strongly penalises coherent leftover edge traces.
    """
    weight = 4.0 * soft_mask * (1.0 - soft_mask)
    pgx, pgy = _sobel_xy(pred)
    tgx, tgy = _sobel_xy(target)
    grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()
    return _weighted_mean(grad_err, weight, channels=pred.shape[1])


def edge_laplacian_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        soft_mask: torch.Tensor) -> torch.Tensor:
    """
    Laplacian mismatch in the edge band.
    Complements gradient loss by targeting ultra-thin halo/rim residues.
    """
    weight = 4.0 * soft_mask * (1.0 - soft_mask)
    lap_k = pred.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=pred.dtype).view(1, 1, 3, 3)
    c = pred.shape[1]
    lap_k = lap_k.repeat(c, 1, 1, 1)
    p_lap = F.conv2d(pred, lap_k, padding=1, groups=c)
    t_lap = F.conv2d(target, lap_k, padding=1, groups=c)
    return _weighted_mean((p_lap - t_lap).abs(), weight, channels=pred.shape[1])


def edge_coherence_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        soft_mask: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    Penalise coherent line-like residual structure in the edge band.

    Residual letter fragments produce high directional coherence in local
    gradients; random pixel noise is isotropic and scores much lower.
    """
    weight = 4.0 * soft_mask * (1.0 - soft_mask)
    residual_gray = _rgb_to_gray(pred - target)
    rx, ry = _sobel_xy(residual_gray)

    # Local structure tensor terms (windowed with 3x3 average pooling).
    j11 = F.avg_pool2d(rx * rx, kernel_size=3, stride=1, padding=1)
    j22 = F.avg_pool2d(ry * ry, kernel_size=3, stride=1, padding=1)
    j12 = F.avg_pool2d(rx * ry, kernel_size=3, stride=1, padding=1)

    trace = j11 + j22
    delta = torch.sqrt((j11 - j22) ** 2 + 4.0 * (j12 ** 2) + eps)
    coherence = delta / (trace + eps)  # [0,1], high for line-like structures
    grad_energy = torch.sqrt(rx * rx + ry * ry + eps)

    return _weighted_mean(coherence * grad_energy, weight)


def drift_anchor_loss(pred: torch.Tensor,
                      ref_pred: torch.Tensor,
                      soft_mask: torch.Tensor,
                      anchor_scale: float = 2.0) -> torch.Tensor:
    """
    Anchor new predictions to a frozen reference model outside the edge band.
    Keeps already-good colour/texture behaviour unchanged while allowing local
    edits near watermark boundaries.
    """
    edge_weight = 4.0 * soft_mask * (1.0 - soft_mask)
    anchor_weight = torch.clamp(1.0 - anchor_scale * edge_weight, min=0.0, max=1.0)
    return _weighted_mean((pred - ref_pred).abs(), anchor_weight, channels=pred.shape[1])


# ──────────────────────────────────────────────────────────────────────────────
# combined loss
# ──────────────────────────────────────────────────────────────────────────────

def _parse_weight(v) -> tuple[float, float]:
    """Parse a loss weight from config. Scalar → fixed; [start, end] → ramp."""
    if isinstance(v, (list, tuple)):
        return (float(v[0]), float(v[1]))
    return (float(v), float(v))


def _lerp_weight(w: tuple[float, float], progress: float) -> float:
    """Linearly interpolate between w[0] and w[1] based on progress (0→1)."""
    return w[0] + (w[1] - w[0]) * progress


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]

        # Each weight is stored as (start, end) for ramping support.
        # Scalar config values become (v, v) — no ramp, constant weight.
        self._w_l1_masked         = _parse_weight(lc["l1_masked"])
        self._w_perceptual        = _parse_weight(lc["perceptual"])
        self._w_color_moment      = _parse_weight(lc.get("color_moment",       0.0))
        self._w_border            = _parse_weight(lc.get("border",             0.0))
        self._w_bg_tv             = _parse_weight(lc.get("bg_tv",              0.0))
        self._w_interior_tv       = _parse_weight(lc.get("interior_tv",        0.0))
        self._w_bg_delta          = _parse_weight(lc.get("bg_delta",           0.0))
        self._w_saturation        = _parse_weight(lc.get("saturation",         0.0))
        self._w_edge_grad         = _parse_weight(lc.get("edge_grad",          0.0))
        self._w_edge_coherence    = _parse_weight(lc.get("edge_coherence",     0.0))
        self._w_edge_laplacian    = _parse_weight(lc.get("edge_laplacian",     0.0))
        self._w_drift             = _parse_weight(lc.get("drift",              0.0))

        # Erosion kernel size for computing mask_interior.
        # Scale with image resolution: 3px at 256 → 7px at 512 preserves the
        # same proportional interior margin (~1.2% of image width).
        self._erosion_kernel = lc.get("erosion_kernel", 3)
        self._drift_anchor_scale = float(lc.get("drift_anchor_scale", 2.0))

        # Current effective weights (updated by set_progress)
        self._progress = 0.0
        self._update_weights()

        self.perceptual = (PerceptualLoss()
                           if max(self._w_perceptual) > 0 else None)
        self.saturation = (SaturationLoss()
                           if max(self._w_saturation) > 0 else None)

    def _update_weights(self):
        t = self._progress
        self.w_l1_masked         = _lerp_weight(self._w_l1_masked,         t)
        self.w_perceptual        = _lerp_weight(self._w_perceptual,        t)
        self.w_color_moment      = _lerp_weight(self._w_color_moment,      t)
        self.w_border            = _lerp_weight(self._w_border,            t)
        self.w_bg_tv             = _lerp_weight(self._w_bg_tv,             t)
        self.w_interior_tv       = _lerp_weight(self._w_interior_tv,       t)
        self.w_bg_delta          = _lerp_weight(self._w_bg_delta,          t)
        self.w_saturation        = _lerp_weight(self._w_saturation,        t)
        self.w_edge_grad         = _lerp_weight(self._w_edge_grad,         t)
        self.w_edge_coherence    = _lerp_weight(self._w_edge_coherence,    t)
        self.w_edge_laplacian    = _lerp_weight(self._w_edge_laplacian,    t)
        self.w_drift             = _lerp_weight(self._w_drift,             t)

    def set_progress(self, progress: float):
        """Set training progress (0.0 = start, 1.0 = end) to interpolate ramped weights."""
        self._progress = max(0.0, min(1.0, progress))
        self._update_weights()

    def forward(self,
                pred:   torch.Tensor,           # Bx3xHxW, [-1,1]
                target: torch.Tensor,           # Bx3xHxW, [-1,1]
                mask:   torch.Tensor,           # Bx1xHxW, [0,1]
                delta:  torch.Tensor | None = None,   # Bx3xHxW, raw model output
                wm:     torch.Tensor | None = None,   # Bx3xHxW, watermarked input [-1,1]
                ref_pred: torch.Tensor | None = None, # Bx3xHxW, frozen reference model output
                use_perceptual: bool = True,
                ) -> tuple[torch.Tensor, dict]:

        # ── erode mask for interior-only losses ───────────────────────────
        # Removes the outer ring so l1_masked, saturation, and gradient
        # fidelity focus only on clearly interior pixels where the watermark
        # signal is well above the JPEG noise floor.
        # erosion_kernel scales with training resolution (7 at 512px).
        k = self._erosion_kernel
        mask_interior = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)

        # ── masked L1 (interior pixels only) ──────────────────────────────
        masked_pred   = pred   * mask_interior
        masked_target = target * mask_interior
        n_masked = mask_interior.sum().clamp(min=1) * 3  # channels
        l1_masked = (masked_pred - masked_target).abs().sum() / n_masked

        # ── perceptual (4-level VGG) ───────────────────────────────────────
        perc_loss = (self.perceptual(pred, target)
                     if (self.perceptual is not None and use_perceptual)
                     else pred.new_zeros(1).squeeze())

        # ── colour moment (interior only) ──────────────────────────────────
        cm_loss = (color_moment_loss(pred, target, mask_interior)
                   if self.w_color_moment > 0
                   else pred.new_zeros(1).squeeze())

        # ── saturation recovery (interior only) ───────────────────────────
        # White overlay desaturates; this term directly targets HSV S + V recovery.
        sat_loss = (self.saturation(pred, target, mask_interior)
                    if (self.saturation is not None and self.w_saturation > 0)
                    else pred.new_zeros(1).squeeze())

        # ── border ring (feathered edge transition zone) ──────────────────
        b_loss = (border_ring_loss(pred, target, mask)
                  if self.w_border > 0
                  else pred.new_zeros(1).squeeze())

        # ── background delta regularisers ────────────────────────────────
        bg_tv_loss = (background_tv_loss(delta, mask)
                      if (self.w_bg_tv > 0 and delta is not None)
                      else pred.new_zeros(1).squeeze())
                      
        int_tv_loss = (interior_tv_loss(delta, mask_interior)
                      if (self.w_interior_tv > 0 and delta is not None)
                      else pred.new_zeros(1).squeeze())

        bg_delta_loss = (background_delta_penalty(delta, mask)
                         if (self.w_bg_delta > 0 and delta is not None)
                         else pred.new_zeros(1).squeeze())

        edge_grad = (edge_gradient_loss(pred, target, mask)
                     if self.w_edge_grad > 0
                     else pred.new_zeros(1).squeeze())

        edge_coh = (edge_coherence_loss(pred, target, mask)
                    if self.w_edge_coherence > 0
                    else pred.new_zeros(1).squeeze())

        edge_lap = (edge_laplacian_loss(pred, target, mask)
                    if self.w_edge_laplacian > 0
                    else pred.new_zeros(1).squeeze())

        drift_loss = (drift_anchor_loss(pred, ref_pred, mask, self._drift_anchor_scale)
                      if (self.w_drift > 0 and ref_pred is not None)
                      else pred.new_zeros(1).squeeze())

        total = (self.w_l1_masked         * l1_masked
               + self.w_perceptual        * perc_loss
               + self.w_color_moment      * cm_loss
               + self.w_saturation        * sat_loss
               + self.w_border            * b_loss
               + self.w_bg_tv             * bg_tv_loss
               + self.w_interior_tv       * int_tv_loss
               + self.w_bg_delta          * bg_delta_loss
               + self.w_edge_grad         * edge_grad
               + self.w_edge_coherence    * edge_coh
               + self.w_edge_laplacian    * edge_lap
               + self.w_drift             * drift_loss)

        breakdown = {
            "l1_masked":          l1_masked.item(),
            "perceptual":         perc_loss.item() if self.perceptual else 0.0,
            "color_moment":       cm_loss.item(),
            "saturation":         sat_loss.item(),
            "border":             b_loss.item(),
            "bg_tv":              bg_tv_loss.item(),
            "interior_tv":        int_tv_loss.item(),
            "bg_delta":           bg_delta_loss.item(),
            "edge_grad":          edge_grad.item(),
            "edge_coherence":     edge_coh.item(),
            "edge_laplacian":     edge_lap.item(),
            "drift":              drift_loss.item(),
            "total":              total.item(),
            # weights for the live_plot / CSV
            "weight_l1_masked":         self.w_l1_masked,
            "weight_perceptual":        self.w_perceptual,
            "weight_color_moment":      self.w_color_moment,
            "weight_saturation":        self.w_saturation,
            "weight_border":            self.w_border,
            "weight_bg_tv":             self.w_bg_tv,
            "weight_interior_tv":       self.w_interior_tv,
            "weight_bg_delta":          self.w_bg_delta,
            "weight_edge_grad":         self.w_edge_grad,
            "weight_edge_coherence":    self.w_edge_coherence,
            "weight_edge_laplacian":    self.w_edge_laplacian,
            "weight_drift":             self.w_drift,
        }
        return total, breakdown
