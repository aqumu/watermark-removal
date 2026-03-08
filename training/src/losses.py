"""
Loss functions for watermark removal
-------------------------------------
CombinedLoss = λ_l1_full      × L1(pred, target)
             + λ_l1_masked    × L1(pred ⊙ soft_mask, target ⊙ soft_mask)
             + λ_perceptual   × Perceptual(pred, target)   [VGG16 features]
             + λ_ssim         × (1 - SSIM(pred, target))
             + λ_border       × BorderRing(pred, target, soft_mask)

All tensors are in [-1, 1] (normalised RGB).
The perceptual loss remaps to [0, 1] internally before passing to VGG.

Design choices to preserve lighting / colour profiles
  - Full-image L1 keeps the unmasked region pixel-perfect.
  - Soft-mask-weighted L1 focuses the model on the watermark region; the
    continuous mask naturally up-weights the feathered transition zone.
  - Border ring loss applies the highest per-pixel penalty exactly at the
    alpha transition (weight = 4·mask·(1−mask), peaks at mask=0.5).
  - Perceptual loss preserves texture and mid-frequency detail.
  - SSIM guards structural consistency at the mask border.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ──────────────────────────────────────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] @ g[None, :]        # HxW
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel_4d                             # CxCxHxW (depthwise)


class SSIMLoss(nn.Module):
    def __init__(self, channels: int = 3, window_size: int = 11):
        super().__init__()
        self.channels = channels
        kernel = _gaussian_kernel(window_size, sigma=1.5, channels=channels)
        self.register_buffer("kernel", kernel)
        self.pad = window_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # remap [-1,1] → [0,1]
        p = (pred   + 1) / 2
        t = (target + 1) / 2

        k = self.kernel.to(p.device)
        mu_p  = F.conv2d(p, k, padding=self.pad, groups=self.channels)
        mu_t  = F.conv2d(t, k, padding=self.pad, groups=self.channels)

        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_pp = F.conv2d(p * p, k, padding=self.pad, groups=self.channels) - mu_pp
        sig_tt = F.conv2d(t * t, k, padding=self.pad, groups=self.channels) - mu_tt
        sig_pt = F.conv2d(p * t, k, padding=self.pad, groups=self.channels) - mu_pt

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / \
                   ((mu_pp + mu_tt + C1) * (sig_pp + sig_tt + C2))
        return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────────────────────────────────────
# VGG perceptual loss
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Uses relu2_2 and relu3_3 features from VGG16.
    Weights are frozen; the model is kept in eval mode.
    """
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        features = vgg.features

        # relu2_2 → up to index 9 (inclusive)
        # relu3_3 → up to index 16 (inclusive)
        self.slice1 = nn.Sequential(*list(features[:10]))   # relu2_2
        self.slice2 = nn.Sequential(*list(features[10:17])) # relu3_3

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
            x = F.interpolate(x, size=self._VGG_SIZE, mode="bilinear",
                              align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._normalise(pred)
        t = self._normalise(target)

        # slice1
        fp1 = self.slice1(p)
        ft1 = self.slice1(t)
        loss = F.l1_loss(fp1, ft1)

        # slice2
        fp2 = self.slice2(fp1)
        ft2 = self.slice2(ft1)
        loss = loss + F.l1_loss(fp2, ft2)

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

    With old binary masks {0, 1} the weight is always 0, so this loss is a
    no-op on existing datasets (fully backward-compatible).

    pred, target : Bx3xHxW  [-1,1]
    soft_mask    : Bx1xHxW  [0,1]
    """
    weight = 4.0 * soft_mask * (1.0 - soft_mask)   # Bx1xHxW, peak = 1.0
    n = weight.sum().clamp(min=1e-6) * pred.shape[1]
    return ((pred - target).abs() * weight).sum() / n


# ──────────────────────────────────────────────────────────────────────────────
# edge ring loss
# ──────────────────────────────────────────────────────────────────────────────

def edge_ring_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor,
                   ring_px: int = 2) -> torch.Tensor:
    """
    High-weight L1 on the outermost `ring_px` pixels of the binary mask.

    The ring is computed as:  binary_mask − erode(binary_mask, ring_px)
    These are pixels that ARE watermarked (mask=1, so correction is expected)
    but sit at the very edge of the mask where the model consistently
    under-corrects due to mixed inside/outside context in its receptive field.

    Unlike border_ring_loss (which uses a smooth 4·m·(1−m) weighting and
    fires across the blurred-mask transition zone), this loss is binary:
    every pixel in the ring receives equal, unweighted L1 penalty.  This
    prevents the loss from being diluted across the wide transition zone.

    ring_px = 0 disables the loss (returns zero).

    pred, target : Bx3xHxW  [-1,1]
    mask         : Bx1xHxW  [0,1]  (blurred loss mask from dataset)
    ring_px      : erosion radius in pixels (integer kernel half-width)
    """
    if ring_px <= 0:
        return pred.new_zeros(1).squeeze()

    # Binarise the mask (blurred, so threshold at 0.5 gives the true footprint)
    mask_bin = (mask >= 0.5).to(dtype=pred.dtype)

    # Erode by ring_px pixels using max-pool on the negated mask
    k = 2 * ring_px + 1
    mask_eroded = -F.max_pool2d(-mask_bin, kernel_size=k, stride=1, padding=ring_px)

    # Ring = outermost ring_px pixels that are still inside the mask
    ring = mask_bin - mask_eroded  # {0, 1}

    n = ring.sum().clamp(min=1e-6) * pred.shape[1]  # pixel × channel count
    return ((pred - target).abs() * ring).sum() / n


# ──────────────────────────────────────────────────────────────────────────────
# combined loss
# ──────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]
        self.w_l1_full     = lc["l1_full"]
        self.w_l1_masked   = lc["l1_masked"]
        self.w_perceptual  = lc["perceptual"]
        self.w_ssim        = lc["ssim"]
        self.w_color_moment  = lc.get("color_moment", 0.0)
        self.w_border        = lc.get("border", 0.0)
        self.w_edge_ring     = lc.get("edge_ring", 0.0)
        self.edge_ring_px    = int(lc.get("edge_ring_px", 2))

        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss() if self.w_perceptual > 0 else None

    def forward(self,
                pred:   torch.Tensor,  # Bx3xHxW, [-1,1]
                target: torch.Tensor,  # Bx3xHxW, [-1,1]
                mask:   torch.Tensor,  # Bx1xHxW, {0,1}
                use_perceptual: bool = True,
                ) -> tuple[torch.Tensor, dict]:

        # ── erode mask for interior-only losses ───────────────────────────
        # Removes the outer ~3px ring so l1_masked and color_moment focus
        # only on clearly interior pixels where the watermark signal is
        # well above the JPEG noise floor.  The border_ring_loss handles
        # the transition zone separately using the original mask.
        mask_interior = -F.max_pool2d(-mask, kernel_size=7, stride=1, padding=3)

        # ── full-image L1 ─────────────────────────────────────────────────
        l1_full = F.l1_loss(pred, target)

        # ── masked L1 (interior pixels only) ──────────────────────────────
        masked_pred   = pred   * mask_interior
        masked_target = target * mask_interior
        n_masked = mask_interior.sum().clamp(min=1) * 3  # channels
        l1_masked = (masked_pred - masked_target).abs().sum() / n_masked

        # ── SSIM ──────────────────────────────────────────────────────────
        ssim_loss = self.ssim(pred, target)

        # ── perceptual ────────────────────────────────────────────────────
        perc_loss = (self.perceptual(pred, target)
                     if (self.perceptual is not None and use_perceptual)
                     else pred.new_zeros(1).squeeze())

        # ── colour moment (interior only) ──────────────────────────────────
        cm_loss = (color_moment_loss(pred, target, mask_interior)
                   if self.w_color_moment > 0
                   else pred.new_zeros(1).squeeze())

        # ── border ring (feathered edge transition zone) ──────────────────
        b_loss = (border_ring_loss(pred, target, mask)
                  if self.w_border > 0
                  else pred.new_zeros(1).squeeze())

        # ── hard edge ring (outermost pixels of binary mask) ──────────────
        er_loss = (edge_ring_loss(pred, target, mask, self.edge_ring_px)
                   if self.w_edge_ring > 0
                   else pred.new_zeros(1).squeeze())

        total = (self.w_l1_full      * l1_full
               + self.w_l1_masked    * l1_masked
               + self.w_ssim         * ssim_loss
               + self.w_perceptual   * perc_loss
               + self.w_color_moment * cm_loss
               + self.w_border       * b_loss
               + self.w_edge_ring    * er_loss)

        breakdown = {
            "l1_full":      l1_full.item(),
            "l1_masked":    l1_masked.item(),
            "ssim":         ssim_loss.item(),
            "perceptual":   perc_loss.item() if self.perceptual else 0.0,
            "color_moment": cm_loss.item(),
            "border":       b_loss.item(),
            "edge_ring":    er_loss.item(),
            "total":        total.item(),
        }
        return total, breakdown
