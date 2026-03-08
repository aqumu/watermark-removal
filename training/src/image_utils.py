import cv2
import numpy as np
import torch
import random

def compute_gradient(bgr: np.ndarray) -> torch.Tensor:
    """
    Compute normalised grayscale Sobel gradient magnitude.
    uint8 BGR HxWx3 → float32 1xHxW in [0, 1]
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val
    return torch.from_numpy(mag).unsqueeze(0)  # 1xHxW


def dilate_mask_input(mask: np.ndarray) -> np.ndarray:
    """
    Create a randomly dilated binary mask for the model's 4th input channel.

    1. Threshold the soft mask at 0.5 → binary {0, 1}.
    2. Dilate by a random kernel (3–7 px) so the true watermark edge
       is always inside the mask boundary.

    mask : HxW float32 in [0, 1]
    returns HxW float32 in [0, 1]
    """
    binary = (mask >= 0.5).astype(np.uint8)  # uint8 needed for cv2.dilate
    ksize = random.choice([3, 5, 7])  # randomise dilation amount
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated.astype(np.float32)


def dilate_mask_for_loss(mask: np.ndarray, dilate_pct: float,
                         image_size: int) -> np.ndarray:
    """
    Dilate the mathematical mask slightly before computing the loss target.
    This compensates for JPEG compression and downscaling smearing the bright 
    watermark pixels past the strict mathematical boundary of the clean mask.
    """
    if dilate_pct <= 0:
        return mask

    # Convert percentage to pixel radius (minimum 1px if pct > 0)
    radius = max(1, int(image_size * (dilate_pct / 100.0)))
    ksize = radius * 2 + 1

    binary = (mask >= 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated.astype(np.float32)


def blur_mask_for_loss(mask: np.ndarray, blur_pct: float,
                       image_size: int) -> np.ndarray:
    """
    Apply Gaussian blur to the ideal mask before it is used for loss weighting.

    The blur softens the hard edge into a smooth gradient, which lets
    edge-focused loss terms (border ring, masked L1) compute meaningful
    gradients across the transition zone instead of seeing a step function.
    """
    if blur_pct <= 0:
        return mask

    sigma = image_size * (blur_pct / 100.0)
    
    # Kernel size must be odd and comfortably cover exactly the ~3 sigma spread
    ksize = int(round(sigma * 3))
    if ksize % 2 == 0:
        ksize += 1
    
    # Needs float32 representation for cv2.GaussianBlur
    mask_blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma)
    return np.clip(mask_blurred, 0, 1)
