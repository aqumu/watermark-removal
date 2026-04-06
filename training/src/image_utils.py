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


def dilate_mask_input(mask: np.ndarray, augment: bool = False,
                      image_size: int = 256) -> np.ndarray:
    """
    Create a randomly perturbed binary mask for the model's 4th input channel.

    augment=False (inference): always dilate by exactly 5 px so the true watermark
        edge is inside the mask boundary. Fixed value for deterministic inference.

    augment=True (training): additionally applies positional jitter and random
        shape variation to teach the model to handle imperfect real-world masks:
          - Translation ±(4 × image_size/256) px in x and y (scales with resolution)
          - Morphological op sampled from: erode-3, erode-5, no-op,
            dilate-3, dilate-5, dilate-7  (dilate weighted 3×)

    mask       : HxW float32 in [0, 1]
    image_size : training resolution — used to scale the translation jitter so
                 ±4px at 256px maps to ±8px at 512px (same proportional coverage).
    returns HxW float32 in [0, 1]
    """
    binary = (mask >= 0.5).astype(np.uint8)

    if augment:
        # Scale translation with resolution so jitter covers the same
        # proportional area regardless of whether we train at 256 or 512.
        max_translate = max(1, round(4 * image_size / 256))
        tx = random.randint(-max_translate, max_translate)
        ty = random.randint(-max_translate, max_translate)
        if tx != 0 or ty != 0:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            binary = cv2.warpAffine(binary, M, (binary.shape[1], binary.shape[0]))

        # Random shape: erode (undersized), no-op, or dilate (oversized)
        op, ksize = random.choice([
            ('erode',  3), ('erode',  5),
            ('none',   0),
            ('dilate', 3), ('dilate', 5), ('dilate', 7),
            ('dilate', 3), ('dilate', 5), ('dilate', 7),  # dilate 3× more likely
        ])
        if op != 'none':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if op == 'dilate':
                binary = cv2.dilate(binary, kernel, iterations=1)
            else:
                binary = cv2.erode(binary, kernel, iterations=1)
    else:
        # Fixed kernel size for deterministic, reproducible inference results.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel, iterations=1)

    return binary.astype(np.float32)



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

    # Kernel must be odd and span ≥3σ on each side (total ≥6σ + 1 pixels).
    # Old formula (sigma*3) only covered 1.5σ per side — half the required width.
    ksize = max(1, int(round(sigma * 6)) | 1)   # bitwise OR 1 forces odd
    
    # Needs float32 representation for cv2.GaussianBlur
    mask_blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma)
    return np.clip(mask_blurred, 0, 1)
