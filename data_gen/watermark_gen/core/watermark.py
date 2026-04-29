import cv2
import numpy as np
from .colorspace import srgb_to_linear, linear_to_srgb

def random_scale(wm, scale_range):
    scale = np.random.uniform(*scale_range)
    h, w = wm.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(wm, new_size, interpolation=cv2.INTER_CUBIC)

def apply_edge_effects(alpha, config):
    sigma = np.random.uniform(*config["edge"]["gaussian_sigma_range"])
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0,0), sigma)

    k = np.random.randint(*config["edge"]["morph_kernel"])
    if k > 0:
        kernel = np.ones((k,k), np.uint8)
        alpha = cv2.dilate(alpha, kernel)

    return alpha

def blend(clean, wm_rgba, config):
    h, w = clean.shape[:2]
    wm_rgb = wm_rgba[..., :3]
    alpha = wm_rgba[..., 3] / 255.0

    alpha = apply_edge_effects(alpha, config)

    # Soft mask: save the normalised feathered alpha BEFORE scaling by
    # base_alpha so the mask spans the full [0, 255] range.  This preserves
    # the 0→1 gradient at the watermark boundary so the model can explicitly
    # see the transition zone instead of a hard binary edge.
    mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    base_alpha = np.random.uniform(*config["alpha"]["base_range"])
    delta = np.random.uniform(-config["alpha"]["perturbation"],
                               config["alpha"]["perturbation"])
    alpha = alpha * (base_alpha + delta)

    if np.random.rand() < config["blending"]["linear_prob"]:
        clean_lin = srgb_to_linear(clean.astype(np.float32))
        wm_lin = srgb_to_linear(wm_rgb.astype(np.float32))
        blended = (1 - alpha[..., None]) * clean_lin + alpha[..., None] * 1.0
        result = linear_to_srgb(blended)
        mode = "linear"
    else:
        result = (1 - alpha[..., None]) * clean + alpha[..., None] * 255
        result = np.clip(result, 0, 255).astype(np.uint8)
        mode = "srgb"

    return result, mask, mode