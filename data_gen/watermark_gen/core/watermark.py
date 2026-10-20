import random
import cv2
import numpy as np

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

def _sample_shadow_param(cfg, key, default, is_int=False):
    val = cfg.get(key + "_range", cfg.get(key, None))
    if isinstance(val, list):
        return random.randint(*val) if is_int else random.uniform(*val)
    return val if val is not None else default

def apply_shadow(clean, alpha, config):
    shadow_cfg = config.get("shadow", {})
    enabled = shadow_cfg.get("enabled", False)
    if not enabled:
        return clean

    offset = _sample_shadow_param(shadow_cfg, "offset", 2, is_int=True)
    opacity = _sample_shadow_param(shadow_cfg, "opacity", 0.04)
    blur = _sample_shadow_param(shadow_cfg, "blur", float(offset))
    color = np.array(shadow_cfg.get("color", [0, 0, 0]), dtype=np.float64)

    k = 2 * offset + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(alpha, kernel)
    if blur > 0:
        dilated = cv2.GaussianBlur(dilated, (0, 0), blur)
    shadow_alpha = dilated * (1 - alpha) * opacity

    result = (1 - shadow_alpha[..., None]) * clean + shadow_alpha[..., None] * color
    return np.clip(result, 0, 255).astype(np.uint8)

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

    result = (1 - alpha[..., None]) * clean + alpha[..., None] * 255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, mask, "srgb"