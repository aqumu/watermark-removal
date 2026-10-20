import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .watermark import blend, apply_edge_effects
from .degrade import degrade
from .io import save_sample

def process_one(args):
    idx, clean_path, wm_rgba, config, output_dir = args

    clean = cv2.imread(str(clean_path))
    if clean is None:
        return None
    h, w = clean.shape[:2]

    # Scale watermark so its width is ~78-80% of the image width
    frac = random.uniform(*config["placement"]["width_fraction"])
    target_w = int(w * frac)
    wm_orig_h, wm_orig_w = wm_rgba.shape[:2]
    target_h = int(wm_orig_h * target_w / wm_orig_w)
    interp = cv2.INTER_AREA if target_w < wm_orig_w else cv2.INTER_CUBIC
    wm_scaled = cv2.resize(wm_rgba, (target_w, target_h), interpolation=interp)
    wm_h, wm_w = wm_scaled.shape[:2]

    # Center with a small random jitter (up to 3% of image dimensions)
    jitter_x = int(w * random.uniform(-0.03, 0.03))
    jitter_y = int(h * random.uniform(-0.03, 0.03))
    x = max(0, min((w - wm_w) // 2 + jitter_x, w - wm_w))
    y = max(0, min((h - wm_h) // 2 + jitter_y, h - wm_h))

    canvas = clean.copy()

    # Full-image shadow composited before watermark blending so the
    # shadow doesn't clip at the watermark-region boundary
    shadow_cfg = config.get("shadow", {})
    if shadow_cfg.get("enabled", False):
        offset_range = shadow_cfg.get("offset_range", shadow_cfg.get("offset", None))
        if isinstance(offset_range, list):
            offset = random.randint(*offset_range)
        else:
            offset = offset_range if offset_range is not None else 2

        opacity_range = shadow_cfg.get("opacity_range", shadow_cfg.get("opacity", None))
        if isinstance(opacity_range, list):
            opacity = random.uniform(*opacity_range)
        else:
            opacity = opacity_range if opacity_range is not None else 0.04

        blur_range = shadow_cfg.get("blur_range", shadow_cfg.get("blur", None))
        if isinstance(blur_range, list):
            blur = random.uniform(*blur_range)
        else:
            blur = blur_range if blur_range is not None else float(offset)

        color = np.array(shadow_cfg.get("color", [0, 0, 0]), dtype=np.float64)

        wm_alpha = wm_scaled[..., 3] / 255.0
        feathered = apply_edge_effects(wm_alpha, config)

        full_alpha = np.zeros((h, w), dtype=np.float64)
        full_alpha[y:y+wm_h, x:x+wm_w] = feathered

        k = 2 * offset + 1
        kernel = np.ones((k, k), np.uint8)
        dilated = cv2.dilate(full_alpha, kernel)
        if blur > 0:
            dilated = cv2.GaussianBlur(dilated, (0, 0), blur)
        shadow_alpha = dilated * (1 - full_alpha) * opacity

        canvas = (1 - shadow_alpha[..., None]) * canvas.astype(np.float64) + shadow_alpha[..., None] * color
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    region = canvas[y:y+wm_h, x:x+wm_w]

    blended, mask, mode = blend(region, wm_scaled, config)
    canvas[y:y+wm_h, x:x+wm_w] = blended

    # Full-resolution mask so it aligns with clean/watermarked images
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y:y+wm_h, x:x+wm_w] = mask

    degraded = degrade(canvas, config)

    meta = {
        "blend_mode": mode,
        "position": [x, y],
    }

    save_sample(output_dir, idx, clean, degraded, full_mask, meta)

def generate_dataset(config):
    clean_dir = Path(config["paths"]["clean_images_dir"]).resolve()
    wm = cv2.imread(str(Path(config["paths"]["watermark_path"]).resolve()), cv2.IMREAD_UNCHANGED)
    if wm is None:
        raise FileNotFoundError(f"Watermark not found: {config['paths']['watermark_path']}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p.resolve() for p in clean_dir.iterdir() if p.suffix.lower() in image_exts]
    random.shuffle(images)

    num_samples = config["dataset"]["num_samples"]
    if num_samples > len(images):
        print(f"[dataset] Warning: num_samples ({num_samples}) > available images ({len(images)}). Capping at {len(images)}.")
        num_samples = len(images)

    args = [
        (i, images[i], wm, config, config["dataset"]["output_dir"])
        for i in range(num_samples)
    ]

    with Pool() as p:
        results = list(tqdm(p.imap(process_one, args), total=num_samples))
    skipped = results.count(None)
    if skipped:
        print(f"[dataset] Skipped {skipped} unreadable image(s).")