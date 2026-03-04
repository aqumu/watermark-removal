import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .watermark import blend
from .degrade import degrade
from .io import save_sample

def process_one(args):
    idx, clean_path, wm_rgba, config, output_dir = args

    clean = cv2.imread(str(clean_path))
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
    clean_dir = Path(config["paths"]["clean_images_dir"])
    wm = cv2.imread(config["paths"]["watermark_path"], cv2.IMREAD_UNCHANGED)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in clean_dir.iterdir() if p.suffix.lower() in image_exts]
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
        list(tqdm(p.imap(process_one, args), total=num_samples))