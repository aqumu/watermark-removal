"""Shared helpers for reverse-engineering the Cian watermark from served photos."""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SAMPLES_DIR = Path(os.environ.get(
    "CIAN_SAMPLES_DIR", r"C:\Users\aqumu\PycharmProjects\deduper\dumps\cian_samples"))
OUT_DIR = Path(__file__).resolve().parent / "out"

# Search window: the mark sits in the bottom-right corner of every served size.
CORNER_W, CORNER_H = 480, 260


def list_samples(full_only: bool = True) -> list[Path]:
    files = sorted(SAMPLES_DIR.glob("*/*.jpg"))
    if full_only:
        files = [f for f in files if f.name.endswith("-1.jpg")]
    return files


def listing_id(path: Path) -> str:
    return path.parent.name


def load_rgb(path: Path) -> np.ndarray:
    """Decode a served photo (they are WebP regardless of the .jpg name) as float32 RGB."""
    return np.asarray(Image.open(path).convert("RGB"), np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def highpass(gray: np.ndarray, k: int = 15) -> np.ndarray:
    """Detail layer: image minus a median background, so the faint mark stands out."""
    bg = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), k).astype(np.float32)
    return gray - bg


def corner_origin(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape[:2]
    return max(0, w - CORNER_W), max(0, h - CORNER_H)


def ncc_map(image: np.ndarray, template: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    if mask is not None:
        return cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask.astype(np.float32))
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def subpixel_peak(score: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Parabolic refinement of an integer peak."""
    def refine(m1, c, p1):
        den = m1 - 2 * c + p1
        return 0.0 if abs(den) < 1e-9 else 0.5 * (m1 - p1) / den
    dy = refine(score[y - 1, x], score[y, x], score[y + 1, x]) if 0 < y < score.shape[0] - 1 else 0.0
    dx = refine(score[y, x - 1], score[y, x], score[y, x + 1]) if 0 < x < score.shape[1] - 1 else 0.0
    return y + dy, x + dx


def id_from_name(name: str) -> str | None:
    m = re.search(r"(\d{9})", name)
    return m.group(1) if m else None
