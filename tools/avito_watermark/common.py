"""Shared helpers for reverse-engineering the small Avito mark from served photos."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
CW, CH = 320, 160          # bottom-right corner crop studied everywhere
_ART = None


def artwork() -> np.ndarray:
    """Alpha of the Avito logo artwork (data_gen/watermark.png, 8001x2032), float32 0..1."""
    global _ART
    if _ART is None:
        _ART = np.asarray(Image.open(REPO / "data_gen" / "watermark.png"))[..., 3].astype(np.float32) / 255.0
    return _ART


def render_logo(width: float, x: float, y: float, frame=(CW, CH), blur: float = 0.0) -> np.ndarray:
    """Artwork scaled to `width` px with its top-left at sub-pixel (x, y) in a frame of size (w, h).

    Area-averaged from the high-resolution raster: a 1x supersampled warp of a
    16x pre-shrink, which keeps edges exact to ~1/16 px.
    """
    art = artwork()
    s = width / art.shape[1]
    ss = 16
    pre = cv2.resize(art, (max(1, int(round(art.shape[1] * s * ss))), max(1, int(round(art.shape[0] * s * ss)))),
                     interpolation=cv2.INTER_AREA)
    sx = art.shape[1] * s / pre.shape[1]; sy = art.shape[0] * s / pre.shape[0]
    big = np.zeros((frame[1] * ss, frame[0] * ss), np.float32)
    M = np.array([[sx * ss, 0, x * ss], [0, sy * ss, y * ss]], np.float32)
    big = cv2.warpAffine(pre, M, (frame[0] * ss, frame[1] * ss), flags=cv2.INTER_LINEAR, borderValue=0)
    out = cv2.resize(big, frame, interpolation=cv2.INTER_AREA)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out


def index(min_w: int = CW, min_h: int = CH) -> list[dict]:
    rows = []
    with open(OUT / "avito_index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["w"], r["h"] = int(r["w"]), int(r["h"])
            if r["w"] >= min_w and r["h"] >= min_h:
                rows.append(r)
    return rows


def split(sha: str) -> str:
    """Stable train/holdout split: ~10% of photos are held out for accuracy checks."""
    return "holdout" if int(sha[:2], 16) < 26 else "fit"


def load_rgb(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)


def corner(img: np.ndarray) -> np.ndarray:
    return img[-CH:, -CW:]


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def detail(g: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    return g - cv2.GaussianBlur(g, (0, 0), sigma)
