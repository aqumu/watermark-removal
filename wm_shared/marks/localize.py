"""
Locate a known mark inside an inference window.

Correlates the window's detail layer with the detail layer of the mark rendered
for the listing (normalised cross-correlation, either sign: the mark brightens
dark backgrounds and darkens near-white ones), then refines the peak to sub-pixel
precision with a parabola. The text style is not known in advance, so every
style is tried and the best match wins. Training runs this same function on
synthetic windows, so the model learns this localiser's actual error distribution.
"""
from __future__ import annotations

import cv2
import numpy as np

from .profile import MarkProfile

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
HP_SIGMA = 6.0


def detail(gray: np.ndarray, sigma: float = HP_SIGMA) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def _parabola(m1: float, c: float, p1: float) -> float:
    den = m1 - 2 * c + p1
    return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))


def _match(window_detail: np.ndarray, mark: np.ndarray,
           box: tuple[int, int, int, int] | None) -> tuple[float, float, float]:
    """NCC peak of the mark's detail layer, searched only inside box = (x0, y0, x1, y1) of origins."""
    if box is not None:
        x0, y0, x1, y1 = box
        mh, mw = mark.shape
        window_detail = window_detail[y0:y1 + mh, x0:x1 + mw]
    else:
        x0 = y0 = 0
    score = cv2.matchTemplate(window_detail, detail(mark), cv2.TM_CCOEFF_NORMED)
    a = np.abs(score)
    y, x = np.unravel_index(int(a.argmax()), a.shape)
    fx = _parabola(a[y, x - 1], a[y, x], a[y, x + 1]) if 0 < x < a.shape[1] - 1 else 0.0
    fy = _parabola(a[y - 1, x], a[y, x], a[y + 1, x]) if 0 < y < a.shape[0] - 1 else 0.0
    return x0 + x + fx, y0 + y + fy, float(score[y, x])


def search_box(profile: MarkProfile, prior: tuple[float, float], window_shape: tuple[int, int],
               margin: float = 3.0) -> tuple[int, int, int, int]:
    """Integer origin range around the expected origin covering the profile's jitter."""
    h, w = window_shape
    fw, fh = profile.frame_size
    jx, jy = profile.jitter["x"], profile.jitter["y"]
    x0 = max(0, int(np.floor(prior[0] + jx[0] - margin)))
    y0 = max(0, int(np.floor(prior[1] + jy[0] - margin)))
    x1 = min(w - fw, int(np.ceil(prior[0] + jx[1] + margin)))
    y1 = min(h - fh, int(np.ceil(prior[1] + jy[1] + margin)))
    return x0, y0, max(x0, x1), max(y0, y1)


def localize(window_rgb: np.ndarray, profile: MarkProfile, listing_id: str,
             prior: tuple[float, float] | None = None,
             styles: list[str] | None = None) -> tuple[float, float, float, str]:
    """Return (x, y, score, style): the mark frame's top-left inside the window, the
    signed NCC peak and the best-matching text style.

    prior: the expected frame origin inside the window; the search is limited to
    the profile's jitter range around it (plus a small margin).
    """
    wd = detail(window_rgb.astype(np.float32) @ _LUMA)
    box = search_box(profile, prior, wd.shape) if prior is not None else None
    best = None
    for style in styles or list(profile.styles):
        x, y, s = _match(wd, profile.render(listing_id, style=style), box)
        if best is None or abs(s) > abs(best[2]):
            best = (x, y, s, style)
    return best
