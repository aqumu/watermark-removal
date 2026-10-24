"""
Fixed-position layered marks (the small Avito logo): the same mark, stamped at the
same pixels relative to the bottom-right corner of every image, after resizing.

Per pixel and channel the stamp is an affine map of the background b:
    out = b * (1 - Alpha) + A
with Alpha / A built from layers (tools/avito_watermark/fit_model.py):
    logo    white(ish) colour c, alpha = o * blur_sigma(coverage)
    shadow  black, s = so * blur_ss(coverage shifted by (sdx, sdy)) * (1 - coverage)
    darken  background-independent d = dk * rgb_d * blur_ds(shifted coverage) * (1 - coverage)
    Alpha = 1 - (1 - s)(1 - alpha),  A = alpha * c - d * (1 - alpha)

Profile directory (assets/marks/<name>/, mark.json "type": "fixed_layered"):
    mark.json  frame / window size, codec, layer parameters, image sizes
    mark.npz   alpha, a        nominal maps (the hint), frame_h x frame_w x 3
               model_alpha, model_a   the parametric layers alone
               bank, bank_dx, bank_dy coverage of the artwork at sub-pixel offsets

The frame is the last frame_h rows x frame_w columns of the image, the inference
window the last window_h x window_w (it ends at the image edge, so it is not on the
JPEG grid: the grid phase within the window is W % 16, H % 16).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


@dataclass
class FixedLayeredMark:
    name: str
    alpha: np.ndarray            # frame_h x frame_w x 3, nominal Alpha (the hint)
    a: np.ndarray                # frame_h x frame_w x 3, nominal A
    model_alpha: np.ndarray
    model_a: np.ndarray
    bank: np.ndarray             # len(bank_dy) x len(bank_dx) x frame_h x frame_w coverage
    bank_dx: np.ndarray
    bank_dy: np.ndarray
    params: dict
    window: tuple[int, int]      # (w, h)
    grid: int
    codec: dict
    sizes: list = field(default_factory=list)     # [(W, H)]
    size_weights: list = field(default_factory=list)
    layered = True

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.alpha.shape[1], self.alpha.shape[0]

    # ── geometry ─────────────────────────────────────────────────────────────

    def window_origin(self, width: int, height: int) -> tuple[int, int]:
        ww, wh = self.window
        return max(0, width - ww), max(0, height - wh)

    def expected_origin(self, width: int, height: int) -> tuple[float, float]:
        """Top-left of the logo in image coordinates."""
        fw, fh = self.frame_size
        return width - fw + self.params["x"], height - fh + self.params["y"]

    def to_window(self, frame_map: np.ndarray) -> np.ndarray:
        """Frame-sized map (h x w [x c]) on a zero window canvas, aligned at the bottom-right."""
        ww, wh = self.window
        fw, fh = self.frame_size
        out = np.zeros((wh, ww) + frame_map.shape[2:], np.float32)
        out[wh - fh:, ww - fw:] = frame_map
        return out

    def window_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Nominal (Alpha, A) on the window, wh x ww x 3."""
        return self.to_window(self.alpha), self.to_window(self.a)

    # ── rendering ────────────────────────────────────────────────────────────

    def coverage(self, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """Artwork coverage with the logo moved by (dx, dy) px (bilinear between bank entries 1/8 px apart)."""
        def pos(grid, v):
            t = float(np.clip((v - grid[0]) / (grid[1] - grid[0]), 0, len(grid) - 1))
            i = min(int(t), len(grid) - 2)
            return i, t - i
        iy, fy = pos(self.bank_dy, dy)
        ix, fx = pos(self.bank_dx, dx)
        b = self.bank
        return ((1 - fy) * ((1 - fx) * b[iy, ix] + fx * b[iy, ix + 1])
                + fy * ((1 - fx) * b[iy + 1, ix] + fx * b[iy + 1, ix + 1])).astype(np.float32)

    def layers_affine(self, q: dict, dx: float = 0.0, dy: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(Alpha, A) of the parametric layers for parameters q (keys as in mark.json "params")."""
        cover = self.coverage(dx, dy)
        logo = cv2.GaussianBlur(cover, (0, 0), max(q["sigma"], 1e-3))
        alpha = logo[..., None] * np.array([q["o"], q["o"], q["o"] * q["oB"]], np.float32)
        base = self.coverage(dx + q["sdx"], dy + q["sdy"])
        s = q["so"] * cv2.GaussianBlur(base, (0, 0), q["ss"]) * (1 - cover)
        d = (q["dk"] * cv2.GaussianBlur(base, (0, 0), q["ds"]) * (1 - cover))[..., None] \
            * np.array([1.0, q["dG"], q["dB"]], np.float32)
        c = np.array([q["cR"], q["cG"], q["cB"]], np.float32)
        Alpha = 1 - (1 - s[..., None]) * (1 - alpha)
        A = alpha * c - d * (1 - alpha)
        return Alpha.astype(np.float32), A.astype(np.float32)

    @staticmethod
    def composite(bg: np.ndarray, Alpha: np.ndarray, A: np.ndarray) -> np.ndarray:
        return bg * (1 - Alpha) + A

    @staticmethod
    def invert(img: np.ndarray, Alpha: np.ndarray, A: np.ndarray) -> np.ndarray:
        return (img - A) / np.maximum(1 - Alpha, 1e-3)


def load_fixed(name: str, root: Path, meta: dict) -> FixedLayeredMark:
    z = np.load(root / "mark.npz")
    sizes = [(int(s[0]), int(s[1])) for s in meta.get("sizes", [])]
    weights = [float(s[2]) if len(s) > 2 else 1.0 for s in meta.get("sizes", [])]
    return FixedLayeredMark(
        name=name, alpha=z["alpha"], a=z["a"], model_alpha=z["model_alpha"], model_a=z["model_a"],
        bank=z["bank"], bank_dx=z["bank_dx"], bank_dy=z["bank_dy"], params=meta["params"],
        window=tuple(meta["window"]), grid=int(meta.get("grid", 16)), codec=meta["codec"],
        sizes=sizes, size_weights=weights)
