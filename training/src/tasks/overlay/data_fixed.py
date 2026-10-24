"""
On-the-fly training data for fixed-position layered marks (the small Avito logo).

The mark sits at the same pixels relative to the bottom-right corner of every
image, so there is nothing to localise: the hint is the nominal mark on the last
window_h x window_w pixels, the same for every photo. What changes between photos
is the background, the JPEG quality and the JPEG grid phase (W % 16, H % 16: the
window ends at the image edge, the grid starts at the top-left).

Every sample imitates what the service sees:
  1. a virtual image size (the real size distribution, plus random grid phases);
  2. a clean background region covering the image's bottom-right corner from one
     grid block left of / above the window to the image edge, so JPEG pads the
     last partial blocks exactly like the encoder does on the real image;
  3. the mark re-rendered from its layers with a small random deviation from the
     measured parameters, plus a weighted part of the free-form correction (the
     measurement's uncertainty, ~2-3 levels; the stamp itself does not vary);
  4. occasionally a second mark: the same stamp twice (a re-uploaded photo), or
     an older, displaced / rescaled one under it (re-uploaded after resizing);
  5. JPEG at the profile's quality, identically for the stamped and the clean
     region; the target is the clean region after the same JPEG.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from wm_shared.marks.fixed import LUMA, FixedLayeredMark

from .data import OverlaySynthDataset, jpeg_roundtrip


class FixedLayeredDataset(OverlaySynthDataset):
    profile: FixedLayeredMark

    def _size(self, rng: random.Random) -> tuple[int, int]:
        p = self.profile
        W, H = rng.choices(p.sizes, p.size_weights)[0]
        if rng.random() < self.cfg.get("random_phase_prob", 0.0):
            W = max(p.window[0] + 32, W // 16 * 16 + rng.randint(0, 15))
            H = max(p.window[1] + 32, H // 16 * 16 + rng.randint(0, 15))
        return W, H

    def _true_mark(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        """(Alpha, A) frame maps of the mark as stamped: the layers with a small deviation."""
        p = self.profile
        dev = self.cfg["mark_deviation"]
        u = lambda key, default=(1.0, 1.0): rng.uniform(*dev.get(key, default))   # noqa: E731
        q = dict(p.params)
        q["o"] *= u("opacity_scale")
        shift = u("colour_shift", (0.0, 0.0))
        for k in ("cR", "cG", "cB"):
            q[k] += shift + u("colour_channel_shift", (0.0, 0.0))
        q["sigma"] = u("logo_sigma", (q["sigma"], q["sigma"]))
        q["so"] *= u("shadow_gain")
        q["ss"] *= u("shadow_blur_scale")
        q["sdx"] += u("shadow_dx", (0.0, 0.0))
        q["sdy"] += u("shadow_dy", (0.0, 0.0))
        q["dk"] *= u("darken_gain")
        q["ds"] *= u("darken_blur_scale")
        jx, jy = u("jitter", (0.0, 0.0)), u("jitter", (0.0, 0.0))
        Alpha, A = p.layers_affine(q, jx, jy)
        w = u("freeform_weight")
        return Alpha + w * (p.alpha - p.model_alpha), A + w * (p.a - p.model_a)

    def _older_mark(self, rng: random.Random, Alpha: np.ndarray, A: np.ndarray, size: tuple[int, int]):
        """A displaced / rescaled copy of the mark on a region canvas (bottom-right aligned)."""
        rw, rh = size
        s = rng.uniform(*self.cfg.get("older_mark_scale", [0.8, 1.0]))
        dx, dy = (rng.uniform(*self.cfg.get("older_mark_shift", [-12.0, 12.0])) for _ in range(2))
        # scale about the image's bottom-right corner (a photo resized after the first stamp), then shift
        mat = np.array([[s, 0, (1 - s) * rw + dx], [0, s, (1 - s) * rh + dy]], np.float32)
        warp = lambda m: cv2.warpAffine(m, mat, (rw, rh), flags=cv2.INTER_LINEAR,   # noqa: E731
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warp(Alpha), warp(A)

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        p = self.profile
        ww, wh = p.window
        fw, fh = p.frame_size
        g = p.grid

        W, H = self._size(rng)
        wx0, wy0 = p.window_origin(W, H)
        X0, Y0 = max(0, wx0 // g * g - g), max(0, wy0 // g * g - g)    # region origin, on the JPEG grid
        rw, rh = W - X0, H - Y0
        ox, oy = wx0 - X0, wy0 - Y0

        bg = self._background_region(rng, rw, rh).astype(np.float32)
        Alpha_f, A_f = self._true_mark(rng)
        Alpha = np.zeros((rh, rw, 3), np.float32)
        A = np.zeros((rh, rw, 3), np.float32)
        Alpha[rh - fh:, rw - fw:] = Alpha_f
        A[rh - fh:, rw - fw:] = A_f

        base = bg
        r = rng.random()
        p_same, p_older = self.cfg.get("double_same_prob", 0.0), self.cfg.get("double_older_prob", 0.0)
        if r < p_same:
            # the photo already carried the same mark at the same place (stamped twice, rounded in between)
            base = np.clip(np.rint(p.composite(bg, Alpha, A)), 0, 255)
        elif r < p_same + p_older:
            Al2, A2 = self._older_mark(rng, Alpha, A, (rw, rh))
            base = np.clip(np.rint(p.composite(bg, Al2, A2)), 0, 255)
        stamped = np.clip(np.rint(p.composite(base, Alpha, A)), 0, 255)
        effect = stamped - bg                       # everything the marks did, before JPEG

        q = self._quality(rng)
        wm = jpeg_roundtrip(stamped.astype(np.uint8), q)[oy:oy + wh, ox:ox + ww]
        clean = jpeg_roundtrip(bg.astype(np.uint8), q)[oy:oy + wh, ox:ox + ww]

        # loss region: every 16x16 macroblock (on the image's grid) the marks touch, one block of margin
        touched = (np.abs(effect).max(-1) > 0).astype(np.uint8)
        bh, bw = -(-rh // g), -(-rw // g)
        padded = np.zeros((bh * g, bw * g), np.uint8)
        padded[:rh, :rw] = touched
        blocks = padded.reshape(bh, g, bw, g).max(axis=(1, 3))
        blocks = cv2.dilate(blocks, np.ones((3, 3), np.uint8))
        region = np.kron(blocks, np.ones((g, g), np.uint8))[:rh, :rw][oy:oy + wh, ox:ox + ww]
        sig = (effect[oy:oy + wh, ox:ox + ww] @ LUMA) / 127.5
        m_true = (Alpha[oy:oy + wh, ox:ox + ww] @ LUMA)

        def chw(img):  # HxWx3 uint8 -> 3xHxW uint8
            return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        return {
            "wm": chw(wm),
            "target": chw(clean),
            "region": torch.from_numpy(np.ascontiguousarray(region)[None]),
            "m_true": torch.from_numpy(m_true[None].astype(np.float16)),
            "sig": torch.from_numpy(sig[None].astype(np.float16)),
            "loc_err": torch.zeros(2, dtype=torch.float32),
            "loc_score": torch.tensor(1.0, dtype=torch.float32),
            "style_ok": torch.tensor(True),
        }
