"""
Inference for anchored overlay marks: image + listing ID -> cleaned image.

Only the profile's window around the expected mark position is processed; the
rest of the image is returned untouched. The window is snapped to the JPEG grid,
the mark is located inside it, and the model gets the stamped pixels, the exact
inverse for the located mark and the mark itself.
"""
from __future__ import annotations

import numpy as np
import torch

from wm_shared.marks import MarkProfile
from wm_shared.marks.localize import localize

from .features import build_input


def prepare_window(rgb: np.ndarray, profile: MarkProfile, listing_id: str, hint_scale: float):
    """Returns (window origin, model input tensor 1x7xhxw, localisation (x, y, score, style))."""
    h, w = rgb.shape[:2]
    ww, wh = profile.window
    x0, y0 = profile.window_origin(w, h)
    win = rgb[y0:y0 + wh, x0:x0 + ww]
    if win.shape[0] != wh or win.shape[1] != ww:  # image smaller than the window: pad by edge replication
        win = np.pad(win, ((0, wh - win.shape[0]), (0, ww - win.shape[1]), (0, 0)), mode="edge")
    ex, ey = profile.expected_origin(w, h)
    lx, ly, score, style = localize(win, profile, listing_id, prior=(ex - x0, ey - y0))
    hint = profile.render(listing_id, lx, ly, size=(ww, wh), style=style)
    wm_u8 = torch.from_numpy(np.ascontiguousarray(win.transpose(2, 0, 1)))[None]
    x = build_input(wm_u8, torch.from_numpy(hint)[None, None], profile.overlay, hint_scale)
    return (x0, y0), x, (lx, ly, score, style)


@torch.no_grad()
def remove_mark(rgb: np.ndarray, listing_id: str, model: torch.nn.Module, profile: MarkProfile,
                device: torch.device | str = "cpu", hint_scale: float = 8.0) -> tuple[np.ndarray, dict]:
    """rgb: HxWx3 uint8. Returns the cleaned image (uint8) and diagnostics."""
    (x0, y0), x, (lx, ly, score, style) = prepare_window(rgb, profile, listing_id, hint_scale)
    x = x.to(device)
    delta = model(x).float()
    pred = (x[:, :3] - delta).clamp(-1, 1)
    out_win = ((pred[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    out = rgb.copy()
    h, w = rgb.shape[:2]
    ph, pw = min(out_win.shape[0], h - y0), min(out_win.shape[1], w - x0)
    out[y0:y0 + ph, x0:x0 + pw] = out_win[:ph, :pw]
    return out, {"window": (x0, y0), "mark_origin": (x0 + lx, y0 + ly), "score": score, "style": style}
