"""
Torch-free overlay removal for deployment: ONNX Runtime + numpy.

    remover = OverlayRemover("student.onnx", mark="cian", threads=6)
    cleaned = remover.remove(rgb, listing_id)                  # one image
    cleaned_list = remover.remove_batch([(rgb, id), ...])      # several, one model call

Preprocessing is the same as in training (features.build_input, localiser, hint
rendering), re-implemented in numpy so the service does not need torch.
"""
from __future__ import annotations

import numpy as np

from wm_shared.marks import MarkProfile, load_mark
from wm_shared.marks.localize import localize


def build_input_np(win_u8: np.ndarray, hint: np.ndarray, overlay: float, hint_scale: float) -> np.ndarray:
    """win_u8: HxWx3 uint8 RGB, hint: HxW opacity. Returns 7xHxW float32 (same as features.build_input)."""
    wm = win_u8.astype(np.float32).transpose(2, 0, 1)
    m = hint.astype(np.float32)[None]
    inv = np.clip((wm - m * overlay) / np.maximum(1.0 - m, 1e-3), -32.0, 287.0)
    return np.concatenate([wm / 127.5 - 1.0, inv / 127.5 - 1.0, m * hint_scale], axis=0)


class OverlayRemover:
    def __init__(self, onnx_path: str, mark: str | MarkProfile = "cian", threads: int = 0, hint_scale: float = 8.0):
        import onnxruntime as ort
        self.profile = load_mark(mark) if isinstance(mark, str) else mark
        self.hint_scale = hint_scale
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def prepare(self, rgb: np.ndarray, listing_id: str) -> tuple[tuple[int, int], np.ndarray, dict]:
        """Window origin, model input (7xhxw) and localisation diagnostics for one image."""
        p = self.profile
        h, w = rgb.shape[:2]
        ww, wh = p.window
        x0, y0 = p.window_origin(w, h)
        win = rgb[y0:y0 + wh, x0:x0 + ww]
        if win.shape[0] != wh or win.shape[1] != ww:
            win = np.pad(win, ((0, wh - win.shape[0]), (0, ww - win.shape[1]), (0, 0)), mode="edge")
        ex, ey = p.expected_origin(w, h)
        lx, ly, score, style, offset, scale = localize(win, p, str(listing_id), prior=(ex - x0, ey - y0))
        hint = p.render(str(listing_id), lx, ly, size=(ww, wh), style=style, text_offset=offset, scale=scale)
        info = {"window": (x0, y0), "mark_origin": (x0 + lx, y0 + ly), "score": score, "style": style,
                "text_offset": offset, "scale": scale}
        return (x0, y0), build_input_np(win, hint, p.overlay, self.hint_scale), info

    @staticmethod
    def _paste(rgb: np.ndarray, x: np.ndarray, delta: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        pred = np.clip(x[:3] - delta, -1.0, 1.0).transpose(1, 2, 0)
        win = np.rint((pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = rgb.copy()
        x0, y0 = origin
        h, w = rgb.shape[:2]
        out[y0:y0 + win.shape[0], x0:x0 + win.shape[1]] = win[:h - y0, :w - x0]
        return out

    def remove(self, rgb: np.ndarray, listing_id: str) -> tuple[np.ndarray, dict]:
        """rgb: HxWx3 uint8 RGB. Returns the cleaned image and diagnostics."""
        return self.remove_batch([(rgb, listing_id)])[0]

    def remove_batch(self, items: list[tuple[np.ndarray, str]]) -> list[tuple[np.ndarray, dict]]:
        prepared = [self.prepare(rgb, lid) for rgb, lid in items]
        xs = np.stack([x for _, x, _ in prepared]).astype(np.float32)
        deltas = self.session.run(None, {"x": xs})[0]
        return [(self._paste(rgb, x, d, origin), info)
                for (rgb, _), (origin, x, info), d in zip(items, prepared, deltas)]
