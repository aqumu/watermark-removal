"""
On-the-fly training data for anchored overlay marks.

Every sample imitates what the service will see, end to end:
  1. a clean background region (in-domain photos away from their own mark, or
     clean photos), randomly downscaled like a resized original;
  2. a virtual image size drawn from the real size distribution, which fixes the
     inference window and the expected mark origin; the true origin adds jitter;
  3. the mark for a random ID, rendered at that fractional position with a small
     random deviation from the nominal template (opacity, overlay level, blur), so
     the model does not depend on the template being exact;
  4. JPEG at the profile's quality, applied identically to the stamped and the
     clean region, with the window on the encoder's 16-px grid as at inference;
  5. the inference localiser on the stamped window, and the nominal mark rendered
     where it says, plus the exact inverse for that hint.

The target is the clean region after the same JPEG, i.e. what would have been
served without the mark. Outside the macroblocks the mark touches, input and
target are bit-identical, so the model learns to leave them alone.
"""
from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from wm_shared.marks import MarkProfile
from wm_shared.marks.localize import localize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Background:
    path: str
    kind: str                       # "mark" (has its own mark in the corner) or "clean"
    listing_id: str | None = None


def is_holdout(key: str, fraction: float) -> bool:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < fraction * 10_000


def load_backgrounds(sources: list[dict], split: str, holdout: float) -> list[Background]:
    """sources: [{"csv": path, "kind": "mark"} | {"dir": path, "kind": "clean"}, ...]"""
    out: list[Background] = []
    for src in sources:
        kind = src.get("kind", "clean")
        if "csv" in src:
            with open(src["csv"], encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = row.get("listing_id") or row["path"]
                    if is_holdout(key, holdout) == (split == "val"):
                        out.append(Background(row["path"], kind, row.get("listing_id")))
        else:
            for p in sorted(Path(src["dir"]).iterdir()):
                if p.suffix.lower() in IMAGE_EXTS and is_holdout(p.name, holdout) == (split == "val"):
                    out.append(Background(str(p), kind))
    return out


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]


def random_listing_id(rng: random.Random, length: int) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(length - 1))


class OverlaySynthDataset(Dataset):
    def __init__(self, profile: MarkProfile, backgrounds: list[Background], length: int,
                 cfg: dict, seed: int | None = None):
        """seed=None: fresh randomness every epoch (training); an int: a fixed set (validation)."""
        self.profile = profile
        self.backgrounds = backgrounds
        self.length = length
        self.cfg = cfg
        self.seed = seed
        if not backgrounds:
            raise RuntimeError("no background images")

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> random.Random:
        if self.seed is not None:
            return random.Random(self.seed * 1_000_003 + index)
        return random.Random()

    # ── pieces ───────────────────────────────────────────────────────────────

    def _quality(self, rng: random.Random) -> int:
        dist = self.profile.codec["jpeg_quality"]
        if rng.random() < self.cfg.get("quality_jitter_prob", 0.0):
            lo, hi = self.cfg.get("quality_jitter_range", [70, 95])
            return rng.randint(lo, hi)
        qs, ps = zip(*[(int(q), float(p)) for q, p in dist.items()])
        return rng.choices(qs, ps)[0]

    def _background_region(self, rng: random.Random, rw: int, rh: int) -> np.ndarray:
        """A clean rw x rh RGB region (uint8)."""
        for _ in range(50):
            bg = rng.choice(self.backgrounds)
            img = cv2.imread(bg.path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            lo, hi = self.cfg["bg_scale"].get(bg.kind, [0.75, 1.0])
            s = rng.uniform(lo, hi)
            sw, sh = int(round(w * s)), int(round(h * s))
            if sw < rw or sh < rh:
                continue
            # the background's own mark sits in its bottom-right corner: stay clear of it
            ex, ey = (np.array(self.profile.expected_origin(w, h)) - 60) * s if bg.kind == "mark" else (sw, sh)
            for _ in range(20):
                x = rng.randint(0, sw - rw)
                y = rng.randint(0, sh - rh)
                if x + rw <= ex or y + rh <= ey:
                    break
            else:
                continue
            # crop generously in source coordinates first, then resize only that part
            x0, y0 = int(x / s), int(y / s)
            x1, y1 = min(w, int(np.ceil((x + rw) / s)) + 2), min(h, int(np.ceil((y + rh) / s)) + 2)
            crop = img[y0:y1, x0:x1]
            region = cv2.resize(crop, (int(round((x1 - x0) * s)), int(round((y1 - y0) * s))),
                                interpolation=cv2.INTER_AREA)[:rh, :rw]
            if region.shape[0] == rh and region.shape[1] == rw:
                if rng.random() < 0.5:
                    region = region[:, ::-1]
                return np.ascontiguousarray(region[..., ::-1])
        raise RuntimeError("could not sample a background region")

    def _true_mark(self, rng: random.Random, listing_id: str, style: str, x: float, y: float,
                   size: tuple[int, int]) -> tuple[np.ndarray, float]:
        """The mark as stamped: nominal template with a small random deviation."""
        p = self.profile
        dev = self.cfg["mark_deviation"]
        m = p.render(listing_id, x, y, size=size, style=style)
        if "sharpness" in dev:
            # real marks are on average a bit softer than the template and vary per photo
            # (each photo resized from a different original): m + k * highpass(m),
            # k < 0 softens, k > 0 sharpens
            k = rng.uniform(*dev["sharpness"])
            m = m + k * (m - cv2.GaussianBlur(m, (0, 0), 1.0))
        elif rng.random() < dev.get("blur_prob", 0.0):
            m = cv2.GaussianBlur(m, (0, 0), rng.uniform(*dev["blur_sigma"]))
        m = m * rng.uniform(*dev["opacity_scale"])
        overlay = p.overlay + rng.uniform(*dev["overlay_shift"])
        return np.clip(m, 0, 1), overlay

    # ── sample ───────────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        p = self.profile
        ww, wh = p.window
        g = p.grid
        rw, rh = ww + 2 * g, wh + 2 * g

        # geometry of a virtual served image
        W, H = rng.choice(p.sizes)
        wx0, wy0 = p.window_origin(W, H)
        ex, ey = p.expected_origin(W, H)
        tx = ex + rng.uniform(*p.jitter["x"]) - wx0
        ty = ey + rng.uniform(*p.jitter["y"]) - wy0
        fw, fh = p.frame_size
        tx, ty = float(np.clip(tx, 0, ww - fw)), float(np.clip(ty, 0, wh - fh))

        lengths, lprobs = zip(*p.id_lengths.items())
        listing_id = random_listing_id(rng, rng.choices(lengths, lprobs)[0])
        names = list(p.styles)
        style = rng.choices(names, [p.styles[n].prob for n in names])[0]
        bg = self._background_region(rng, rw, rh).astype(np.float32)
        m_true, overlay = self._true_mark(rng, listing_id, style, tx + g, ty + g, (rw, rh))
        base = bg
        if rng.random() < self.cfg.get("double_mark_prob", 0.0):
            # a photo re-used from another listing keeps that listing's mark and gets a second
            # one; only the listing's own mark is hinted, but both must go
            jx, jy = p.jitter["x"], p.jitter["y"]
            while True:
                dx = rng.uniform(*jx) - rng.uniform(*jx)
                dy = rng.uniform(*jy) - rng.uniform(*jy)
                if abs(dx) > 3 or abs(dy) > 3:
                    break
            other_style = rng.choices(names, [p.styles[n].prob for n in names])[0]
            m_other, overlay_other = self._true_mark(rng, random_listing_id(rng, len(listing_id)), other_style,
                                                     tx + g + dx, ty + g + dy, (rw, rh))
            base = p.composite(bg, m_other, overlay_other)
        stamped = np.clip(np.rint(p.composite(base, m_true, overlay)), 0, 255).astype(np.uint8)
        if base is not bg:
            m_true = np.clip(m_true + m_other, 0, 1)   # loss region and ghost term cover both marks

        q = self._quality(rng)
        wm = jpeg_roundtrip(stamped, q)[g:g + wh, g:g + ww]
        clean = jpeg_roundtrip(bg.astype(np.uint8), q)[g:g + wh, g:g + ww]
        m_true = m_true[g:g + wh, g:g + ww]

        # the hint, exactly as inference builds it
        # a few photos are re-used from another listing and carry its ID, so the
        # ID the service gets is occasionally wrong: the model must not trust the
        # hint's text blindly
        hint_id = listing_id
        if rng.random() < self.cfg.get("wrong_id_prob", 0.0):
            hint_id = random_listing_id(rng, len(listing_id))
        lx, ly, score, found_style = localize(wm, p, hint_id, prior=(ex - wx0, ey - wy0))
        hint = p.render(hint_id, lx, ly, size=(ww, wh), style=found_style)

        # loss region: every 16x16 macroblock the mark touches, one block of margin
        touched = (m_true > 1e-3).astype(np.uint8)
        blocks = cv2.resize(touched, (ww // g, wh // g), interpolation=cv2.INTER_AREA) > 0
        blocks = cv2.dilate(blocks.astype(np.uint8), np.ones((3, 3), np.uint8))
        region = cv2.resize(blocks, (ww, wh), interpolation=cv2.INTER_NEAREST)

        def chw(img):  # HxWx3 uint8 -> 3xHxW uint8
            return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # uint8 / fp16 keeps worker -> trainer traffic small; the model input
        # (including the inverse) is built on the GPU by features.build_input
        return {
            "wm": chw(wm),
            "hint": torch.from_numpy(hint[None].astype(np.float16)),
            "target": chw(clean),
            "region": torch.from_numpy(region[None]),
            "m_true": torch.from_numpy(m_true[None].astype(np.float16)),
            "loc_err": torch.tensor([lx - tx, ly - ty], dtype=torch.float32),
            "loc_score": torch.tensor(score, dtype=torch.float32),
            "style_ok": torch.tensor(found_style == style),
        }
