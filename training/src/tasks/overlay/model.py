"""
Residual U-Net for overlay removal.

Input  : 7 channels — stamped window (RGB), exact inverse for the hint (RGB),
         hint opacity (1), all roughly in [-1, 1]
Output : 3 channels — residual; prediction = stamped - residual

No normalisation layers: the network stays a plain conv stack, which keeps CPU
inference simple and makes pruning / int8 quantisation straightforward. The
head starts at zero, so an untrained model is the identity.

unshuffle=k (CPU-oriented): a lossless pixel-unshuffle folds each k x k block
into channels, the U-Net runs at 1/k resolution and a pixel-shuffle restores the
full-resolution residual. Nothing is discarded, but the expensive full-resolution
convolutions go away (about k^2 fewer MACs at equal widths).

refine=c (with unshuffle > 1): a thin full-resolution head (three 3x3 convs, c
channels) sees the full-resolution input and the shuffled residual and adds a
correction. Half-resolution students otherwise leave faint ghosts of the 2-3 px
text strokes; the head is zero-initialised, so a trained student loads into it
unchanged (strict=False) and starts from its own output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(y + self.skip(x))


class OverlayUNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 3,
                 widths: tuple[int, ...] = (32, 64, 96, 128), unshuffle: int = 1, refine: int = 0):
        super().__init__()
        self.unshuffle = unshuffle
        self.refine = None
        if unshuffle > 1 and refine > 0:
            self.refine = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, refine, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(refine, refine, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(refine, out_channels, 3, padding=1))
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        in_channels *= unshuffle * unshuffle
        out_channels *= unshuffle * unshuffle
        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(ResBlock(widths[max(i - 1, 0)], w))
            if i < len(widths) - 1:
                self.down.append(nn.Conv2d(w, w, 2, stride=2))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(ResBlock(2 * widths[i - 1], widths[i - 1]))
        self.head = nn.Conv2d(widths[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.multiple = unshuffle * 2 ** (len(widths) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = x
        if self.unshuffle > 1:
            x = F.pixel_unshuffle(x, self.unshuffle)
        h = self.stem(x)
        skips = []
        for i, block in enumerate(self.enc):
            h = block(h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        for up, block in zip(self.up, self.dec):
            h = up(h)
            h = block(torch.cat([h, skips.pop()], dim=1))
        out = self.head(h)
        if self.unshuffle == 1:
            return out
        out = F.pixel_shuffle(out, self.unshuffle)
        if self.refine is not None:
            out = out + self.refine(torch.cat([full, out], dim=1))
        return out


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    return OverlayUNet(in_channels=m.get("in_channels", 7), widths=tuple(m.get("widths", (32, 64, 96, 128))),
                       unshuffle=int(m.get("unshuffle", 1)), refine=int(m.get("refine", 0)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
