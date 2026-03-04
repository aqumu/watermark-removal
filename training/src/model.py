"""
Masked U-Net for semi-transparent watermark removal
----------------------------------------------------
Input  : 4 channels  (RGB watermarked in [-1,1]  +  binary mask in {0,1})
Output : 3 channels  (RGB residual delta in [0,1])
         pred_clean = watermarked_rgb − model_output   (clamped to [-1,1])

The network predicts how much white to subtract from each pixel rather than
reconstructing the clean image directly.  This is more robust:
  - Interior : learns a consistent small positive offset to remove
  - Edges    : learns to predict near-zero where watermark fades to nothing
  - Outside  : naturally outputs ≈0, leaving clean pixels untouched

Architecture
  Encoder : N blocks of  Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
  Bridge  : same double-conv without pooling
  Decoder : Upsample(bilinear) → concat(skip) → double-conv
  Head    : 1×1 Conv(C→3) + Sigmoid

The mask is concatenated as the 4th input channel so the network can
distinguish masked (watermarked) pixels from clean context at every level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# building blocks
# ──────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv-BN-ReLU × 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip   # (pooled, skip-connection)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# U-Net
# ──────────────────────────────────────────────────────────────────────────────

class MaskedUNet(nn.Module):
    """
    Parameters
    ----------
    base_channels : feature width at the first encoder stage.
                    Subsequent stages double: base, 2×, 4×, 8×
    depth         : number of encoder/decoder stages (≥ 2, ≤ 5)
    in_channels   : 4  (RGB + mask)
    out_channels  : 3  (RGB)
    """

    def __init__(self, base_channels: int = 32, depth: int = 4,
                 in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        assert 2 <= depth <= 5, "depth must be between 2 and 5"

        chs = [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.encoders.append(EncoderBlock(prev, c))
            prev = c

        # bridge
        bridge_ch = chs[-1] * 2
        self.bridge = DoubleConv(prev, bridge_ch)
        prev = bridge_ch

        # decoder (reversed channel list)
        self.decoders = nn.ModuleList()
        for c in reversed(chs):
            self.decoders.append(DecoderBlock(prev, c, c))
            prev = c

        # output head — predicts residual delta in [0, 1]
        self.head = nn.Sequential(
            nn.Conv2d(prev, out_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx4xHxW → out: Bx3xHxW"""
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bridge(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# convenience
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> MaskedUNet:
    return MaskedUNet(
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
