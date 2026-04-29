"""
Direct clean-image predictor for semi-transparent watermark removal.

This task keeps the same 5-channel input contract as the delta-based removal
model, but the network returns the restored clean RGB image directly.
To preserve the identity mapping at initialisation, the head predicts a bounded
RGB correction that is added to the watermarked input inside forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        mid_ch = in_ch // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(mid_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DirectCleanUNet(nn.Module):
    def __init__(self, base_channels: int = 32, depth: int = 4,
                 in_channels: int = 5, out_channels: int = 3,
                 use_checkpoint: bool = False):
        super().__init__()
        assert 2 <= depth <= 5, "depth must be between 2 and 5"

        self.use_checkpoint = use_checkpoint
        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.encoders.append(EncoderBlock(prev, c))
            prev = c

        bridge_ch = chs[-1] * 2
        self.bridge = DoubleConv(prev, bridge_ch)
        prev = bridge_ch

        self.decoders = nn.ModuleList()
        for c in reversed(chs):
            self.decoders.append(DecoderBlock(prev, c, c))
            prev = c

        self.head = nn.Conv2d(prev, out_channels, 1)
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

        # Zero head => zero correction => exact identity at step 0.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        work = x
        for enc in self.encoders:
            if self.use_checkpoint and work.requires_grad:
                work, skip = grad_checkpoint(enc, work, use_reentrant=False)
            else:
                work, skip = enc(work)
            skips.append(skip)

        if self.use_checkpoint and work.requires_grad:
            work = grad_checkpoint(self.bridge, work, use_reentrant=False)
        else:
            work = self.bridge(work)

        for dec, skip in zip(self.decoders, reversed(skips)):
            if self.use_checkpoint and work.requires_grad:
                work = grad_checkpoint(dec, work, skip, use_reentrant=False)
            else:
                work = dec(work, skip)

        correction = 2.0 * torch.tanh(self.head(work))
        return (x[:, :3] + correction).clamp(-1, 1)


class _DirectCleanWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = 2.0 * torch.tanh(self.backbone(x))
        return (x[:, :3] + correction).clamp(-1, 1)


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m.get("type", "scratch") == "pretrained":
        import segmentation_models_pytorch as smp
        backbone = smp.Unet(
            encoder_name=m.get("encoder", "efficientnet-b0"),
            encoder_weights=m.get("encoder_weights", "imagenet"),
            in_channels=5,
            classes=3,
            activation=None,
        )
        return _DirectCleanWrapper(backbone)
    return DirectCleanUNet(
        base_channels=m["base_channels"],
        depth=m["depth"],
        use_checkpoint=m.get("use_checkpoint", False),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
