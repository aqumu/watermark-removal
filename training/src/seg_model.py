"""
Segmentation model for watermark mask prediction
-------------------------------------------------
Input  : 3-ch RGB watermarked image (normalised to ImageNet mean/std)
Output : 1-ch soft mask in [0, 1] (sigmoid activation)

Uses segmentation_models_pytorch with a pretrained EfficientNet-B0 encoder
(ImageNet weights) for faster convergence than a from-scratch U-Net.
"""

import segmentation_models_pytorch as smp


def build_seg_model(cfg: dict):
    model_cfg = cfg["model"]
    encoder   = model_cfg.get("encoder", "efficientnet-b0")
    weights   = model_cfg.get("encoder_weights", "imagenet")

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
