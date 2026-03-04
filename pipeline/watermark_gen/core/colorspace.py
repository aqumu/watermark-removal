import numpy as np

def srgb_to_linear(img):
    img = img / 255.0
    mask = img <= 0.04045
    linear = np.where(mask, img / 12.92,
                      ((img + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_srgb(img):
    mask = img <= 0.0031308
    srgb = np.where(mask, img * 12.92,
                    1.055 * (img ** (1 / 2.4)) - 0.055)
    srgb = np.clip(srgb, 0, 1)
    return (srgb * 255).astype(np.uint8)