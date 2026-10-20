import cv2
import numpy as np
import random

def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, 1)

def webp_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    _, enc = cv2.imencode(".webp", img, encode_param)
    return cv2.imdecode(enc, 1)

def resize_artifact(img, downscale_range):
    factor = random.uniform(*downscale_range)
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*factor), int(h*factor)))
    return cv2.resize(small, (w,h))

def gaussian_noise(img, std):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def degrade(img, config):
    operations = [
        lambda x: jpeg_compress(x, random.randint(*config["degradation"]["jpeg_quality"])),
        lambda x: webp_compress(x, random.randint(*config["degradation"]["webp_quality"])),
        lambda x: resize_artifact(x, config["degradation"]["downscale_range"]),
    ]

    k = random.randint(1, min(2, len(operations)))
    selected = random.sample(operations, k)

    for op in selected:
        img = op(img)

    return img