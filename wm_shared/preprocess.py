import cv2
import numpy as np
import torch


def _resize_to_shape(image: np.ndarray, width: int, height: int, is_mask: bool = False) -> np.ndarray:
    """Resize an image or mask with interpolation chosen by direction."""
    h, w = image.shape[:2]
    if h == height and w == width:
        return image.copy()

    shrinking = h > height or w > width
    if is_mask:
        interp = cv2.INTER_LINEAR if image.dtype != np.uint8 else cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interp)


def _crop_with_reflect_padding(image: np.ndarray,
                               x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop a possibly out-of-bounds rectangle by reflect-padding the source first."""
    h, w = image.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    x0 += pad_left
    x1 += pad_left
    y0 += pad_top
    y1 += pad_top
    return image[y0:y1, x0:x1]


def make_fixed_aspect_crop(mask: np.ndarray,
                           crop_aspect_ratio: float,
                           margin_ratio: float = 0.10,
                           min_width_ratio: float = 0.50) -> dict:
    """
    Build a fixed-aspect rectangular crop centered on the watermark mask.

    Returns a dict with x0, y0, width, height, x1, y1 in original image coordinates.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    if crop_aspect_ratio <= 0:
        raise ValueError("crop_aspect_ratio must be > 0")

    h, w = mask.shape
    binary = (mask > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        crop_w = max(1, int(round(w * min_width_ratio)))
        crop_h = max(1, int(round(crop_w / crop_aspect_ratio)))
        cx = w / 2.0
        cy = h / 2.0
    else:
        x_min, x_max = xs.min(), xs.max() + 1
        y_min, y_max = ys.min(), ys.max() + 1
        box_w = x_max - x_min
        box_h = y_max - y_min

        margin_x = int(round(box_w * margin_ratio))
        margin_y = int(round(box_h * margin_ratio))
        box_w += 2 * margin_x
        box_h += 2 * margin_y

        crop_w = max(box_w, int(round(box_h * crop_aspect_ratio)))
        crop_h = max(box_h, int(round(crop_w / crop_aspect_ratio)))
        crop_w = max(crop_w, int(round(w * min_width_ratio)))
        crop_h = max(crop_h, int(round(crop_w / crop_aspect_ratio)))
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

    crop_w = int(crop_w)
    crop_h = int(crop_h)
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return {
        "x0": x0,
        "y0": y0,
        "width": crop_w,
        "height": crop_h,
        "x1": int(x1),
        "y1": int(y1),
    }


def crop_removal_roi(wm: np.ndarray,
                     clean: np.ndarray,
                     mask: np.ndarray,
                     width: int,
                     height: int,
                     crop_aspect_ratio: float = 3.54,
                     margin_ratio: float = 0.10,
                     min_width_ratio: float = 0.50,
                     use_augmented_mask: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Extract a fixed-aspect, mask-guided ROI for removal training or inference."""
    crop_mask = dilate_mask_input(mask, augment=use_augmented_mask, image_size=max(width, height))
    roi = make_fixed_aspect_crop(
        crop_mask,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=margin_ratio,
        min_width_ratio=min_width_ratio,
    )
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    x1 = int(roi["x1"])
    y1 = int(roi["y1"])

    wm_crop = _crop_with_reflect_padding(wm, x0, y0, x1, y1)
    clean_crop = _crop_with_reflect_padding(clean, x0, y0, x1, y1)
    mask_crop = _crop_with_reflect_padding(mask, x0, y0, x1, y1)
    crop_mask_crop = _crop_with_reflect_padding(crop_mask, x0, y0, x1, y1)

    wm_sq = _resize_to_shape(wm_crop, width, height, is_mask=False)
    clean_sq = _resize_to_shape(clean_crop, width, height, is_mask=False)
    mask_sq = _resize_to_shape(mask_crop, width, height, is_mask=False).astype(np.float32)
    crop_mask_sq = _resize_to_shape(crop_mask_crop, width, height, is_mask=False).astype(np.float32)

    if mask_sq.max() > 1.0:
        mask_sq /= 255.0
    if crop_mask_sq.max() > 1.0:
        crop_mask_sq /= 255.0

    return wm_sq, clean_sq, mask_sq, roi, crop_mask_sq


def crop_by_roi(image: np.ndarray,
                roi: dict,
                width: int,
                height: int,
                is_mask: bool = False) -> np.ndarray:
    """Extract an explicit ROI, pad if needed, then resize to the target shape."""
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    crop_w = int(roi["width"])
    crop_h = int(roi["height"])
    crop = _crop_with_reflect_padding(image, x0, y0, x0 + crop_w, y0 + crop_h)
    return _resize_to_shape(crop, width, height, is_mask=is_mask)


def compute_gradient(bgr: np.ndarray) -> torch.Tensor:
    """Compute normalised grayscale Sobel gradient magnitude."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val
    return torch.from_numpy(mag).unsqueeze(0)


def dilate_mask_input(mask: np.ndarray, augment: bool = False,
                      image_size: int = 256) -> np.ndarray:
    """Apply fixed 5px ELLIPSE dilation to produce the model's mask input hint.

    If augment=True, additionally applies random scale and translation jitter
    to simulate imperfect alignment from the segmentation pipeline.
    """
    binary = (mask >= 0.5).astype(np.uint8)
    
    if augment:
        import random
        # Reduced range of possible scales by 3x (+/- 0.05) and positional jitter by 2x (+/- 5%)
        scale = random.uniform(0.95, 1.05)
        
        max_shift = int(image_size * 0.05)
        tx = random.uniform(-max_shift, max_shift)
        ty = random.uniform(-max_shift, max_shift)
        
        h, w = binary.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary.astype(np.float32)


def blur_mask_for_loss(mask: np.ndarray, blur_pct: float,
                       image_size: int | tuple[int, int]) -> np.ndarray:
    """Apply Gaussian blur to the ideal mask before it is used for loss weighting."""
    if blur_pct <= 0:
        return mask

    if isinstance(image_size, tuple):
        image_size = max(image_size)
    sigma = image_size * (blur_pct / 100.0)
    ksize = max(1, int(round(sigma * 6)) | 1)
    mask_blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma)
    return np.clip(mask_blurred, 0, 1)
