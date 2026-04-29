import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from wm_shared.preprocess import compute_gradient, crop_removal_roi, dilate_mask_input


STORE_VERSION = 1


def build_store_signature(image_width: int,
                          image_height: int,
                          crop_aspect_ratio: float,
                          crop_margin_ratio: float,
                          crop_min_width_ratio: float,
                          use_augmented_mask: bool = False) -> str:
    payload = {
        "version": STORE_VERSION,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "crop_aspect_ratio": float(crop_aspect_ratio),
        "crop_margin_ratio": float(crop_margin_ratio),
        "crop_min_width_ratio": float(crop_min_width_ratio),
        "use_augmented_mask": bool(use_augmented_mask),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    suffix = "-aug" if use_augmented_mask else ""
    return f"removal-{image_width}x{image_height}-{digest}{suffix}"


def get_store_root(dataset_root: str | Path, store_root: str | Path | None = None) -> Path:
    if store_root is not None:
        return Path(store_root)
    return Path(dataset_root).parent / "preprocessed_store"


def iter_sample_dirs(root: str | Path) -> list[Path]:
    return sorted([
        d for d in Path(root).iterdir()
        if d.is_dir()
        and (d / "watermarked.jpg").exists()
        and (d / "clean.png").exists()
        and (d / "mask.png").exists()
    ])


def sample_store_paths(store_root: str | Path,
                       dataset_root: str | Path,
                       sample_dir: str | Path,
                       signature: str) -> dict[str, Path]:
    dataset_root = Path(dataset_root).resolve()
    sample_dir = Path(sample_dir).resolve()
    rel = sample_dir.relative_to(dataset_root)
    base = Path(store_root).resolve() / signature / rel
    return {
        "dir": base,
        "wm": base / "watermarked.jpg",
        "clean": base / "clean.png",
        "mask": base / "mask.png",
        "mask_input": base / "mask_input.png",
    }


def load_preprocessed_sample(store_root: str | Path,
                             dataset_root: str | Path,
                             sample_dir: str | Path,
                             signature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    paths = sample_store_paths(store_root, dataset_root, sample_dir, signature)
    if not all(paths[key].exists() for key in ("wm", "clean", "mask", "mask_input")):
        return None

    wm = cv2.imread(str(paths["wm"]), cv2.IMREAD_COLOR)
    clean = cv2.imread(str(paths["clean"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(paths["mask"]), cv2.IMREAD_GRAYSCALE)
    mask_input = cv2.imread(str(paths["mask_input"]), cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (wm, clean, mask, mask_input)):
        return None

    return (
        wm,
        clean,
        mask.astype(np.float32) / 255.0,
        mask_input.astype(np.float32) / 255.0,
    )


def write_preprocessed_sample(store_root: str | Path,
                              dataset_root: str | Path,
                              sample_dir: str | Path,
                              signature: str,
                              wm: np.ndarray,
                              clean: np.ndarray,
                              mask: np.ndarray,
                              mask_input: np.ndarray) -> None:
    paths = sample_store_paths(store_root, dataset_root, sample_dir, signature)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths["wm"]), wm)
    cv2.imwrite(str(paths["clean"]), clean)
    cv2.imwrite(str(paths["mask"]), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(paths["mask_input"]), (mask_input * 255).astype(np.uint8))


def build_preprocessed_sample(sample_dir: str | Path,
                              image_width: int,
                              image_height: int,
                              crop_aspect_ratio: float,
                              crop_margin_ratio: float,
                              crop_min_width_ratio: float,
                              use_augmented_mask: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_dir = Path(sample_dir)
    wm = cv2.imread(str(sample_dir / "watermarked.jpg"), cv2.IMREAD_COLOR)
    clean = cv2.imread(str(sample_dir / "clean.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (wm, clean, mask)):
        raise FileNotFoundError(f"Unreadable sample files in {sample_dir}")

    wm_r, clean_r, mask_r, _, mask_input = crop_removal_roi(
        wm,
        clean,
        mask.astype(np.float32) / 255.0,
        image_width,
        image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=crop_margin_ratio,
        min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=use_augmented_mask,
    )
    return wm_r, clean_r, mask_r, mask_input


def load_or_build_preprocessed_sample(store_root: str | Path,
                                      dataset_root: str | Path,
                                      sample_dir: str | Path,
                                      signature: str,
                                      image_width: int,
                                      image_height: int,
                                      crop_aspect_ratio: float,
                                      crop_margin_ratio: float,
                                      crop_min_width_ratio: float,
                                      use_augmented_mask: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = load_preprocessed_sample(store_root, dataset_root, sample_dir, signature)
    if cached is not None:
        return cached

    wm, clean, mask, mask_input = build_preprocessed_sample(
        sample_dir,
        image_width=image_width,
        image_height=image_height,
        crop_aspect_ratio=crop_aspect_ratio,
        crop_margin_ratio=crop_margin_ratio,
        crop_min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=use_augmented_mask,
    )
    write_preprocessed_sample(store_root, dataset_root, sample_dir, signature, wm, clean, mask, mask_input)
    return wm, clean, mask, mask_input


def clear_preprocessed_store(store_root: str | Path, signature: str | None = None) -> None:
    root = Path(store_root)
    target = root / signature if signature else root
    if target.exists():
        shutil.rmtree(target)


def precompute_preprocessed_store(samples: list[Path],
                                  dataset_root: str | Path,
                                  store_root: str | Path,
                                  signature: str,
                                  image_width: int,
                                  image_height: int,
                                  crop_aspect_ratio: float,
                                  crop_margin_ratio: float,
                                  crop_min_width_ratio: float,
                                  use_augmented_mask: bool = False) -> int:
    total = len(samples)
    for i, sample_dir in enumerate(samples, start=1):
        load_or_build_preprocessed_sample(
            store_root=store_root,
            dataset_root=dataset_root,
            sample_dir=sample_dir,
            signature=signature,
            image_width=image_width,
            image_height=image_height,
            crop_aspect_ratio=crop_aspect_ratio,
            crop_margin_ratio=crop_margin_ratio,
            crop_min_width_ratio=crop_min_width_ratio,
            use_augmented_mask=use_augmented_mask,
        )
        if i == 1 or i == total or i % 100 == 0:
            print(f"[{i}/{total}] prepared {Path(sample_dir).name}")
    return total


def prepare_roi_input(wm: np.ndarray,
                      mask_u8: np.ndarray,
                      model_width: int,
                      model_height: int,
                      dilate: bool = False,
                      crop_aspect_ratio: float = 3.54,
                      crop_margin_ratio: float = 0.10,
                      crop_min_width_ratio: float = 0.50,
                      use_augmented_mask: bool = False):
    mask_f = mask_u8.astype(np.float32) / 255.0
    wm_r, _, mask_soft_r, roi, crop_mask_r = crop_removal_roi(
        wm,
        wm,
        mask_f,
        model_width,
        model_height,
        crop_aspect_ratio=crop_aspect_ratio,
        margin_ratio=crop_margin_ratio,
        min_width_ratio=crop_min_width_ratio,
        use_augmented_mask=use_augmented_mask,
    )
    mask_binary = (mask_soft_r > 0.5).astype(np.float32)
    mask_input = dilate_mask_input(mask_binary, image_size=max(model_width, model_height)) if dilate else crop_mask_r

    rgb = cv2.cvtColor(wm_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mask_t = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0)

    grad_t = compute_gradient(wm_r)
    grad_np = (grad_t.squeeze().numpy() * 255).astype(np.uint8)
    grad_t = grad_t.unsqueeze(0)

    inp = torch.cat([rgb_t, mask_t, grad_t], dim=1)
    return inp, wm_r, mask_input, grad_np, mask_binary, roi


def blend_back(pred_bgr: np.ndarray,
               orig_wm: np.ndarray,
               orig_mask: np.ndarray,
               roi: dict,
               feather: int = 9,
               mask_expand: int = 0) -> np.ndarray:
    out = orig_wm.copy()
    crop_w = int(roi["width"])
    crop_h = int(roi["height"])
    x0 = int(roi["x0"])
    y0 = int(roi["y0"])
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    orig_h, orig_w = orig_wm.shape[:2]
    ox0 = max(0, x0)
    oy0 = max(0, y0)
    ox1 = min(orig_w, x1)
    oy1 = min(orig_h, y1)
    if ox0 >= ox1 or oy0 >= oy1:
        return out

    pred_up = cv2.resize(pred_bgr, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    px0 = ox0 - x0
    py0 = oy0 - y0
    px1 = px0 + (ox1 - ox0)
    py1 = py0 + (oy1 - oy0)

    pred_crop = pred_up[py0:py1, px0:px1]
    orig_crop = orig_wm[oy0:oy1, ox0:ox1]

    scale = crop_w / pred_bgr.shape[1]
    if feather > 0 or mask_expand > 0:
        feather_scaled = max(1, round(feather * scale))
        mask_expand_scaled = round(mask_expand * scale)

        working_mask = orig_mask
        if mask_expand_scaled > 0:
            exp_k = mask_expand_scaled * 2 + 1
            working_mask = cv2.dilate(
                working_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exp_k, exp_k)),
            )
        if feather_scaled > 0:
            dil_k = feather_scaled * 2 + 1
            working_mask = cv2.dilate(
                working_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)),
            )
            working_mask = cv2.GaussianBlur(
                working_mask.astype(np.float32),
                (dil_k, dil_k),
                feather_scaled / 2,
            )
        mask_crop = working_mask[oy0:oy1, ox0:ox1]
        m = (mask_crop / 255.0)[:, :, None]
    else:
        mask_crop = orig_mask[oy0:oy1, ox0:ox1]
        m = (mask_crop > 127)[:, :, None].astype(np.float32)

    out[oy0:oy1, ox0:ox1] = (pred_crop * m + orig_crop * (1 - m)).clip(0, 255).astype(np.uint8)
    return out
