import cv2
import json
from pathlib import Path

def save_sample(base_dir, idx, clean, watermarked, mask, meta):
    sample_dir = Path(base_dir) / f"sample_{idx:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # original massive resolution
    cv2.imwrite(str(sample_dir / "clean.png"), clean)
    cv2.imwrite(str(sample_dir / "watermarked.jpg"), watermarked)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)

    # 512px downscaled training cache
    clean_512 = cv2.resize(clean, (512, 512), interpolation=cv2.INTER_AREA)
    wm_512    = cv2.resize(watermarked, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512  = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(sample_dir / "clean_512.png"), clean_512)
    cv2.imwrite(str(sample_dir / "watermarked_512.jpg"), wm_512)
    cv2.imwrite(str(sample_dir / "mask_512.png"), mask_512)

    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)