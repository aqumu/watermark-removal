import cv2
import json
from pathlib import Path

def save_sample(base_dir, idx, clean, watermarked, mask, meta):
    sample_dir = Path(base_dir) / f"sample_{idx:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sample_dir / "clean.png"), clean)
    cv2.imwrite(str(sample_dir / "watermarked.jpg"), watermarked)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)

    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)