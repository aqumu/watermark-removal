"""
remove_overlay.py  remove the Cian mark from photos with an exported ONNX model
------------------------------------------------------------------------------
Usage:
  python training/remove_overlay.py --onnx student.onnx --listing-id 333768830 in.jpg out.png
  python training/remove_overlay.py --onnx student.onnx --csv jobs.csv --out-dir cleaned/
      (jobs.csv columns: path, listing_id)

Only the window around the mark is changed; the rest of the image is passed
through. Output format follows the output file extension (PNG keeps it lossless).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tasks.overlay.onnx_remover import OverlayRemover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mark", default="cian")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--listing-id")
    ap.add_argument("--csv")
    ap.add_argument("--out-dir", default="cleaned")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    remover = OverlayRemover(args.onnx, args.mark, threads=args.threads)
    if args.csv:
        jobs = [(r["path"], r["listing_id"], str(Path(args.out_dir) / (Path(r["path"]).stem + ".png")))
                for r in csv.DictReader(open(args.csv, encoding="utf-8"))]
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    else:
        if len(args.paths) != 2 or not args.listing_id:
            ap.error("give --listing-id and <input> <output>, or --csv")
        jobs = [(args.paths[0], args.listing_id, args.paths[1])]

    t0 = time.time()
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        images = [(cv2.imread(src, cv2.IMREAD_COLOR)[..., ::-1].copy(), lid) for src, lid, _ in chunk]
        for (out, info), (_, _, dst) in zip(remover.remove_batch(images), chunk):
            cv2.imwrite(dst, out[..., ::-1])
    print(f"{len(jobs)} images in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
