"""
bench_overlay_throughput.py  photos per second with all cores busy
------------------------------------------------------------------
N worker processes, each with a single-threaded ONNX Runtime session, each
decoding photos, localising the mark, running the model on batches and pasting
the window back. Reports end-to-end photos/second for the whole machine.

Usage:
  python training/bench_overlay_throughput.py --onnx a.onnx [b.onnx ...] [--workers 6] [--n 240] [--batch 8]
"""

import argparse
import csv
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
for p in (REPO_ROOT, TRAINING_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_REMOVER = None


def _init(onnx_path: str):
    global _REMOVER
    import cv2
    cv2.setNumThreads(1)
    from src.tasks.overlay.onnx_remover import OverlayRemover
    _REMOVER = OverlayRemover(onnx_path, threads=1)


def _work(chunk):
    import cv2
    items = [(cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1].copy(), lid) for p, lid in chunk]
    _REMOVER.remove_batch(items)
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from src.tasks.overlay.data import is_holdout
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "data_gen/backgrounds/cian_index.csv", encoding="utf-8"))
            if is_holdout(r["listing_id"], 0.05)]
    random.Random(11).shuffle(rows)
    jobs = [(r["path"], r["listing_id"]) for r in rows[:args.n]]
    chunks = [jobs[i:i + args.batch] for i in range(0, len(jobs), args.batch)]
    for onnx_path in args.onnx:
        with Pool(args.workers, initializer=_init, initargs=(onnx_path,)) as pool:
            pool.map(_work, chunks[:args.workers])          # warm-up
            t = time.perf_counter()
            done = sum(pool.imap_unordered(_work, chunks))
            dt = time.perf_counter() - t
        print(f"{Path(onnx_path).stem:16s} {args.workers} workers: {done / dt:6.1f} photos/s "
              f"({dt / done * 1000:.1f} ms/photo, {done / dt * 3600:,.0f} photos/hour)", flush=True)


if __name__ == "__main__":
    main()
