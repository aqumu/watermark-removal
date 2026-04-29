import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.tasks.removal.store_cli import load_removal_cfg, rebuild_store  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "training" / "configs" / "train_512.yaml"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--store-root", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clear-first", action="store_true")
    args = parser.parse_args()

    cfg = load_removal_cfg(args.config)
    total, namespace = rebuild_store(
        cfg,
        dataset_root=args.dataset_root,
        store_root=args.store_root,
        limit=args.limit,
        clear_first=args.clear_first,
    )
    print(f"Done. Prepared {total} samples in {namespace}")


if __name__ == "__main__":
    main()
