import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.tasks.removal.store_cli import cleanup_legacy_dataset_sidecars  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data_gen" / "dataset"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    removed_total, touched_dirs = cleanup_legacy_dataset_sidecars(
        dataset_root=args.dataset_root,
        dry_run=args.dry_run,
    )
    mode = "Would remove" if args.dry_run else "Removed"
    print(f"{mode} {removed_total} file(s) across {touched_dirs} sample directorie(s)")


if __name__ == "__main__":
    main()
