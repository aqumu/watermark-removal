from __future__ import annotations

from pathlib import Path

from wm_shared.config import load_yaml_config, validate_removal_config

from ...common.restoration import (
    build_store_signature,
    clear_preprocessed_store,
    get_store_root,
    iter_sample_dirs,
    precompute_preprocessed_store,
)


KEEP_DATASET_FILES = {
    "clean.png",
    "watermarked.jpg",
    "mask.png",
    "meta.json",
}


def load_removal_cfg(config_path: str | Path) -> dict:
    return validate_removal_config(load_yaml_config(Path(config_path).resolve()))


def resolve_store_context(cfg: dict,
                          dataset_root: str | Path | None = None,
                          store_root: str | Path | None = None,
                          use_augmented_mask: bool = False) -> dict:
    ds_cfg = cfg["dataset"]
    resolved_dataset_root = Path(dataset_root or ds_cfg["root"]).resolve()
    resolved_store_root = get_store_root(resolved_dataset_root, store_root or ds_cfg.get("preprocessed_store_dir"))
    signature = build_store_signature(
        image_width=ds_cfg["image_width"],
        image_height=ds_cfg["image_height"],
        crop_aspect_ratio=ds_cfg.get("crop_aspect_ratio", 3.54),
        crop_margin_ratio=ds_cfg.get("crop_margin_ratio", 0.10),
        crop_min_width_ratio=ds_cfg.get("crop_min_width_ratio", 0.50),
        use_augmented_mask=use_augmented_mask,
    )
    return {
        "dataset_root": resolved_dataset_root,
        "store_root": resolved_store_root,
        "signature": signature,
        "namespace_path": resolved_store_root / signature,
    }


def clear_store(cfg: dict,
                dataset_root: str | Path | None = None,
                store_root: str | Path | None = None) -> Path:
    ctx = resolve_store_context(cfg, dataset_root=dataset_root, store_root=store_root)
    clear_preprocessed_store(ctx["store_root"], ctx["signature"])
    return ctx["namespace_path"]


def rebuild_store(cfg: dict,
                  dataset_root: str | Path | None = None,
                  store_root: str | Path | None = None,
                  limit: int = 0,
                  clear_first: bool = False) -> tuple[int, Path]:
    """Rebuild both augmented (training) and stable (val/vis) caches."""
    total = 0
    namespace = None
    for aug in [True, False]:
        ctx = resolve_store_context(cfg, dataset_root=dataset_root, store_root=store_root, use_augmented_mask=aug)
        if clear_first:
            clear_preprocessed_store(ctx["store_root"], ctx["signature"])

        ds_cfg = cfg["dataset"]
        samples = iter_sample_dirs(ctx["dataset_root"])
        max_samples = ds_cfg.get("max_samples")
        if max_samples is not None:
            samples = samples[:max_samples]
        if limit > 0:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"No valid samples found in {ctx['dataset_root']}")

        n = precompute_preprocessed_store(
            samples=samples,
            dataset_root=ctx["dataset_root"],
            store_root=ctx["store_root"],
            signature=ctx["signature"],
            image_width=ds_cfg["image_width"],
            image_height=ds_cfg["image_height"],
            crop_aspect_ratio=ds_cfg.get("crop_aspect_ratio", 3.54),
            crop_margin_ratio=ds_cfg.get("crop_margin_ratio", 0.10),
            crop_min_width_ratio=ds_cfg.get("crop_min_width_ratio", 0.50),
            use_augmented_mask=aug,
        )
        total += n
        namespace = ctx["namespace_path"]
        
    return total, namespace


def cleanup_legacy_dataset_sidecars(dataset_root: str | Path, dry_run: bool = False) -> tuple[int, int]:
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    sample_dirs = iter_sample_dirs(root)
    if not sample_dirs:
        raise RuntimeError(f"No valid sample directories found in {root}")

    removed_total = 0
    touched_dirs = 0

    for sample_dir in sample_dirs:
        removed = []
        for path in sample_dir.iterdir():
            if not path.is_file():
                continue
            if path.name in KEEP_DATASET_FILES:
                continue
            removed.append(path)
            if not dry_run:
                path.unlink()

        if removed:
            touched_dirs += 1
            removed_total += len(removed)
            print(f"{sample_dir.name}: removed {len(removed)} file(s)")

    return removed_total, touched_dirs
