from ..removal.store_cli import (
    cleanup_legacy_dataset_sidecars,
    clear_store,
    load_removal_cfg,
    rebuild_store,
    resolve_store_context,
)


def load_restoration_cfg(config_path):
    return load_removal_cfg(config_path)


__all__ = [
    "cleanup_legacy_dataset_sidecars",
    "clear_store",
    "load_restoration_cfg",
    "rebuild_store",
    "resolve_store_context",
]
