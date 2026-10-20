from __future__ import annotations

from pathlib import Path

from wm_shared.config import load_yaml_config, validate_removal_config


def load_removal_cfg(config_path: str | Path) -> dict:
    return validate_removal_config(load_yaml_config(Path(config_path).resolve()))
