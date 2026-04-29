from pathlib import Path

import yaml

from .profiles import get_crop_settings, load_profile


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile_ref = cfg.get("watermark_profile")
    if profile_ref:
        profile_path = (config_path.parent / profile_ref).resolve()
        profile = load_profile(profile_path)
        cfg["watermark_profile_resolved"] = str(profile_path)
    else:
        profile = load_profile()

    cfg["watermark_profile_data"] = profile

    ds_cfg = cfg.setdefault("dataset", {})
    aspect_ratio, margin_ratio, min_width_ratio = get_crop_settings(cfg, profile)
    ds_cfg.setdefault("crop_aspect_ratio", aspect_ratio)
    ds_cfg.setdefault("crop_margin_ratio", margin_ratio)
    ds_cfg.setdefault("crop_min_width_ratio", min_width_ratio)

    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg.setdefault("keep_latest_runs", 15)

    dashboard_cfg = cfg.setdefault("dashboard", {})
    dashboard_cfg.setdefault("enabled", False)
    dashboard_cfg.setdefault("host", "0.0.0.0")
    dashboard_cfg.setdefault("port", 8765)
    dashboard_cfg.setdefault("open_browser", True)
    return cfg


def validate_removal_config(cfg: dict) -> dict:
    ds_cfg = cfg.get("dataset", {})
    missing = [key for key in ("image_width", "image_height") if key not in ds_cfg]
    if missing:
        missing_str = ", ".join(f"dataset.{key}" for key in missing)
        raise KeyError(
            f"Removal config requires explicit rectangular ROI dimensions: missing {missing_str}. "
            "dataset.image_size is no longer supported for removal."
        )

    for key in ("image_width", "image_height"):
        value = ds_cfg[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"dataset.{key} must be a positive integer, got {value!r}")

    return cfg
