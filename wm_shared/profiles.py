from pathlib import Path

import yaml


DEFAULT_PROFILE = {
    "name": "default",
    "watermark_asset": "data_gen/watermark.png",
    "aspect_ratio": 3.54,
    "placement": {
        "width_fraction": [0.76, 0.82],
        "center_jitter": True,
    },
    "removal_crop": {
        "aspect_ratio": 3.54,
        "margin_ratio": 0.12,
        "min_width_ratio": 0.50,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_profile(path: str | Path | None = None) -> dict:
    if path is None:
        return dict(DEFAULT_PROFILE)

    profile_path = Path(path)
    with open(profile_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_PROFILE, data)


def get_crop_settings(cfg: dict, profile: dict | None = None) -> tuple[float, float, float]:
    profile = DEFAULT_PROFILE if profile is None else profile
    ds_cfg = cfg.get("dataset", {})
    crop_cfg = profile.get("removal_crop", {})
    aspect_ratio = float(ds_cfg.get("crop_aspect_ratio", crop_cfg.get("aspect_ratio", DEFAULT_PROFILE["removal_crop"]["aspect_ratio"])))
    margin_ratio = float(ds_cfg.get("crop_margin_ratio", crop_cfg.get("margin_ratio", DEFAULT_PROFILE["removal_crop"]["margin_ratio"])))
    min_width_ratio = float(ds_cfg.get("crop_min_width_ratio", crop_cfg.get("min_width_ratio", DEFAULT_PROFILE["removal_crop"]["min_width_ratio"])))
    return aspect_ratio, margin_ratio, min_width_ratio
