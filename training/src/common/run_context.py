from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wm_shared.run_manifest import (
    RunManifest,
    create_run_manifest,
    load_run_manifest,
    make_code_fingerprint,
    make_config_fingerprint,
    prune_old_runs,
    save_latest_pointer,
)


@dataclass
class RunContext:
    manifest: RunManifest
    family_root: Path
    checkpoint_dir: Path
    config_fingerprint: str


def _copy_cfg_without_runtime_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    copied = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            copied[key] = _copy_cfg_without_runtime_paths(value)
        elif isinstance(value, list):
            copied[key] = [
                _copy_cfg_without_runtime_paths(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            copied[key] = value

    logging_cfg = copied.get("logging", {})
    checkpoint_cfg = copied.get("checkpointing", {})
    logging_cfg.pop("dir", None)
    checkpoint_cfg.pop("dir", None)
    return copied


def _project_run_name(cfg: dict[str, Any], task_name: str) -> str:
    logging_dir = Path(cfg["logging"]["dir"])
    return cfg["logging"].get("project_run") or logging_dir.name or task_name


def _load_parent_manifest(checkpoint_path: str | Path | None) -> RunManifest | None:
    if not checkpoint_path:
        return None

    path = Path(checkpoint_path).resolve()
    for parent in (path,) + tuple(path.parents):
        run_json = parent / "meta" / "run.json"
        if run_json.exists():
            return load_run_manifest(run_json)
    return None


def _checkpoint_epoch(checkpoint_path: str | Path | None) -> int | None:
    if not checkpoint_path:
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    epoch = payload.get("epoch")
    return int(epoch) if isinstance(epoch, int) else None


def prepare_run_context(
    *,
    task_name: str,
    cfg: dict[str, Any],
    config_path: str | Path,
    resume: str | None = None,
    load_weights: str | None = None,
    repo_root: str | Path | None = None,
    force_continue: bool = False,
) -> RunContext:
    family_root = Path(cfg["logging"]["dir"]).resolve()
    # Safety: if logging.dir points to a specific run folder, use the parent family folder.
    if (family_root / "meta" / "run.json").exists():
        family_root = family_root.parent
    family_root.mkdir(parents=True, exist_ok=True)

    project_run = _project_run_name(cfg, task_name)
    config_fingerprint = make_config_fingerprint(_copy_cfg_without_runtime_paths(cfg))
    code_fingerprint = make_code_fingerprint(repo_root)

    start_mode = "fresh"
    history_mode = "fork"
    parent_manifest = None
    parent_checkpoint = None

    if resume:
        start_mode = "resume"
        parent_checkpoint = Path(resume).resolve()
        parent_manifest = _load_parent_manifest(parent_checkpoint)
    elif load_weights:
        start_mode = "load_weights"
        parent_checkpoint = Path(load_weights).resolve()
        parent_manifest = _load_parent_manifest(parent_checkpoint)

    parent_run_id = parent_manifest.identity.run_id if parent_manifest else None
    parent_timeline_id = parent_manifest.identity.timeline_id if parent_manifest else None
    resumed_from_epoch = _checkpoint_epoch(parent_checkpoint) if start_mode == "resume" else None

    timeline_id = None
    if start_mode == "resume" and parent_manifest is not None:
        if force_continue or parent_manifest.identity.config_fingerprint == config_fingerprint:
            history_mode = "continue"
            timeline_id = parent_manifest.identity.timeline_id
        else:
            history_mode = "fork"

    # When resuming the same config, reuse the existing run directory instead of
    # creating a new one — this preserves the run_id, metrics CSVs, and previews.
    if start_mode == "resume" and history_mode == "continue" and parent_manifest is not None:
        manifest = parent_manifest
        manifest.identity.status = "running"
        manifest.identity.lineage.start_mode = "resume"
        manifest.identity.lineage.parent_checkpoint = str(parent_checkpoint) if parent_checkpoint else None
        manifest.identity.lineage.resumed_from_epoch = resumed_from_epoch
        manifest.save()
    else:
        manifest = create_run_manifest(
            project_run=project_run,
            task_name=task_name,
            root_dir=family_root,
            start_mode=start_mode,
            history_mode=history_mode,
            config_fingerprint=config_fingerprint,
            code_fingerprint=code_fingerprint,
            device=cfg.get("device"),
            parent_run_id=parent_run_id,
            parent_timeline_id=parent_timeline_id,
            parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None,
            resumed_from_epoch=resumed_from_epoch,
            timeline_id=timeline_id,
        )

    checkpoint_dir = manifest.paths.artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg["logging"]["dir"] = str(manifest.paths.run_dir)
    cfg.setdefault("checkpointing", {})
    cfg["checkpointing"]["dir"] = str(checkpoint_dir)
    cfg["logging"]["project_run"] = project_run
    cfg["logging"]["family_dir"] = str(family_root)
    cfg["logging"]["run_id"] = manifest.identity.run_id
    cfg["logging"]["timeline_id"] = manifest.identity.timeline_id
    cfg["logging"]["config_path"] = str(Path(config_path).resolve())

    manifest.save()
    save_latest_pointer(manifest)

    # Save a physical copy of the config for future reference and dashboard resumption
    config_copy = manifest.paths.meta_dir / "config.yaml"
    import shutil
    shutil.copy2(config_path, config_copy)

    keep_latest_runs = int(cfg["logging"].get("keep_latest_runs", 15))
    prune_old_runs(family_root, keep_latest_runs, preserve_run_id=manifest.identity.run_id)

    return RunContext(
        manifest=manifest,
        family_root=family_root,
        checkpoint_dir=checkpoint_dir,
        config_fingerprint=config_fingerprint,
    )
