from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from .run_manifest import RunManifest, save_latest_pointer


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_jsonable(v) for v in value]
    return value


@dataclass
class ExperimentPaths:
    run_dir: Path
    meta_dir: Path
    preview_dir: Path
    artifact_dir: Path
    tracker_dir: Path


class ExperimentSession:
    """
    Tracker-neutral run metadata and artifact staging.

    The training code writes into this structure today; later, a ClearML adapter
    can mirror the same events without another trainer refactor.
    """

    def __init__(self, task_name: str, cfg: dict, manifest: RunManifest | None = None, dashboard=None):
        logging_cfg = cfg["logging"]
        self.task_name = task_name
        self.manifest = manifest
        self.dashboard = dashboard
        self.run_dir = Path(logging_cfg["dir"])
        if manifest is not None:
            self.run_dir = manifest.paths.run_dir
            self.paths = ExperimentPaths(
                run_dir=manifest.paths.run_dir,
                meta_dir=manifest.paths.meta_dir,
                preview_dir=manifest.paths.preview_dir,
                artifact_dir=manifest.paths.artifact_dir,
                tracker_dir=manifest.paths.tracker_dir,
            )
        else:
            self.paths = ExperimentPaths(
                run_dir=self.run_dir,
                meta_dir=self.run_dir / "meta",
                preview_dir=self.run_dir / "previews",
                artifact_dir=self.run_dir / "artifacts",
                tracker_dir=self.run_dir / "tracker",
            )
        for path in (
            self.paths.run_dir,
            self.paths.meta_dir,
            self.paths.preview_dir,
            self.paths.artifact_dir,
            self.paths.tracker_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.session_id = _now_stamp()
        self._write_run_manifest(cfg)

    def _write_run_manifest(self, cfg: dict) -> None:
        if self.manifest is not None:
            self.manifest.save(self.paths.meta_dir / "run.json")
            manifest = self.manifest.to_dict()
        else:
            manifest = {
                "session_id": self.session_id,
                "task_name": self.task_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "device": cfg.get("device"),
                "seed": cfg.get("seed"),
                "config_path_hints": {
                    "logging_dir": cfg.get("logging", {}).get("dir"),
                    "checkpoint_dir": cfg.get("checkpointing", {}).get("dir"),
                },
            }
        (self.paths.meta_dir / "run.json").write_text(
            json.dumps(_ensure_jsonable(manifest), indent=2),
            encoding="utf-8",
        )
        with open(self.paths.meta_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(_ensure_jsonable(cfg), f, sort_keys=False, allow_unicode=False)

    def set_status(self, status: str) -> None:
        if self.manifest is None:
            return
        self.manifest.set_status(status)
        save_latest_pointer(self.manifest)
        if self.dashboard is not None:
            self.dashboard.set_status(status)

    def log_model_overview(
        self,
        *,
        model_name: str,
        parameter_count: int,
        optimizer_name: str,
        scheduler_name: str,
        extra: dict | None = None,
    ) -> None:
        overview = {
            "model_name": model_name,
            "parameter_count": int(parameter_count),
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "extra": _ensure_jsonable(extra or {}),
        }
        (self.paths.meta_dir / "model_overview.json").write_text(
            json.dumps(overview, indent=2),
            encoding="utf-8",
        )
        if self.dashboard is not None:
            self.dashboard.log_model_overview(overview)

    def log_preview_set(self, step_or_epoch: int, name_to_image: dict[str, object]) -> None:
        target_dir = self.paths.preview_dir / f"{int(step_or_epoch):06d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, image in name_to_image.items():
            out = target_dir / f"{name}.png"
            self._write_image(out, image)
            written[name] = out
        if self.dashboard is not None:
            self.dashboard.log_preview(step_or_epoch, written)

    def stage_artifact(
        self,
        source: str | Path,
        *,
        category: str,
        name: str | None = None,
        copy_file: bool = False,
    ) -> Path:
        source_path = Path(source)
        target_dir = self.paths.artifact_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        label = name or source_path.name
        if copy_file:
            target = target_dir / label
            if source_path.resolve() != target.resolve():
                shutil.copy2(source_path, target)
            if self.dashboard is not None:
                self.dashboard.log_artifact(category=category, path=target, name=label)
            return target

        ref_path = target_dir / f"{label}.json"
        ref_path.write_text(
            json.dumps(
                {
                    "label": label,
                    "source_path": str(source_path),
                    "staged_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ref_path

    @staticmethod
    def _write_image(path: Path, image) -> None:
        if image is None:
            return
        if getattr(image, "dtype", None) is None:
            return
        if len(image.shape) == 2:
            if image.dtype != "uint8":
                # Normalize to [0, 1] by the image's own maximum so the brightest
                # pixel is always white.  This is correct for loss heatmaps (which
                # may have values > 1) and harmless for masks (already in [0, 1]).
                max_val = float(image.max())
                if max_val > 0:
                    img = ((image / max_val).clip(0, 1) * 255).astype("uint8")
                else:
                    img = image.clip(0, 255).astype("uint8")
            else:
                img = image
            cv2.imwrite(str(path), img)
            return

        img = image
        if img.dtype != "uint8":
            img = (img.clip(0, 1) * 255).astype("uint8")
        cv2.imwrite(str(path), img)
