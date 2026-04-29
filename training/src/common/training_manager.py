from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml

from wm_shared.run_manifest import load_latest_manifest, load_run_manifest

from .orchestration_state import OrchestrationState
from .training_control import clear_pause_request, request_pause


DEFAULT_DRAFT_FIELDS = {
    "seed": 42,
    "dashboard": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8765,
        "open_browser": False,
    },
}


@dataclass
class DraftConfigRecord:
    family_name: str
    template_id: str
    task_type: str
    path: Path
    data: dict[str, Any]


@dataclass
class TemplateRecord:
    template_id: str
    label: str
    task_type: str
    path: Path
    data: dict[str, Any]


@dataclass
class ActiveJobRecord:
    command: list[str]
    process: subprocess.Popen[str]
    family_name: str
    template_id: str
    task_type: str
    draft_path: Path
    launch_config_path: Path
    mode: str
    checkpoint_path: str | None = None
    previous_run_id: str | None = None
    run_id: str | None = None
    run_dir: Path | None = None
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    pause_requested: bool = False


@dataclass
class CheckpointRecord:
    run_id: str
    project_run: str
    task_name: str
    checkpoint_name: str
    checkpoint_path: Path
    checkpoint_url: str
    artifact_dir: Path
    staged_at: str | None = None
    source_path: str | None = None
    kind: str = "checkpoint"
    epoch: int | None = None
    status: str | None = None
    family_root: Path | None = None


@dataclass(frozen=True)
class RunStartRequest:
    family_name: str
    mode: str = "fresh"
    checkpoint_path: str | None = None
    checkpoint_run_id: str | None = None
    checkpoint_name: str | None = None
    config_path: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RunStartRequest":
        family_name = str(payload.get("family_name", "")).strip()
        mode = _normalize_start_mode(payload.get("mode"))
        checkpoint_path = payload.get("checkpoint_path")
        checkpoint_run_id = payload.get("checkpoint_run_id")
        checkpoint_name = payload.get("checkpoint_name")
        return cls(
            family_name=family_name,
            mode=mode,
            checkpoint_path=str(checkpoint_path) if checkpoint_path not in {None, ""} else None,
            checkpoint_run_id=str(checkpoint_run_id) if checkpoint_run_id not in {None, ""} else None,
            checkpoint_name=str(checkpoint_name) if checkpoint_name not in {None, ""} else None,
            config_path=str(payload.get("config_path")) if payload.get("config_path") else None,
        )

    def normalized_checkpoint_path(self) -> str | None:
        if self.checkpoint_path in {None, ""}:
            return None
        return self.checkpoint_path

    def validate(self) -> None:
        if not self.family_name:
            raise ValueError("Run family name is required.")
        checkpoint_path = self.normalized_checkpoint_path()
        if self.mode == "fresh":
            if checkpoint_path is not None:
                raise ValueError("Fresh starts must not include a checkpoint path.")
            return
        if self.mode not in {"load_weights", "resume"}:
            raise ValueError(f"Unsupported start mode: {self.mode}")
        if checkpoint_path is None:
            raise ValueError(f"Checkpoint path is required for '{self.mode}' starts.")


def _legacy_manifest_from_payload(run_dir: Path, payload: dict[str, Any]):
    from wm_shared.run_manifest import RunIdentity, RunLineage, RunManifest, RunPaths

    session_id = str(payload.get("session_id") or run_dir.name)
    task_name = str(payload.get("task_name") or "unknown")
    lineage = RunLineage(start_mode="legacy", history_mode="legacy")
    identity = RunIdentity(
        project_run=run_dir.name,
        run_id=run_dir.name,
        timeline_id=session_id,
        task_name=task_name,
        created_at=str(payload.get("created_at") or ""),
        status=str(payload.get("status") or "unknown"),
        config_fingerprint=str(payload.get("config_fingerprint") or ""),
        code_fingerprint=str(payload.get("code_fingerprint") or ""),
        device=str(payload.get("device")) if payload.get("device") is not None else None,
        lineage=lineage,
    )
    paths = RunPaths(
        root_dir=run_dir.parent,
        run_dir=run_dir,
        meta_dir=run_dir / "meta",
        preview_dir=run_dir / "previews",
        artifact_dir=run_dir / "artifacts",
        tracker_dir=run_dir / "tracker",
        train_csv=run_dir / "train.csv",
        val_csv=run_dir / "val.csv",
    )
    return RunManifest(identity=identity, paths=paths)


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("_-")
    return cleaned or "draft"


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any], _depth: int = 0) -> dict[str, Any]:
    if _depth > 20:
        raise ValueError("Config nesting exceeds maximum depth of 20 levels")
    result = _deep_copy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value, _depth + 1)
        else:
            result[key] = _deep_copy(value)
    return result


def _strip_runtime_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    copied = _deep_copy(cfg)
    for key in ("seed",):
        copied.pop(key, None)
    dashboard_cfg = copied.get("dashboard")
    if isinstance(dashboard_cfg, dict):
        for key in ("enabled", "host", "port", "open_browser"):
            dashboard_cfg.pop(key, None)
        if not dashboard_cfg:
            copied.pop("dashboard", None)
    logging_cfg = copied.get("logging")
    if isinstance(logging_cfg, dict):
        for key in ("dir", "family_dir", "run_id", "timeline_id", "project_run", "config_path"):
            logging_cfg.pop(key, None)
        if not logging_cfg:
            copied.pop("logging", None)
    checkpointing_cfg = copied.get("checkpointing")
    if isinstance(checkpointing_cfg, dict):
        checkpointing_cfg.pop("dir", None)
        if not checkpointing_cfg:
            copied.pop("checkpointing", None)
    return copied


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _infer_task_type(cfg: dict[str, Any], path: Path) -> str:
    dataset = cfg.get("dataset", {})
    if isinstance(dataset, dict):
        dashboard = cfg.get("dashboard", {})
        if isinstance(dashboard, dict):
            task_type = dashboard.get("task_type")
            if isinstance(task_type, str) and task_type.strip():
                return _normalize_task_type(task_type)
        if "image_size" in dataset:
            return "segmentation"
        if "image_width" in dataset or "image_height" in dataset:
            filename = path.stem.lower()
            if "restoration" in filename:
                return "restoration"
            return "removal"

    filename = path.stem.lower()
    if "seg" in filename:
        return "segmentation"
    return "removal"


def _normalize_task_type(value: str | None) -> str:
    if not value:
        return "removal"
    normalized = value.strip().lower()
    if normalized in {"seg", "segmentation", "detection"}:
        return "segmentation"
    if normalized == "removal":
        return "removal"
    if normalized == "restoration":
        return "restoration"
    return normalized


def _normalize_start_mode(value: Any) -> str:
    if value is None:
        return "fresh"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fresh", "load_weights", "resume"}:
        return normalized
    if normalized in {"loadweights", "weights", "load"}:
        return "load_weights"
    return normalized


_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _is_path_safe_name(value: str) -> bool:
    if not value or value != _slugify(value) or len(value) > 64 or not value[0].isalnum():
        return False
    if value.lower() in _WINDOWS_RESERVED_NAMES:
        return False
    return True


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_relative_path(value: Any, *, base_dir: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


class TrainingManager:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        runs_root: str | Path,
        configs_root: str | Path,
        orchestration_state: OrchestrationState,
        python_executable: str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.configs_root = Path(configs_root).resolve()
        self.dashboard_configs_root = self.configs_root / "dashboard"
        self.generated_configs_root = self.dashboard_configs_root / ".generated"
        self.orchestration_state = orchestration_state
        self.python_executable = python_executable or sys.executable
        self._lock = threading.Lock()
        self._templates: dict[str, TemplateRecord] = {}
        self._drafts: dict[str, DraftConfigRecord] = {}
        self._active_job: ActiveJobRecord | None = None
        self._recent_logs: deque[str] = deque(maxlen=200)
        self._run_id_ready = threading.Event()
        self.dashboard_configs_root.mkdir(parents=True, exist_ok=True)
        self._load_templates()
        self._load_drafts()
        self.refresh_checkpoint_inventory()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            job = self._active_job
            return {
                "templates": [self._template_payload(template) for template in self._templates.values()],
                "drafts": [self._draft_payload(draft) for draft in self._drafts.values()],
                "checkpoints": list(self._checkpoint_inventory),
                "active_job": self._job_summary(job),
                "recent_logs": list(self._recent_logs)[-50:],
                "run_id_ready": self._run_id_ready.is_set(),
            }

    def list_templates(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._template_payload(template) for template in self._templates.values()]

    def list_drafts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._draft_payload(draft) for draft in self._drafts.values()]

    def list_checkpoints(
        self,
        *,
        compatible_for: str | None = None,
        task_type: str | None = None,
        family_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.refresh_checkpoint_inventory()
        compatibility_filter = self._compatibility_signature_for_family(compatible_for) if compatible_for else None
        task_filter = _normalize_task_type(task_type) if task_type else None
        family_filter = _slugify(family_name) if family_name else None
        run_filter = _slugify(run_id) if run_id else None
        with self._lock:
            records = list(self._checkpoint_inventory)
        filtered: list[dict[str, Any]] = []
        for record in records:
            if compatibility_filter and not self._checkpoint_is_compatible(record, compatibility_filter):
                continue
            if task_filter and _normalize_task_type(str(record.get("task_type"))) != task_filter:
                continue
            if family_filter and record.get("project_run") != family_filter:
                continue
            if run_filter and record.get("run_id") != run_filter:
                continue
            filtered.append(record)
        return filtered

    def refresh_checkpoint_inventory(self) -> list[dict[str, Any]]:
        inventory = self._discover_checkpoints()
        with self._lock:
            self._checkpoint_inventory = inventory
        return inventory

    def compatible_checkpoints_for(self, family_name: str) -> list[dict[str, Any]]:
        return self.list_checkpoints(compatible_for=family_name)

    def create_draft(
        self,
        *,
        template_id: str,
        family_name: str | None = None,
        task_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        template = self._require_template(template_id)
        family_name = family_name or template.data.get("logging", {}).get("dir", "").split("/")[-1] or template.template_id
        slug = self._validate_family_name(family_name)
        self._assert_no_draft_conflict(slug)
        draft_path = self.dashboard_configs_root / f"{slug}.yaml"
        draft_data = _strip_runtime_fields(template.data)
        if payload:
            draft_data = _deep_merge(draft_data, _strip_runtime_fields(payload))
        if task_type:
            draft_data.setdefault("dashboard", {})["task_type"] = _normalize_task_type(task_type)
        _dump_yaml(draft_path, draft_data)

        record = DraftConfigRecord(
            family_name=slug,
            template_id=template.template_id,
            task_type=_normalize_task_type(task_type or template.task_type),
            path=draft_path,
            data=draft_data,
        )
        with self._lock:
            self._drafts[record.family_name] = record
        return self._draft_summary(record)

    def save_draft(self, *, family_name: str, payload: dict[str, Any], template_id: str | None = None) -> dict[str, Any]:
        slug = self._validate_family_name(family_name)
        draft_path = self.dashboard_configs_root / f"{slug}.yaml"
        draft_data = _strip_runtime_fields(payload)
        _dump_yaml(draft_path, draft_data)
        task_type = self._infer_task_type_for_draft(draft_data, draft_path)
        record = DraftConfigRecord(
            family_name=slug,
            template_id=template_id or self._find_template_for_task(task_type),
            task_type=task_type,
            path=draft_path,
            data=draft_data,
        )
        with self._lock:
            self._drafts[slug] = record
        return self._draft_summary(record)

    def load_draft(self, family_name: str) -> dict[str, Any] | None:
        slug = self._validate_family_name(family_name)
        draft_path = self.dashboard_configs_root / f"{slug}.yaml"
        if not draft_path.exists():
            return None
        draft_data = _load_yaml(draft_path)
        record = DraftConfigRecord(
            family_name=slug,
            template_id=self._find_template_for_task(
                self._infer_task_type_for_draft(draft_data, draft_path)
            ),
            task_type=self._infer_task_type_for_draft(draft_data, draft_path),
            path=draft_path,
            data=draft_data,
        )
        with self._lock:
            self._drafts[slug] = record
        return self._draft_summary(record)

    def delete_draft(self, family_name: str) -> dict[str, Any]:
        slug = self._validate_family_name(family_name)
        draft_path = self.dashboard_configs_root / f"{slug}.yaml"
        try:
            draft_path.unlink()
        except FileNotFoundError:
            raise FileNotFoundError(f"Draft '{slug}' does not exist.")
        with self._lock:
            self._drafts.pop(slug, None)
        return {"family_name": slug}

    def render_launch_config(self, *, family_name: str, checkpoint_path: str | None = None) -> Path:
        draft = self._require_draft(family_name)
        template = self._require_template(draft.template_id)
        launch_config = _deep_merge(template.data, draft.data)
        launch_config = _strip_runtime_fields(launch_config)
        template_dir = template.path.parent
        training_root = self.repo_root / "training"

        resolved_profile = _resolve_relative_path(launch_config.get("watermark_profile"), base_dir=template_dir)
        if resolved_profile:
            launch_config["watermark_profile"] = resolved_profile

        dataset_cfg = launch_config.setdefault("dataset", {})
        if isinstance(dataset_cfg, dict):
            resolved_dataset_root = _resolve_relative_path(dataset_cfg.get("root"), base_dir=training_root)
            if resolved_dataset_root:
                dataset_cfg["root"] = resolved_dataset_root
            resolved_store_root = _resolve_relative_path(dataset_cfg.get("preprocessed_store_dir"), base_dir=training_root)
            if resolved_store_root:
                dataset_cfg["preprocessed_store_dir"] = resolved_store_root

        launch_config.setdefault("logging", {})
        launch_config["logging"]["dir"] = str((self.runs_root / draft.family_name).resolve())
        launch_config["logging"].setdefault("keep_latest_runs", 15)
        launch_config["logging"]["project_run"] = draft.family_name
        seed_value = launch_config.get("seed", 42)
        launch_config["seed"] = int(seed_value) if seed_value is not None else 42
        launch_config["dashboard"] = {
            "enabled": False,
            "host": "0.0.0.0",
            "port": 8765,
            "open_browser": False,
            "task_type": draft.task_type,
        }
        if checkpoint_path:
            launch_config.setdefault("checkpointing", {})
            launch_config["checkpointing"]["resume_from"] = checkpoint_path

        self.generated_configs_root.mkdir(parents=True, exist_ok=True)
        launch_path = self.generated_configs_root / f"{draft.family_name}__{self._timestamp()}.yaml"
        _dump_yaml(launch_path, launch_config)
        return launch_path

    def start_run(self, request: RunStartRequest) -> dict[str, Any]:
        request.validate()
        draft = self._require_draft(request.family_name)
        self._assert_no_active_family_conflict(draft.family_name)
        checkpoint_path = request.normalized_checkpoint_path()
        if getattr(request, "config_path", None):
            launch_path = Path(request.config_path)
        else:
            launch_path = self.render_launch_config(family_name=request.family_name, checkpoint_path=checkpoint_path)
        _KNOWN_TASK_TYPES = {"segmentation", "removal", "restoration"}
        if draft.task_type not in _KNOWN_TASK_TYPES:
            raise ValueError(f"Unknown task type '{draft.task_type}'. Expected one of: {sorted(_KNOWN_TASK_TYPES)}")
        train_script = self.repo_root / "training" / (
            "train_seg.py" if draft.task_type == "segmentation"
            else "train_restoration.py" if draft.task_type == "restoration"
            else "train.py"
        )
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        with self._lock:
            if self._active_job is not None and self._active_job.process.poll() is None:
                raise RuntimeError("A training job is already running")
            latest = load_latest_manifest(self.runs_root / draft.family_name)
            previous_run_id = latest.identity.run_id if latest is not None else None

        command = [
            self.python_executable,
            str(train_script),
            "--config",
            str(launch_path),
        ]
        if request.mode == "resume" and checkpoint_path:
            command.extend(["--resume", checkpoint_path, "--force-continue"])
        elif checkpoint_path:
            command.extend(["--load-weights", checkpoint_path])

        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        job = ActiveJobRecord(
            command=command,
            process=process,
            family_name=draft.family_name,
            template_id=draft.template_id,
            task_type=draft.task_type,
            draft_path=draft.path,
            launch_config_path=launch_path,
            mode=request.mode,
            checkpoint_path=checkpoint_path,
            previous_run_id=previous_run_id,
            started_at=time.time(),
        )

        with self._lock:
            self._active_job = job
            self._run_id_ready.clear()
        self.orchestration_state.set_starting(
            selected_family=draft.family_name,
            draft_config_path=str(draft.path),
            active_checkpoint=checkpoint_path,
            metadata={
                "template_id": draft.template_id,
                "task_type": draft.task_type,
                "mode": request.mode,
            },
        )
        threading.Thread(target=self._track_job, args=(job,), daemon=True).start()
        threading.Thread(target=self._capture_output, args=(job,), daemon=True).start()
        return self._job_summary(job) or {}

    def pause_run(self) -> dict[str, Any]:
        with self._lock:
            job = self._active_job
            if job is None or job.process.poll() is not None:
                raise RuntimeError("No active training job is running.")
            job.pause_requested = True

        run_dir = self._ensure_job_run_dir(job)
        request_pause(run_dir, requested_by="dashboard")
        self.orchestration_state.set_pausing()
        return self._job_summary(job) or {}

    def resume_run(self, *, family_name: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        snapshot = self.orchestration_state.snapshot()
        explicit_family = _slugify(family_name) if family_name else None
        explicit_run = _slugify(run_id) if run_id else None

        if explicit_family or explicit_run:
            resolved_family = explicit_family or self._family_name_for_run(explicit_run)
            if not resolved_family:
                raise RuntimeError("Paused run family is unavailable.")

            latest = load_latest_manifest(self.runs_root / resolved_family)
            if latest is None:
                raise RuntimeError(f"No completed run metadata found for '{resolved_family}'.")

            # Try to restore missing draft from the run directory itself if it's missing from the dashboard
            if resolved_family not in self._drafts:
                self._materialize_draft_from_run(resolved_family, explicit_run or latest.identity.run_id)

            if explicit_run and latest.identity.run_id != explicit_run:
                manifest = load_run_manifest(self.runs_root / resolved_family / explicit_run / "meta" / "run.json")
                if manifest is None:
                    raise RuntimeError(f"Run '{explicit_run}' was not found for '{resolved_family}'.")
            checkpoint_record = self._latest_checkpoint_record_for_run(explicit_run or latest.identity.run_id)
            if checkpoint_record is None:
                raise RuntimeError(f"No checkpoint was found for '{resolved_family}'.")
            checkpoint_path = Path(checkpoint_record["checkpoint_path"])
            if not checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint file no longer exists: {checkpoint_path}")

            request = RunStartRequest(
                family_name=resolved_family,
                mode="resume",
                checkpoint_path=str(checkpoint_path),
                checkpoint_run_id=str(checkpoint_record["run_id"]),
                checkpoint_name=str(checkpoint_record["checkpoint_name"]),
            )
            return self.start_run(request)

        phase = snapshot.get("phase")
        if phase not in {"paused", "failed"}:
            raise RuntimeError("There is no paused or failed job to resume.")

        selected_family = snapshot.get("selected_family")
        if not isinstance(selected_family, str) or not selected_family:
            raise RuntimeError("Resumable run family is unavailable.")

        latest = load_latest_manifest(self.runs_root / selected_family)
        if latest is None:
            raise RuntimeError(f"No completed run metadata found for '{selected_family}'.")

        target_run_id = snapshot.get("active_run_id")
        if not isinstance(target_run_id, str) or not target_run_id:
            target_run_id = latest.identity.run_id

        # Restore the draft from the run's saved config if it was deleted after
        # the run started (the dashboard deletes drafts once training begins so
        # they don't clutter the draft list, but resume still needs one).
        if selected_family not in self._drafts:
            self._materialize_draft_from_run(selected_family, target_run_id)

        checkpoint_record = self._latest_checkpoint_record_for_run(target_run_id)
        if checkpoint_record is None:
            raise RuntimeError(f"No checkpoint was found for '{selected_family}'.")
        checkpoint_path = Path(checkpoint_record["checkpoint_path"])
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint file no longer exists: {checkpoint_path}")

        request = RunStartRequest(
            family_name=selected_family,
            mode="resume",
            checkpoint_path=str(checkpoint_path),
            checkpoint_run_id=str(checkpoint_record["run_id"]),
            checkpoint_name=str(checkpoint_record["checkpoint_name"]),
        )
        return self.start_run(request)

    def _family_name_for_run(self, run_id: str | None) -> str | None:
        if not run_id:
            return None
        normalized = _slugify(run_id)
        for family_dir in self.runs_root.iterdir():
            if not family_dir.is_dir():
                continue
            if (family_dir / normalized).exists():
                return family_dir.name
        return None

    def _materialize_draft_from_run(self, family_name: str, run_id: str) -> bool:
        """Try to recreate a draft config from the run's saved config.yaml."""
        run_config_path = self.runs_root / family_name / run_id / "meta" / "config.yaml"
        if not run_config_path.exists():
            return False

        draft_path = self.dashboard_configs_root / f"{family_name}.yaml"
        try:
            import shutil
            shutil.copy2(run_config_path, draft_path)
            self.load_draft(family_name)
            return True
        except Exception:
            return False

    def _capture_output(self, job: ActiveJobRecord) -> None:
        if job.process.stdout is None:
            return
        buffer: list[str] = []
        log_file = None
        try:
            for line in job.process.stdout:
                text = line.rstrip()
                if not text:
                    continue
                with self._lock:
                    self._recent_logs.append(text)
                # Try to open the run log file once the run directory is known
                if log_file is None:
                    run_dir = job.run_dir  # set by _track_job once run ID is resolved
                    if run_dir is not None:
                        log_path = run_dir / "training.log"
                        try:
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            log_file = open(log_path, "w", encoding="utf-8", buffering=1)
                            for buffered_line in buffer:
                                log_file.write(buffered_line + "\n")
                            buffer.clear()
                        except OSError:
                            pass
                    else:
                        buffer.append(text)
                if log_file is not None:
                    log_file.write(text + "\n")
        except (OSError, ValueError):
            pass
        finally:
            if log_file is not None:
                log_file.close()
            elif buffer and job.run_dir is not None:
                # Process finished before run_dir was known — flush buffer now
                try:
                    log_path = job.run_dir / "training.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "w", encoding="utf-8") as f:
                        for buffered_line in buffer:
                            f.write(buffered_line + "\n")
                except Exception:
                    pass

    def _track_job(self, job: ActiveJobRecord) -> None:
        run_id = self._wait_for_run_id(job.family_name)
        if run_id:
            with self._lock:
                if self._active_job is job:
                    job.run_id = run_id
                    job.run_dir = self.runs_root / job.family_name / run_id
                    self._run_id_ready.set()
            self.orchestration_state.set_running(active_run_id=run_id, active_checkpoint=job.checkpoint_path, metadata={
                "family_name": job.family_name,
                "template_id": job.template_id,
                "task_type": job.task_type,
                "mode": job.mode,
            })

        while job.process.poll() is None:
            time.sleep(1)
        return_code = job.process.returncode
        job.return_code = return_code
        job.finished_at = time.time()
        if return_code == 0:
            if job.pause_requested:
                self.orchestration_state.set_paused(active_run_id=job.run_id)
            else:
                self.orchestration_state.set_idle()
        else:
            if job.pause_requested:
                self.orchestration_state.set_failed(f"pause request ended with code {return_code}")
            else:
                self.orchestration_state.set_failed(f"training exited with code {return_code}")
        with self._lock:
            if self._active_job is job:
                self._active_job = None

        if job.run_dir is not None:
            clear_pause_request(job.run_dir)

    def _wait_for_run_id(self, family_name: str, timeout_s: float = 30.0) -> str | None:
        deadline = time.time() + timeout_s
        family_root = self.runs_root / family_name
        previous_run_id = None
        with self._lock:
            if self._active_job is not None and self._active_job.family_name == family_name:
                previous_run_id = self._active_job.previous_run_id
        while time.time() < deadline:
            latest = load_latest_manifest(family_root)
            if latest is not None:
                # If the run ID has changed, it's a fork — that's our new run.
                if latest.identity.run_id != previous_run_id:
                    return latest.identity.run_id
                # If the run ID is the same but the status is now 'running', it's a
                # successful 'continue' resume.
                if latest.identity.status == "running":
                    return latest.identity.run_id
            time.sleep(0.25)
        return None

    def _ensure_job_run_dir(self, job: ActiveJobRecord) -> Path:
        if job.run_dir is not None:
            return job.run_dir
        run_id = job.run_id or self._wait_for_run_id(job.family_name)
        if not run_id:
            raise RuntimeError("The active run has not registered yet.")
        run_dir = self.runs_root / job.family_name / run_id
        with self._lock:
            if self._active_job is job:
                job.run_id = run_id
                job.run_dir = run_dir
                self._run_id_ready.set()
        return run_dir

    def _latest_checkpoint_record_for_run(self, run_id: str) -> dict[str, Any] | None:
        records = self.list_checkpoints(run_id=run_id)
        if not records:
            return None

        def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
            epoch = record.get("epoch")
            epoch_value = int(epoch) if isinstance(epoch, int) else -1
            best_bias = 1 if record.get("kind") == "best" else 0
            return (epoch_value, best_bias, str(record.get("checkpoint_name") or ""))

        return max(records, key=sort_key)

    def _load_templates(self) -> None:
        templates: dict[str, TemplateRecord] = {}
        if self.configs_root.exists():
            for path in sorted(self.configs_root.glob("*.yaml")):
                if path.parent.name == "dashboard":
                    continue
                cfg = _load_yaml(path)
                template_id = path.stem
                templates[template_id] = TemplateRecord(
                    template_id=template_id,
                    label=path.stem.replace("_", " ").title(),
                    task_type=_infer_task_type(cfg, path),
                    path=path,
                    data=cfg,
                )
        with self._lock:
            self._templates = templates

    def _load_drafts(self) -> None:
        drafts: dict[str, DraftConfigRecord] = {}
        if self.dashboard_configs_root.exists():
            for path in sorted(self.dashboard_configs_root.glob("*.yaml")):
                cfg = _load_yaml(path)
                family_name = path.stem
                task_type = self._infer_task_type_for_draft(cfg, path)
                template_id = self._find_template_for_task(task_type)
                drafts[family_name] = DraftConfigRecord(
                    family_name=family_name,
                    template_id=template_id,
                    task_type=task_type,
                    path=path,
                    data=cfg,
                )
        with self._lock:
            self._drafts = drafts
        self.refresh_checkpoint_inventory()

    def _infer_task_type_for_draft(self, cfg: dict[str, Any], path: Path) -> str:
        dashboard_section = cfg.get("dashboard", {})
        if isinstance(dashboard_section, dict):
            task_type = dashboard_section.get("task_type")
            if isinstance(task_type, str) and task_type:
                return _normalize_task_type(task_type)
        return _normalize_task_type(_infer_task_type(cfg, path))

    def _find_template_for_task(self, task_type: str) -> str:
        normalized_task_type = _normalize_task_type(task_type)
        with self._lock:
            for template in self._templates.values():
                if template.task_type == normalized_task_type:
                    return template.template_id
            return next(iter(self._templates.keys()), "train_256")

    def _require_template(self, template_id: str) -> TemplateRecord:
        with self._lock:
            template = self._templates.get(template_id)
        if template is None:
            raise KeyError(f"Unknown template: {template_id}")
        return template

    def _require_draft(self, family_name: str) -> DraftConfigRecord:
        slug = self._validate_family_name(family_name)
        with self._lock:
            draft = self._drafts.get(slug)
        if draft is None:
            self.load_draft(slug)
            with self._lock:
                draft = self._drafts.get(slug)
        if draft is None:
            raise KeyError(f"Unknown draft config: {family_name}")
        return draft

    @staticmethod
    def _template_summary(template: TemplateRecord) -> dict[str, Any]:
        return {
            "template_id": template.template_id,
            "label": template.label,
            "task_type": template.task_type,
            "path": str(template.path),
        }

    @staticmethod
    def _template_payload(template: TemplateRecord) -> dict[str, Any]:
        payload = TrainingManager._template_summary(template)
        payload["data"] = _deep_copy(template.data)
        return payload

    @staticmethod
    def _draft_summary(draft: DraftConfigRecord) -> dict[str, Any]:
        return {
            "family_name": draft.family_name,
            "template_id": draft.template_id,
            "task_type": draft.task_type,
            "path": str(draft.path),
        }

    @staticmethod
    def _draft_payload(draft: DraftConfigRecord) -> dict[str, Any]:
        payload = TrainingManager._draft_summary(draft)
        payload["data"] = _deep_copy(draft.data)
        return payload

    @staticmethod
    def _job_summary(job: ActiveJobRecord | None) -> dict[str, Any] | None:
        if job is None:
            return None
        return {
            "family_name": job.family_name,
            "template_id": job.template_id,
            "task_type": job.task_type,
            "draft_path": str(job.draft_path),
            "launch_config_path": str(job.launch_config_path),
            "mode": job.mode,
            "checkpoint_path": job.checkpoint_path,
            "run_id": job.run_id,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "return_code": job.return_code,
        }

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def _validate_family_name(self, family_name: str) -> str:
        slug = _slugify(family_name)
        if not _is_path_safe_name(slug):
            raise ValueError(
                "Run family name must contain only letters, digits, underscores, or dashes, "
                "and must start with a letter or digit."
            )
        if slug in {"dashboard", ".generated"}:
            raise ValueError(f"Run family name '{family_name}' is reserved.")
        return slug

    def _assert_no_draft_conflict(self, slug: str) -> None:
        with self._lock:
            if slug in self._drafts:
                raise FileExistsError(f"Draft '{slug}' already exists. Choose a different run family name.")

    def _assert_no_active_family_conflict(self, slug: str) -> None:
        with self._lock:
            if self._active_job is not None and self._active_job.family_name == slug and self._active_job.process.poll() is None:
                raise RuntimeError(f"Run family '{slug}' already has an active training job.")

    def _load_any_manifest(self, run_json: Path):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            return None

        if isinstance(payload, dict) and "identity" in payload and "paths" in payload:
            try:
                return load_run_manifest(run_json)
            except Exception:
                return None

        if isinstance(payload, dict):
            try:
                return _legacy_manifest_from_payload(run_json.parent.parent, payload)
            except Exception:
                return None
        return None

    def _discover_checkpoints(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        if not self.runs_root.exists():
            return discovered

        run_json_paths = sorted(self.runs_root.rglob("meta/run.json"), key=lambda path: path.as_posix(), reverse=True)
        for run_json in run_json_paths:
            manifest = self._load_any_manifest(run_json)
            if manifest is None:
                continue

            run_dir = run_json.parent.parent
            checkpoint_dir = run_dir / "artifacts" / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            for checkpoint_path in sorted(checkpoint_dir.glob("*.pth"), reverse=True):
                record = self._checkpoint_record_from_path(manifest, checkpoint_path, run_dir)
                if record is not None:
                    discovered.append(record)

        return discovered

    def _checkpoint_record_from_path(self, manifest, checkpoint_path: Path, run_dir: Path) -> dict[str, Any] | None:
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        sidecar_data: dict[str, Any] = {}
        if sidecar.exists():
            try:
                loaded = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    sidecar_data = loaded
            except Exception:
                sidecar_data = {}

        epoch = None
        stem = checkpoint_path.stem
        if stem.startswith("epoch_"):
            try:
                epoch = int(stem.split("_", 1)[1])
            except Exception:
                epoch = None

        run_config = _load_yaml_config(run_dir / "meta" / "config.yaml")
        checkpoint_rel = checkpoint_path.relative_to(run_dir).as_posix()
        return {
            "run_id": manifest.identity.run_id,
            "project_run": manifest.identity.project_run,
            "task_name": manifest.identity.task_name,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_url": f"/runs/{manifest.identity.run_id}/{checkpoint_rel}",
            "artifact_dir": str(run_dir / "artifacts"),
            "checkpoint_dir": str(checkpoint_path.parent),
            "kind": "best" if checkpoint_path.name == "best.pth" else "epoch",
            "epoch": epoch,
            "status": manifest.identity.status,
            "staged_at": sidecar_data.get("staged_at"),
            "source_path": sidecar_data.get("source_path"),
            "family_root": str(run_dir.parent),
            "task_type": _normalize_task_type(manifest.identity.task_name),
            "signature": self._config_signature(run_config),
        }

    def _compatibility_signature_for_family(self, family_name: str) -> dict[str, Any] | None:
        draft = self._drafts.get(_slugify(family_name))
        if draft is None:
            try:
                draft = self._require_draft(family_name)
            except Exception:
                return None
        template = self._templates.get(draft.template_id)
        base_cfg: dict[str, Any] = {}
        if template is not None:
            base_cfg = _deep_merge(template.data, draft.data)
        else:
            base_cfg = _deep_copy(draft.data)
        return self._config_signature(base_cfg)

    def _config_signature(self, cfg: dict[str, Any]) -> dict[str, Any]:
        dashboard = cfg.get("dashboard")
        explicit = dashboard.get("task_type") if isinstance(dashboard, dict) else None
        if not explicit:
            # Rendered configs write task_type at the top level; also try structural inference.
            explicit = cfg.get("task_type")
        task_type = _normalize_task_type(_infer_task_type(cfg, Path("."))) if not explicit else _normalize_task_type(str(explicit))
        dataset = cfg.get("dataset", {})
        model = cfg.get("model", {})
        signature: dict[str, Any] = {"task_type": task_type}
        if task_type in {"removal", "restoration"}:
            if isinstance(dataset, dict):
                signature["resolution"] = {
                    "image_width": dataset.get("image_width"),
                    "image_height": dataset.get("image_height"),
                }
            if isinstance(model, dict):
                signature["model"] = {
                    "type": model.get("type"),
                    "base_channels": model.get("base_channels"),
                    "depth": model.get("depth"),
                    "use_checkpoint": model.get("use_checkpoint"),
                }
        else:
            if isinstance(dataset, dict):
                signature["resolution"] = {"image_size": dataset.get("image_size")}
            if isinstance(model, dict):
                signature["model"] = {
                    "encoder": model.get("encoder"),
                }
        return signature

    def _checkpoint_is_compatible(self, record: dict[str, Any], target_signature: dict[str, Any]) -> bool:
        if _normalize_task_type(str(record.get("task_type"))) != _normalize_task_type(str(target_signature.get("task_type"))):
            return False
        source_signature = record.get("signature")
        if not isinstance(source_signature, dict):
            return False
        for section in ("resolution", "model"):
            target_section = target_signature.get(section)
            if not isinstance(target_section, dict):
                continue
            source_section = source_signature.get(section)
            if not isinstance(source_section, dict):
                return False
            for key, value in target_section.items():
                if value is None:
                    continue
                if source_section.get(key) != value:
                    return False
        return True
