from __future__ import annotations

import csv
import json
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

from wm_shared.run_manifest import RunIdentity, RunLineage, RunManifest, RunPaths, build_run_paths, load_run_manifest


MAX_POINTS = 500
MAX_EVENTS = 200
MAX_PREVIEW_STEPS = 50


def _coerce(value: str) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        int_val = int(text)
        return int_val
    except ValueError:
        pass
    try:
        float_val = float(text)
        # Map non-finite values to None so charting libraries don't choke
        import math
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return float_val
    except ValueError:
        pass
    return text


def _downsample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Downsample to at most `limit` rows, always preserving the tail for recency detail."""
    if len(rows) <= limit:
        return rows
    tail_n = min(50, limit // 10)
    tail = rows[-tail_n:]
    head = rows[:-tail_n]
    head_limit = limit - tail_n
    if len(head) <= head_limit:
        return head + tail
    step = len(head) / head_limit
    return [head[int(i * step)] for i in range(head_limit)] + tail


def _read_csv_rows(path: Path, limit: int = MAX_POINTS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    all_rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append({key: _coerce(value) for key, value in row.items()})
    return _downsample(all_rows, limit)


def _latest_preview(preview_dir: Path, run_id: str) -> dict[str, str]:
    history = _preview_history(preview_dir, run_id, limit=1)
    if not history:
        return {}
    return history[-1]["previews"]


def _preview_history(preview_dir: Path, run_id: str, *, limit: int = MAX_PREVIEW_STEPS) -> list[dict[str, Any]]:
    if not preview_dir.exists():
        return []

    history: list[dict[str, Any]] = []
    preview_steps = sorted([path for path in preview_dir.iterdir() if path.is_dir()])
    for step_dir in preview_steps[-limit:]:
        previews = {
            image.stem: f"/runs/{run_id}/previews/{step_dir.name}/{image.name}"
            for image in sorted(step_dir.glob("*.png"))
        }
        if not previews:
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            step = step_dir.name
        history.append(
            {
                "step": step,
                "label": step_dir.name,
                "previews": previews,
            }
        )
    return history


def _artifact_entries(run_id: str, artifact_dir: Path) -> list[dict[str, str]]:
    if not artifact_dir.exists():
        return []
    entries: list[dict[str, str]] = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.suffix == ".json":
            continue
        rel_path = path.relative_to(artifact_dir.parent).as_posix()
        entries.append(
            {
                "name": path.name,
                "category": path.parent.relative_to(artifact_dir).as_posix() if path.parent != artifact_dir else "",
                "url": f"/runs/{run_id}/{rel_path}",
            }
        )
    return entries[-20:]


def _prefer_longer_list(primary: list[Any], fallback: list[Any]) -> list[Any]:
    return primary if len(primary) >= len(fallback) else fallback


def _legacy_manifest(run_dir: Path, payload: dict[str, Any]) -> RunManifest:
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


def _load_any_manifest(run_json: Path) -> RunManifest:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "identity" in payload and "paths" in payload:
        manifest = load_run_manifest(run_json)
        # Paths in run.json are absolute and baked in at training time — they break
        # on any machine other than the one that created the run.  Re-derive from
        # the actual location of run.json on disk so cloned repos work correctly.
        run_dir = run_json.parent.parent
        manifest.paths = build_run_paths(run_dir.parent, run_dir.name)
        return manifest
    if isinstance(payload, dict):
        return _legacy_manifest(run_json.parent.parent, payload)
    return _legacy_manifest(run_json.parent.parent, {})


def _manifest_summary(manifest: RunManifest, *, starred: bool = False, note: str = "") -> dict[str, Any]:
    return {
        "project_run": manifest.identity.project_run,
        "run_id": manifest.identity.run_id,
        "timeline_id": manifest.identity.timeline_id,
        "task_name": manifest.identity.task_name,
        "created_at": manifest.identity.created_at,
        "status": manifest.identity.status,
        "device": manifest.identity.device,
        "config_fingerprint": manifest.identity.config_fingerprint,
        "code_fingerprint": manifest.identity.code_fingerprint,
        "lineage": {
            "start_mode": manifest.identity.lineage.start_mode,
            "history_mode": manifest.identity.lineage.history_mode,
            "parent_run_id": manifest.identity.lineage.parent_run_id,
            "parent_timeline_id": manifest.identity.lineage.parent_timeline_id,
            "parent_checkpoint": manifest.identity.lineage.parent_checkpoint,
            "resumed_from_epoch": manifest.identity.lineage.resumed_from_epoch,
        },
        "starred": starred,
        "note": note,
    }


class DashboardState:
    def __init__(self, *, family_root: Path, current_run_id: str | None = None):
        self.family_root = family_root
        self.current_run_id = current_run_id
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._runs: dict[str, dict[str, Any]] = {}
        self._run_dirs: dict[str, Path] = {}

    def _load_record_from_run_json(self, run_json: Path) -> tuple[str, dict[str, Any], Path] | None:
        # Skip runs that have been soft-deleted via delete_run().
        if (run_json.parent / "deleted.json").exists():
            return None
        run_dir = run_json.parent.parent
        try:
            manifest = _load_any_manifest(run_json)
        except Exception:
            return None
        run_id = manifest.identity.run_id
        starred = False
        note = ""
        star_file = run_json.parent / "star.json"
        if star_file.exists():
            try:
                star_data = json.loads(star_file.read_text(encoding="utf-8"))
                starred = bool(star_data.get("starred", False))
                note = str(star_data.get("note", ""))
            except Exception:
                pass
        config_file = manifest.paths.meta_dir / "config.yaml"
        config_data = None
        if config_file.exists():
            try:
                import yaml
                config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        record = {
            "manifest": _manifest_summary(manifest, starred=starred, note=note),
            "config": config_data,
            "model_overview": self._load_model_overview(manifest.paths.meta_dir / "model_overview.json"),
            "train_metrics": _read_csv_rows(manifest.paths.train_csv),
            "val_metrics": _read_csv_rows(manifest.paths.val_csv),
            "previews": _latest_preview(manifest.paths.preview_dir, manifest.identity.run_id),
            "preview_history": _preview_history(manifest.paths.preview_dir, manifest.identity.run_id),
            "artifacts": _artifact_entries(manifest.identity.run_id, manifest.paths.artifact_dir),
        }
        return run_id, record, run_dir

    def _load_record_for_run_id(self, run_id: str) -> dict[str, Any] | None:
        if not self.family_root.exists():
            return None
        run_json_paths = sorted(self.family_root.rglob("meta/run.json"), key=lambda path: path.as_posix(), reverse=True)
        for run_json in run_json_paths:
            loaded = self._load_record_from_run_json(run_json)
            if loaded is None:
                continue
            loaded_run_id, record, run_dir = loaded
            if loaded_run_id != run_id:
                continue
            with self._lock:
                self._runs[loaded_run_id] = record
                self._run_dirs[loaded_run_id] = run_dir
            return record
        return None

    def load_from_disk(self) -> None:
        runs: dict[str, dict[str, Any]] = {}
        run_dirs: dict[str, Path] = {}
        if self.family_root.exists():
            try:
                run_json_paths = sorted(self.family_root.rglob("meta/run.json"), key=lambda path: path.as_posix(), reverse=True)
            except OSError:
                run_json_paths = []
            for run_json in run_json_paths:
                loaded = self._load_record_from_run_json(run_json)
                if loaded is None:
                    continue
                run_id, record, run_dir = loaded
                runs[run_id] = record
                run_dirs[run_id] = run_dir
        with self._lock:
            merged_runs = dict(self._runs)
            for run_id, disk_record in runs.items():
                existing = merged_runs.get(run_id)
                if existing is None:
                    merged_runs[run_id] = disk_record
                    continue
                merged_runs[run_id] = {
                    "manifest": dict(disk_record.get("manifest") or existing.get("manifest") or {}),
                    "model_overview": disk_record.get("model_overview") or existing.get("model_overview"),
                    "train_metrics": _prefer_longer_list(
                        list(existing.get("train_metrics") or []),
                        list(disk_record.get("train_metrics") or []),
                    ),
                    "val_metrics": _prefer_longer_list(
                        list(existing.get("val_metrics") or []),
                        list(disk_record.get("val_metrics") or []),
                    ),
                    "previews": dict(disk_record.get("previews") or existing.get("previews") or {}),
                    "preview_history": _prefer_longer_list(
                        list(existing.get("preview_history") or []),
                        list(disk_record.get("preview_history") or []),
                    ),
                    "artifacts": _prefer_longer_list(
                        list(existing.get("artifacts") or []),
                        list(disk_record.get("artifacts") or []),
                    ),
                }
            self._runs = merged_runs
            self._run_dirs.update(run_dirs)

    @staticmethod
    def _load_model_overview(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            runs = [self._runs[run_id] for run_id in sorted(self._runs.keys(), reverse=True)]
            current_run_id = self.current_run_id
        current = self.run_snapshot(current_run_id) if current_run_id else None
        return {
            "family_root": str(self.family_root),
            "current_run_id": current_run_id,
            "runs": runs,
            "current_run": current,
        }

    def run_snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                return dict(record)
        record = self._load_record_for_run_id(run_id)
        if record is None:
            return None
        return dict(record)

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subscribers.add(subscriber)
        subscriber.put({"type": "snapshot", "data": self.snapshot()})
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "data": data}
        dead: list[queue.Queue] = []
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                dead.append(subscriber)
        if dead:
            with self._lock:
                self._subscribers.difference_update(dead)

    def _ensure_run(self, manifest: RunManifest) -> dict[str, Any]:
        run_id = manifest.identity.run_id
        record = self._runs.get(run_id)
        if record is None:
            record = {
                "manifest": _manifest_summary(manifest),
                "model_overview": None,
                "train_metrics": [],
                "val_metrics": [],
                "previews": {},
                "preview_history": [],
                "artifacts": [],
            }
            self._runs[run_id] = record
        self._run_dirs[run_id] = manifest.paths.run_dir
        return record

    def register_run(self, manifest: RunManifest) -> None:
        with self._lock:
            self._ensure_run(manifest)
        self.publish("run_registered", {"run_id": manifest.identity.run_id, "manifest": _manifest_summary(manifest)})

    def set_status(self, manifest: RunManifest, status: str) -> None:
        with self._lock:
            record = self._ensure_run(manifest)
            record["manifest"]["status"] = status
        self.publish("status", {"run_id": manifest.identity.run_id, "status": status})

    def set_model_overview(self, manifest: RunManifest, overview: dict[str, Any]) -> None:
        with self._lock:
            record = self._ensure_run(manifest)
            record["model_overview"] = overview
        self.publish("model_overview", {"run_id": manifest.identity.run_id, "overview": overview})

    def append_train_metrics(self, manifest: RunManifest, row: dict[str, Any]) -> None:
        with self._lock:
            record = self._ensure_run(manifest)
            record["train_metrics"].append(dict(row))
            if len(record["train_metrics"]) > MAX_POINTS:
                record["train_metrics"] = _downsample(record["train_metrics"], MAX_POINTS)
        self.publish("train_metrics", {"run_id": manifest.identity.run_id, "row": row})

    def append_val_metrics(self, manifest: RunManifest, row: dict[str, Any]) -> None:
        with self._lock:
            record = self._ensure_run(manifest)
            record["val_metrics"].append(dict(row))
            if len(record["val_metrics"]) > MAX_POINTS:
                record["val_metrics"] = _downsample(record["val_metrics"], MAX_POINTS)
        self.publish("val_metrics", {"run_id": manifest.identity.run_id, "row": row})

    def set_preview(self, manifest: RunManifest, step_or_epoch: int, preview_urls: dict[str, str]) -> None:
        preview_entry = {
            "step": int(step_or_epoch),
            "label": f"{int(step_or_epoch):06d}",
            "previews": dict(preview_urls),
        }
        with self._lock:
            record = self._ensure_run(manifest)
            record["previews"] = dict(preview_urls)
            history = record.setdefault("preview_history", [])
            if history and history[-1]["label"] == preview_entry["label"]:
                history[-1] = preview_entry
            else:
                history.append(preview_entry)
            if len(history) > MAX_PREVIEW_STEPS:
                del history[:-MAX_PREVIEW_STEPS]
        self.publish(
            "preview",
            {
                "run_id": manifest.identity.run_id,
                "step": int(step_or_epoch),
                "preview": preview_entry,
                "previews": preview_urls,
            },
        )

    def add_artifact(self, manifest: RunManifest, artifact: dict[str, str]) -> None:
        with self._lock:
            record = self._ensure_run(manifest)
            record["artifacts"].append(artifact)
            record["artifacts"] = record["artifacts"][-20:]
        self.publish("artifact", {"run_id": manifest.identity.run_id, "artifact": artifact})

    def set_star(self, run_id: str, starred: bool, note: str) -> bool:
        """Write meta/star.json for a run. Returns False if the run is unknown."""
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
            record = self._runs.get(run_id)
        if run_dir is None or record is None:
            return False
        star_file = run_dir / "meta" / "star.json"
        star_file.parent.mkdir(parents=True, exist_ok=True)
        star_file.write_text(
            json.dumps({"starred": starred, "note": note}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["manifest"]["starred"] = starred
                self._runs[run_id]["manifest"]["note"] = note
        self.publish("star", {"run_id": run_id, "starred": starred, "note": note})
        return True

    def delete_run(self, run_id: str) -> bool:
        """Remove a run from the in-memory registry and write a deleted marker so it is
        excluded on the next load_from_disk().  Returns False if the run is unknown."""
        with self._lock:
            run_dir = self._run_dirs.pop(run_id, None)
            self._runs.pop(run_id, None)
        if run_dir is None:
            return False
        # Soft-delete marker: keeps run files on disk (checkpoints, etc.) but prevents
        # the run from being rediscovered when the dashboard restarts.
        marker = run_dir / "meta" / "deleted.json"
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"deleted": True}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        self.publish("run_deleted", {"run_id": run_id})
        return True

    def resolve_run_file(self, run_id: str, relative_path: str) -> Path | None:
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
            if run_dir is None:
                return None
        resolved_family = self.family_root.resolve()
        resolved_run_dir = run_dir.resolve()
        # Ensure the run_dir itself hasn't escaped the family root via a symlink
        try:
            resolved_run_dir.relative_to(resolved_family)
        except ValueError:
            return None
        # Ensure the requested file stays within the run directory (no .. traversal,
        # no symlinks pointing outside)
        path = (run_dir / relative_path).resolve()
        try:
            path.relative_to(resolved_run_dir)
        except ValueError:
            return None
        if not path.exists() or not path.is_file():
            return None
        return path
