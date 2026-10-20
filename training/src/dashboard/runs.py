"""
Read-only index of training runs on disk.

The viewer never talks to a training process. Everything it shows comes from
files a run writes into its folder (see wm_shared/experiment.py):

  <family>/<run_id>/meta/run.json            identity and lineage
  <family>/<run_id>/meta/status.json         status + heartbeat (epoch, step, updated_at)
  <family>/<run_id>/meta/config.yaml         resolved config
  <family>/<run_id>/meta/model_overview.json model summary
  <family>/<run_id>/train.csv, val.csv       metric history
  <family>/<run_id>/previews/<epoch>/*.png   preview images
  <family>/<run_id>/artifacts/...            checkpoints
  <family>/<run_id>/training.log             console output

A background thread polls file modification times and tells subscribers which
runs changed. The only things the viewer writes are viewer annotations in
meta/star.json and meta/deleted.json (hide from the list).
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wm_shared.run_manifest import RunIdentity, RunLineage, RunManifest, RunPaths, build_run_paths, load_run_manifest


MAX_POINTS = 500
MAX_PREVIEW_STEPS = 50
LOG_TAIL_BYTES = 256 * 1024
DEFAULT_STALE_AFTER_S = 15 * 60
DEFAULT_POLL_INTERVAL_S = 2.0

_WATCHED_FILES = (
    "meta/run.json",
    "meta/status.json",
    "meta/star.json",
    "meta/config.yaml",
    "meta/model_overview.json",
    "train.csv",
    "val.csv",
    "training.log",
)
_WATCHED_DIRS = ("previews", "artifacts", "artifacts/checkpoints")


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
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append({key: _coerce(value) for key, value in row.items() if key is not None})
    except (OSError, csv.Error, UnicodeDecodeError):
        # The trainer may be mid-append; the next poll will pick up a clean read.
        pass
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




def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


def resolve_status(run_dir: Path, manifest_status: str, *, now: float, stale_after: float) -> dict[str, Any]:
    """Combine run.json, meta/status.json and file activity into a display status.

    A run that claims to be running but has shown no activity for `stale_after`
    seconds is reported as "stopped" (the process was killed or crashed hard).
    """
    status_file = _read_json(run_dir / "meta" / "status.json") or {}
    status = str(status_file.get("status") or manifest_status or "unknown")
    activity = [
        _parse_iso(status_file.get("updated_at")),
        _mtime(run_dir / "training.log"),
        _mtime(run_dir / "train.csv"),
        _mtime(run_dir / "val.csv"),
        _mtime(run_dir / "meta" / "run.json"),
    ]
    last_seen = max((ts for ts in activity if ts is not None), default=None)
    if status == "running" and (last_seen is None or now - last_seen > stale_after):
        status = "stopped"
    progress = {key: status_file.get(key) for key in ("epoch", "step", "global_step") if status_file.get(key) is not None}
    return {"status": status, "last_seen": _iso(last_seen), "progress": progress}


def _run_signature(run_dir: Path) -> tuple:
    parts: list[Any] = []
    for rel in _WATCHED_FILES:
        try:
            st = (run_dir / rel).stat()
            parts.append((st.st_mtime_ns, st.st_size))
        except OSError:
            parts.append(None)
    for rel in _WATCHED_DIRS:
        parts.append(_mtime(run_dir / rel))
    preview_dir = run_dir / "previews"
    try:
        latest = max((p for p in preview_dir.iterdir() if p.is_dir()), default=None, key=lambda p: p.name)
    except OSError:
        latest = None
    parts.append((latest.name, _mtime(latest)) if latest is not None else None)
    parts.append((run_dir / "meta" / "deleted.json").exists())
    return tuple(parts)


def _discover_run_dirs(roots: list[Path]) -> list[Path]:
    """Run folders are <root>/<family>/<run_id>/meta/run.json, or <root>/<run_id>/... if root is a family."""
    found: dict[Path, None] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*/meta/run.json", "*/*/meta/run.json"):
            try:
                for run_json in root.glob(pattern):
                    found[run_json.parent.parent] = None
            except OSError:
                continue
    return list(found)


class RunIndex:
    def __init__(
        self,
        roots: list[Path],
        *,
        stale_after: float = DEFAULT_STALE_AFTER_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    ):
        self.roots = [Path(root).resolve() for root in roots]
        self.stale_after = stale_after
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._run_dirs: dict[str, Path] = {}
        self._signatures: dict[Path, tuple] = {}
        self._dir_to_run_id: dict[Path, str] = {}
        self._subscribers: set[queue.Queue] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_record(self, run_dir: Path) -> tuple[str, dict[str, Any]] | None:
        meta_dir = run_dir / "meta"
        if (meta_dir / "deleted.json").exists():
            return None
        try:
            manifest = _load_any_manifest(meta_dir / "run.json")
        except Exception:
            return None
        run_id = manifest.identity.run_id
        star = _read_json(meta_dir / "star.json") or {}
        summary = _manifest_summary(
            manifest,
            starred=bool(star.get("starred", False)),
            note=str(star.get("note", "")),
        )
        live = resolve_status(run_dir, manifest.identity.status, now=time.time(), stale_after=self.stale_after)
        summary["status"] = live["status"]
        summary["last_seen"] = live["last_seen"]
        summary["progress"] = live["progress"]
        summary["family_dir"] = str(run_dir.parent)

        config_data = None
        config_file = meta_dir / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            except Exception:
                config_data = None
        if isinstance(config_data, dict):
            epochs = (config_data.get("training") or {}).get("epochs")
            if isinstance(epochs, int):
                summary["progress"]["epochs_total"] = epochs

        record = {
            "manifest": summary,
            "config": config_data,
            "model_overview": _read_json(meta_dir / "model_overview.json"),
            "train_metrics": _read_csv_rows(manifest.paths.train_csv),
            "val_metrics": _read_csv_rows(manifest.paths.val_csv),
            "previews": _latest_preview(manifest.paths.preview_dir, run_id),
            "preview_history": _preview_history(manifest.paths.preview_dir, run_id),
            "artifacts": _artifact_entries(run_id, manifest.paths.artifact_dir),
        }
        return run_id, record

    def scan(self) -> tuple[set[str], bool]:
        """Re-read runs whose files changed. Returns (changed run ids, whether the run list changed)."""
        run_dirs = _discover_run_dirs(self.roots)
        seen: set[Path] = set()
        changed: set[str] = set()
        list_changed = False
        now = time.time()

        for run_dir in run_dirs:
            seen.add(run_dir)
            signature = _run_signature(run_dir)
            previous_id = self._dir_to_run_id.get(run_dir)
            if self._signatures.get(run_dir) == signature and previous_id is not None:
                # Files are unchanged, but a running run can still go stale with time.
                with self._lock:
                    record = self._records.get(previous_id)
                if record is not None and record["manifest"]["status"] == "running":
                    live = resolve_status(run_dir, "running", now=now, stale_after=self.stale_after)
                    if live["status"] != "running":
                        with self._lock:
                            record["manifest"]["status"] = live["status"]
                        changed.add(previous_id)
                continue

            self._signatures[run_dir] = signature
            loaded = self._load_record(run_dir)
            with self._lock:
                if loaded is None:
                    if previous_id is not None:
                        self._records.pop(previous_id, None)
                        self._run_dirs.pop(previous_id, None)
                        self._dir_to_run_id.pop(run_dir, None)
                        list_changed = True
                    continue
                run_id, record = loaded
                if run_id in self._run_dirs and self._run_dirs[run_id] != run_dir:
                    # Same run id in two roots: keep the first one found.
                    continue
                if previous_id is None:
                    list_changed = True
                self._records[run_id] = record
                self._run_dirs[run_id] = run_dir
                self._dir_to_run_id[run_dir] = run_id
            changed.add(run_id)

        for run_dir in list(self._signatures):
            if run_dir in seen:
                continue
            self._signatures.pop(run_dir, None)
            run_id = self._dir_to_run_id.pop(run_dir, None)
            if run_id is not None:
                with self._lock:
                    self._records.pop(run_id, None)
                    self._run_dirs.pop(run_id, None)
                list_changed = True
        return changed, list_changed

    # ── watching ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.scan()
        self._thread = threading.Thread(target=self._watch, name="run-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                changed, list_changed = self.scan()
            except Exception as exc:  # keep watching whatever happens on disk
                print(f"[viewer] scan failed: {exc}")
                continue
            if changed or list_changed:
                self.publish("runs_changed", {"run_ids": sorted(changed), "list_changed": list_changed})

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "data": data}
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                # A slow client only needs to know that something changed.
                pass

    # ── queries ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Lightweight list for the run picker: no train history, previews or config."""
        with self._lock:
            records = list(self._records.values())
        runs = []
        for record in sorted(records, key=lambda r: (r["manifest"].get("created_at") or "", r["manifest"]["run_id"]), reverse=True):
            runs.append(
                {
                    "manifest": record["manifest"],
                    "train_metrics": record["train_metrics"][-1:],
                    "val_metrics": record["val_metrics"],
                }
            )
        return {"roots": [str(root) for root in self.roots], "runs": runs}

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(run_id)
            return dict(record) if record is not None else None

    def log_tail(self, run_id: str, max_bytes: int = LOG_TAIL_BYTES) -> dict[str, Any] | None:
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
        if run_dir is None:
            return None
        path = run_dir / "training.log"
        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                data = f.read()
        except OSError:
            return {"exists": False, "truncated": False, "text": ""}
        # Logs are written in text mode, so they have CRLF line endings on Windows.
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        truncated = size > max_bytes
        if truncated:
            text = text.split("\n", 1)[-1]
        return {"exists": True, "truncated": truncated, "text": text}

    # ── viewer annotations ───────────────────────────────────────────────────

    def set_star(self, run_id: str, starred: bool, note: str) -> bool:
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
        if run_dir is None:
            return False
        star_file = run_dir / "meta" / "star.json"
        star_file.write_text(
            json.dumps({"starred": starred, "note": note}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._rescan_and_publish()
        return True

    def hide_run(self, run_id: str) -> bool:
        """Hide a run from the list. Files stay on disk; delete meta/deleted.json to bring it back."""
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
        if run_dir is None:
            return False
        (run_dir / "meta" / "deleted.json").write_text(json.dumps({"deleted": True}), encoding="utf-8")
        self._rescan_and_publish()
        return True

    def _rescan_and_publish(self) -> None:
        changed, list_changed = self.scan()
        if changed or list_changed:
            self.publish("runs_changed", {"run_ids": sorted(changed), "list_changed": list_changed})

    def resolve_run_file(self, run_id: str, relative_path: str) -> Path | None:
        with self._lock:
            run_dir = self._run_dirs.get(run_id)
        if run_dir is None:
            return None
        resolved_run_dir = run_dir.resolve()
        if not any(_is_relative_to(resolved_run_dir, root) for root in self.roots):
            return None
        path = (run_dir / relative_path).resolve()
        if not _is_relative_to(path, resolved_run_dir):
            return None
        if not path.is_file():
            return None
        return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
