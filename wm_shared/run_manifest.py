from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


@dataclass
class RunLineage:
    start_mode: str
    history_mode: str
    parent_run_id: str | None = None
    parent_timeline_id: str | None = None
    parent_checkpoint: str | None = None
    resumed_from_epoch: int | None = None


@dataclass
class RunIdentity:
    project_run: str
    run_id: str
    timeline_id: str
    task_name: str
    created_at: str
    status: str
    config_fingerprint: str
    code_fingerprint: str
    device: str | None
    lineage: RunLineage


@dataclass
class RunPaths:
    root_dir: Path
    run_dir: Path
    meta_dir: Path
    preview_dir: Path
    artifact_dir: Path
    tracker_dir: Path
    train_csv: Path
    val_csv: Path


@dataclass
class RunManifest:
    identity: RunIdentity
    paths: RunPaths

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": asdict(self.identity),
            "paths": asdict(self.paths),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.paths.meta_dir / "run.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        identity_data = dict(payload["identity"])
        lineage = RunLineage(**identity_data.pop("lineage"))
        identity = RunIdentity(lineage=lineage, **identity_data)

        path_data = {key: Path(value) for key, value in payload["paths"].items()}
        paths = RunPaths(**path_data)
        return cls(identity=identity, paths=paths)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def set_status(self, status: str) -> None:
        self.identity.status = status
        self.save()


def make_config_fingerprint(cfg: dict[str, Any]) -> str:
    payload = json.dumps(_normalize_for_hash(cfg), sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_code_fingerprint(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        head = result.stdout.strip()
        if head:
            return head
    except Exception:
        pass

    digest = hashlib.sha256()
    for path in sorted((root / "wm_shared").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def make_run_id(project_run: str) -> str:
    return f"{project_run}__{_run_stamp()}"


def build_run_paths(root_dir: str | Path, run_id: str) -> RunPaths:
    root_path = Path(root_dir)
    run_dir = root_path / run_id
    meta_dir = run_dir / "meta"
    preview_dir = run_dir / "previews"
    artifact_dir = run_dir / "artifacts"
    tracker_dir = run_dir / "tracker"
    return RunPaths(
        root_dir=root_path,
        run_dir=run_dir,
        meta_dir=meta_dir,
        preview_dir=preview_dir,
        artifact_dir=artifact_dir,
        tracker_dir=tracker_dir,
        train_csv=run_dir / "train.csv",
        val_csv=run_dir / "val.csv",
    )


def create_run_manifest(
    *,
    project_run: str,
    task_name: str,
    root_dir: str | Path,
    start_mode: str,
    history_mode: str,
    config_fingerprint: str,
    code_fingerprint: str,
    device: str | None,
    parent_run_id: str | None = None,
    parent_timeline_id: str | None = None,
    parent_checkpoint: str | None = None,
    resumed_from_epoch: int | None = None,
    timeline_id: str | None = None,
) -> RunManifest:
    run_id = make_run_id(project_run)
    paths = build_run_paths(root_dir, run_id)
    for path in (paths.root_dir, paths.run_dir, paths.meta_dir, paths.preview_dir, paths.artifact_dir, paths.tracker_dir):
        path.mkdir(parents=True, exist_ok=True)

    identity = RunIdentity(
        project_run=project_run,
        run_id=run_id,
        timeline_id=timeline_id or run_id,
        task_name=task_name,
        created_at=_utc_now_iso(),
        status="running",
        config_fingerprint=config_fingerprint,
        code_fingerprint=code_fingerprint,
        device=device,
        lineage=RunLineage(
            start_mode=start_mode,
            history_mode=history_mode,
            parent_run_id=parent_run_id,
            parent_timeline_id=parent_timeline_id,
            parent_checkpoint=parent_checkpoint,
            resumed_from_epoch=resumed_from_epoch,
        ),
    )
    manifest = RunManifest(identity=identity, paths=paths)
    manifest.save()
    return manifest


def load_run_manifest(path: str | Path) -> RunManifest:
    return RunManifest.load(path)


def load_latest_manifest(root_dir: str | Path) -> RunManifest | None:
    latest_path = Path(root_dir) / "latest.json"
    if not latest_path.exists():
        return None
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_json = Path(payload["run_manifest"])
    if not run_json.exists():
        return None
    return RunManifest.load(run_json)


def save_latest_pointer(manifest: RunManifest) -> Path:
    latest_path = manifest.paths.root_dir / "latest.json"
    payload = {
        "project_run": manifest.identity.project_run,
        "run_id": manifest.identity.run_id,
        "timeline_id": manifest.identity.timeline_id,
        "status": manifest.identity.status,
        "updated_at": _utc_now_iso(),
        "run_dir": str(manifest.paths.run_dir),
        "run_manifest": str(manifest.paths.meta_dir / "run.json"),
    }
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest_path


def prune_old_runs(root_dir: str | Path, keep_latest_runs: int, *, preserve_run_id: str | None = None) -> list[Path]:
    if keep_latest_runs <= 0:
        return []

    root_path = Path(root_dir)
    run_dirs = sorted(
        [path for path in root_path.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    removed: list[Path] = []
    kept = 0
    for run_dir in run_dirs:
        if preserve_run_id and run_dir.name == preserve_run_id:
            kept += 1
            continue
        star_file = run_dir / "meta" / "star.json"
        if star_file.exists():
            try:
                star_data = json.loads(star_file.read_text(encoding="utf-8"))
                if star_data.get("starred"):
                    kept += 1
                    continue
            except Exception:
                pass
        if kept < keep_latest_runs:
            kept += 1
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        removed.append(run_dir)
    return removed
