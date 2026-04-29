from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAUSE_REQUEST_FILENAME = "pause.request.json"


class TrainingPaused(RuntimeError):
    pass


def pause_request_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "meta" / "control" / PAUSE_REQUEST_FILENAME


def pause_requested(run_dir: str | Path) -> bool:
    return pause_request_path(run_dir).exists()


def request_pause(
    run_dir: str | Path,
    *,
    requested_by: str = "dashboard",
    reason: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "requested_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "requested_by": requested_by,
    }
    if reason:
        payload["reason"] = reason

    target = pause_request_path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def clear_pause_request(run_dir: str | Path) -> None:
    target = pause_request_path(run_dir)
    try:
        target.unlink()
    except FileNotFoundError:
        pass
