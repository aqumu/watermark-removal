from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


ORCHESTRATION_PHASES = {
    "idle",
    "configured",
    "starting",
    "running",
    "pausing",
    "paused",
    "failed",
}


@dataclass
class OrchestrationSnapshot:
    phase: str = "idle"
    selected_family: str | None = None
    draft_config_path: str | None = None
    active_run_id: str | None = None
    active_checkpoint: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchestrationState:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = OrchestrationSnapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self._snapshot.phase,
                "selected_family": self._snapshot.selected_family,
                "draft_config_path": self._snapshot.draft_config_path,
                "active_run_id": self._snapshot.active_run_id,
                "active_checkpoint": self._snapshot.active_checkpoint,
                "last_error": self._snapshot.last_error,
                "metadata": dict(self._snapshot.metadata),
            }

    def set_idle(self) -> None:
        with self._lock:
            self._snapshot = OrchestrationSnapshot()

    def set_configured(
        self,
        *,
        selected_family: str | None = None,
        draft_config_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._snapshot.phase = "configured"
            self._snapshot.selected_family = selected_family
            self._snapshot.draft_config_path = draft_config_path
            self._snapshot.metadata = dict(metadata or {})
            self._snapshot.last_error = None

    def set_starting(
        self,
        *,
        selected_family: str | None = None,
        draft_config_path: str | None = None,
        active_run_id: str | None = None,
        active_checkpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._snapshot.phase = "starting"
            if selected_family is not None:
                self._snapshot.selected_family = selected_family
            if draft_config_path is not None:
                self._snapshot.draft_config_path = draft_config_path
            self._snapshot.active_run_id = active_run_id
            self._snapshot.active_checkpoint = active_checkpoint
            if metadata is not None:
                self._snapshot.metadata = dict(metadata)
            self._snapshot.last_error = None

    def set_running(
        self,
        *,
        active_run_id: str | None = None,
        active_checkpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._snapshot.phase = "running"
            if active_run_id is not None:
                self._snapshot.active_run_id = active_run_id
            if active_checkpoint is not None:
                self._snapshot.active_checkpoint = active_checkpoint
            if metadata:
                self._snapshot.metadata.update(metadata)
            self._snapshot.last_error = None

    def set_pausing(self) -> None:
        with self._lock:
            self._snapshot.phase = "pausing"

    def set_paused(self, *, active_run_id: str | None = None) -> None:
        with self._lock:
            self._snapshot.phase = "paused"
            if active_run_id is not None:
                self._snapshot.active_run_id = active_run_id

    def set_failed(self, error: str) -> None:
        with self._lock:
            self._snapshot.phase = "failed"
            self._snapshot.last_error = error
