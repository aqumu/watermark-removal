from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DashboardEventSink(Protocol):
    def set_status(self, status: str) -> None:
        ...

    def log_model_overview(self, overview: dict) -> None:
        ...

    def log_train_metrics(self, row: dict) -> None:
        ...

    def log_val_metrics(self, row: dict) -> None:
        ...

    def log_preview(self, step_or_epoch: int, name_to_path: dict[str, Path]) -> None:
        ...

    def log_artifact(self, *, category: str, path: Path, name: str | None = None) -> None:
        ...


class NullDashboardEventSink:
    def set_status(self, status: str) -> None:
        pass

    def log_model_overview(self, overview: dict) -> None:
        pass

    def log_train_metrics(self, row: dict) -> None:
        pass

    def log_val_metrics(self, row: dict) -> None:
        pass

    def log_preview(self, step_or_epoch: int, name_to_path: dict[str, Path]) -> None:
        pass

    def log_artifact(self, *, category: str, path: Path, name: str | None = None) -> None:
        pass
