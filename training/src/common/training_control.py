from __future__ import annotations


class TrainingPaused(RuntimeError):
    """Raised by a trainer after Ctrl+C once the interrupt checkpoint is saved."""
