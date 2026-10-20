from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

LOG_FILENAME = "training.log"


class _Tee:
    """Writes to the original console stream and to the run's log file."""

    def __init__(self, stream: TextIO, log_file: TextIO):
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except (OSError, ValueError):
            pass
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_console_to_run_log(run_dir: str | Path) -> Path:
    """Mirror stdout and stderr into <run_dir>/training.log (appending on resume)."""
    path = Path(run_dir) / LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(path, "a", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return path
