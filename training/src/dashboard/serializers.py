from __future__ import annotations

from typing import Any


def sse_message(event_type: str, payload: dict[str, Any]) -> bytes:
    import json

    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")
