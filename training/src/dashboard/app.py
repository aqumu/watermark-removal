"""
HTTP server for the read-only run viewer.

GET  /api/state                 run list (summaries)
GET  /api/runs/<run_id>         full run record (metrics, previews, config, artifacts)
GET  /api/runs/<run_id>/log     tail of training.log
GET  /api/events                server-sent events: "runs_changed" when files on disk change
GET  /runs/<run_id>/<path>      a file from the run folder (previews, config, checkpoints)
PATCH  /api/runs/<run_id>/star  viewer annotation (meta/star.json)
DELETE /api/runs/<run_id>       hide the run from the list (meta/deleted.json, files are kept)
"""

from __future__ import annotations

import json
import mimetypes
import queue
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .runs import RunIndex

SSE_KEEPALIVE_S = 15
_MAX_BODY_BYTES = 64 * 1024


class _ClientGone(Exception):
    pass


class ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, *, index: RunIndex, web_root: Path):
        super().__init__(server_address, RequestHandlerClass)
        self.index = index
        self.web_root = web_root
        # Browser requests that change annotations must come from the viewer page itself.
        host, port = server_address[:2]
        self.allowed_origins: frozenset[str] = frozenset(
            filter(None, [
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
                f"http://{host}:{port}" if host not in ("0.0.0.0", "::", "127.0.0.1", "localhost") else None,
            ])
        )

    def handle_error(self, request, client_address):
        # Browsers drop keep-alive connections on reload; that is not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server: ViewerHTTPServer

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlsplit(self.path).path

        if path == "/api/state":
            return self._send_json(self.server.index.snapshot())
        if path == "/api/health":
            return self._send_json({"ok": True})
        if path == "/api/events":
            return self._serve_events()
        if path.startswith("/api/runs/"):
            run_id, _, action = unquote(path[len("/api/runs/"):]).strip("/").partition("/")
            if action == "log":
                payload = self.server.index.log_tail(run_id)
            elif action == "":
                payload = self.server.index.run_detail(run_id)
            else:
                payload = None
            if payload is None:
                return self._send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return self._send_json(payload)
        if path.startswith("/runs/"):
            return self._serve_run_file(path)
        return self._serve_web_asset(path)

    def do_PATCH(self):
        if not self._origin_allowed():
            return self._send_json({"error": "Cross-origin request rejected"}, HTTPStatus.FORBIDDEN)
        path = urlsplit(self.path).path
        if path.startswith("/api/runs/") and path.endswith("/star"):
            run_id = unquote(path[len("/api/runs/"):-len("/star")])
            payload = self._read_json_body()
            if payload is None:
                return self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
            starred = bool(payload.get("starred", False))
            note = str(payload.get("note", ""))[:500]
            if not self.server.index.set_star(run_id, starred, note):
                return self._send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return self._send_json({"ok": True, "run_id": run_id, "starred": starred, "note": note})
        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        if not self._origin_allowed():
            return self._send_json({"error": "Cross-origin request rejected"}, HTTPStatus.FORBIDDEN)
        path = urlsplit(self.path).path
        if path.startswith("/api/runs/"):
            run_id = unquote(path[len("/api/runs/"):]).strip("/")
            if not run_id or "/" in run_id:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            if not self.server.index.hide_run(run_id):
                return self._send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return self._send_json({"ok": True, "run_id": run_id})
        return self.send_error(HTTPStatus.NOT_FOUND)

    # ── handlers ─────────────────────────────────────────────────────────────

    def _serve_events(self):
        subscriber = self.server.index.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._write_event("hello", {})
            while True:
                try:
                    event = subscriber.get(timeout=SSE_KEEPALIVE_S)
                except queue.Empty:
                    self._write_raw(b": keepalive\n\n")
                    continue
                self._write_event(event["type"], event["data"])
        except _ClientGone:
            return
        finally:
            self.server.index.unsubscribe(subscriber)

    def _serve_run_file(self, path: str):
        run_id, _, relative = path[len("/runs/"):].partition("/")
        file_path = self.server.index.resolve_run_file(unquote(run_id), unquote(relative))
        if file_path is None:
            return self.send_error(HTTPStatus.NOT_FOUND)
        return self._send_file(file_path, cache=False)

    def _serve_web_asset(self, path: str):
        web_root = self.server.web_root.resolve()
        relative = path.lstrip("/") or "index.html"
        candidate = (web_root / relative).resolve()
        try:
            candidate.relative_to(web_root)
        except ValueError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        if candidate.is_file():
            # Built assets have content hashes in their names, so they can be cached.
            return self._send_file(candidate, cache=relative.startswith("assets/"))
        if Path(path).suffix == "":
            index_path = web_root / "index.html"
            if index_path.is_file():
                return self._send_file(index_path, cache=False)
            return self._send_text(
                "The viewer frontend is not built. Run `npm install && npm run build` in "
                "training/src/dashboard/frontend.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        return self.send_error(HTTPStatus.NOT_FOUND)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in self.server.allowed_origins

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            return None
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status, cache=False)

    def _send_text(self, text: str, status: HTTPStatus):
        self._send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status, cache=False)

    def _send_file(self, path: Path, *, cache: bool):
        try:
            data = path.read_bytes()
        except OSError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".log", ".yaml", ".yml", ".csv"}:
            content_type = "text/plain; charset=utf-8"
        self._send_bytes(data, content_type, HTTPStatus.OK, cache=cache)

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus, *, cache: bool):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _write_event(self, event_type: str, data: dict):
        payload = json.dumps(data, separators=(",", ":"))
        self._write_raw(f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8"))

    def _write_raw(self, data: bytes):
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
            raise _ClientGone() from exc

    def log_message(self, format, *args):
        # Only log errors; the page polls and streams constantly.
        if len(args) > 1 and not str(args[1]).startswith(("2", "3")):
            print(f"[viewer] {self.address_string()} {format % args}", file=sys.stderr)
