from __future__ import annotations

import json
import mimetypes
import queue
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .serializers import sse_message
from ..common.training_manager import RunStartRequest, TrainingManager


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        *,
        state,
        web_root: Path,
        orchestration_state=None,
        training_manager: TrainingManager | None = None,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.state = state
        self.web_root = web_root
        self.orchestration_state = orchestration_state
        self.training_manager = training_manager
        # Derive allowed origins from the bound address. Both 127.0.0.1 and
        # localhost are always permitted so browser-launched tabs work regardless
        # of which alias the user typed.
        host, port = server_address
        self.allowed_origins: frozenset[str] = frozenset(
            filter(None, [
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
                f"http://{host}:{port}" if host not in ("0.0.0.0", "::", "127.0.0.1", "localhost") else None,
            ])
        )


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def do_GET(self):
        request_path = urlsplit(self.path).path

        if request_path == "/api/state":
            self.server.state.load_from_disk()
            payload = self.server.state.snapshot()
            payload["orchestration"] = self._orchestration_snapshot()
            payload["training_manager"] = self._training_manager_snapshot()
            return self._serve_json(payload)
        if request_path == "/api/orchestration":
            return self._serve_json(self._orchestration_snapshot())
        if request_path == "/api/config-templates":
            return self._serve_json(self._training_manager_list("list_templates"))
        if request_path == "/api/draft-configs":
            return self._serve_json(self._training_manager_list("list_drafts"))
        if request_path == "/api/checkpoints":
            return self._serve_json(self._training_manager_checkpoints())
        if request_path.startswith("/api/runs/"):
            return self._serve_run_state()
        if request_path == "/api/hardware":
            from ..common.hardware import get_hardware_info
            return self._serve_json(get_hardware_info())
        if request_path == "/api/store-info":
            return self._serve_store_info()
        if request_path == "/api/store-info/precompute":
            return self._serve_json(self._precompute_status())
        if request_path == "/api/health":
            return self._serve_json({"ok": True})
        if request_path == "/api/events":
            return self._serve_sse()
        if request_path.startswith("/runs/"):
            return self._serve_run_file()
        return self._serve_web_asset(request_path)

    def _check_origin(self) -> bool:
        """Return True if the request origin is allowed, False otherwise.

        Requests with no Origin header (e.g. curl, server-side tools) are
        always allowed. Only cross-origin browser requests carry an Origin, so
        rejecting unexpected origins is sufficient CSRF protection for a
        local-only dashboard.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin in self.server.allowed_origins

    def do_POST(self):
        if not self._check_origin():
            return self._serve_json(
                {"ok": False, "error": "Cross-origin request rejected"},
                status=HTTPStatus.FORBIDDEN,
            )
        request_path = urlsplit(self.path).path

        if request_path == "/api/store-info/precompute":
            return self._start_precompute()

        if request_path == "/api/draft-configs":
            payload = self._read_json_body()
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                result = manager.create_draft(
                    template_id=str(payload.get("template_id", "")),
                    family_name=payload.get("family_name"),
                    task_type=payload.get("task_type"),
                    payload=payload,
                )
            except FileExistsError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "draft": result}, status=HTTPStatus.CREATED)

        if request_path.startswith("/api/draft-configs/"):
            payload = self._read_json_body()
            family_name = unquote(request_path.rsplit("/", 1)[-1])
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                result = manager.save_draft(
                    family_name=family_name,
                    payload=payload,
                    template_id=payload.get("template_id"),
                )
            except FileExistsError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "draft": result})

        if request_path == "/api/runs/start":
            payload = self._read_json_body()
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                start_request = RunStartRequest.from_payload(payload)
                result = manager.start_run(start_request)
            except NotImplementedError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.NOT_IMPLEMENTED)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "job": result}, status=HTTPStatus.ACCEPTED)

        if request_path == "/api/runs/pause":
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                result = manager.pause_run()
            except NotImplementedError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.NOT_IMPLEMENTED)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "job": result}, status=HTTPStatus.ACCEPTED)
        if request_path == "/api/runs/resume":
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                payload = self._read_json_body() if self.headers.get("Content-Length") else {}
                result = manager.resume_run(
                    family_name=payload.get("family_name"),
                    run_id=payload.get("run_id"),
                )
            except NotImplementedError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.NOT_IMPLEMENTED)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "job": result}, status=HTTPStatus.ACCEPTED)

        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_PATCH(self):
        if not self._check_origin():
            return self._serve_json(
                {"ok": False, "error": "Cross-origin request rejected"},
                status=HTTPStatus.FORBIDDEN,
            )
        request_path = urlsplit(self.path).path

        if request_path.startswith("/api/runs/") and request_path.endswith("/star"):
            run_id = unquote(request_path[len("/api/runs/"):-len("/star")])
            if not run_id:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            payload = self._read_json_body()
            starred = bool(payload.get("starred", False))
            note = str(payload.get("note", ""))
            ok = self.server.state.set_star(run_id, starred, note)
            if not ok:
                return self._serve_json({"ok": False, "error": "Run not found"}, status=HTTPStatus.NOT_FOUND)
            return self._serve_json({"ok": True, "run_id": run_id, "starred": starred, "note": note})

        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        if not self._check_origin():
            return self._serve_json(
                {"ok": False, "error": "Cross-origin request rejected"},
                status=HTTPStatus.FORBIDDEN,
            )
        request_path = urlsplit(self.path).path

        if request_path == "/api/store-info/precompute":
            return self._cancel_precompute()

        if request_path.startswith("/api/draft-configs/"):
            family_name = unquote(request_path.rsplit("/", 1)[-1])
            manager = self._training_manager()
            if manager is None:
                return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            try:
                result = manager.delete_draft(family_name)
            except FileNotFoundError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self._serve_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return self._serve_json({"ok": True, "draft": result})

        if request_path.startswith("/api/runs/"):
            run_id = unquote(request_path[len("/api/runs/"):])
            if not run_id or "/" in run_id:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            # Ensure state is current before attempting the delete — the same
            # pattern used by _serve_run_state for GET requests.
            self.server.state.load_from_disk()
            ok = self.server.state.delete_run(run_id)
            if not ok:
                return self._serve_json({"ok": False, "error": "Run not found"}, status=HTTPStatus.NOT_FOUND)
            return self._serve_json({"ok": True, "run_id": run_id})

        return self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        # Only log non-2xx responses to avoid noise from frequent polling
        if args and not str(args[1]).startswith("2"):
            import sys
            print(f"[dashboard] {self.address_string()} {format % args}", file=sys.stderr)

    def _serve_store_info(self):
        """GET /api/store-info?path=<store_namespace_path>
        Returns {cache_type: "aligned"|"legacy"|"missing"} for a given store path.
        "aligned"  – built by precompute_aligned_masks.py (has _aligned_cache.json marker)
        "legacy"   – directory exists but was built without the aligned pipeline
        "missing"  – directory does not exist or path param absent
        """
        from urllib.parse import parse_qs, urlsplit
        qs = parse_qs(urlsplit(self.path).query)
        raw_path = qs.get("path", [None])[0]
        if not raw_path:
            return self._serve_json({"cache_type": "missing"})
            
        manager = self._training_manager()
        # Config paths are relative to the 'training' subdirectory
        base_dir = manager.repo_root / "training" if manager else Path("training")
        store_dir = (base_dir / raw_path).resolve()
        
        if not store_dir.is_dir():
            return self._serve_json({"cache_type": "missing"})
        if (store_dir / "_aligned_cache.json").exists():
            return self._serve_json({"cache_type": "aligned"})

        resolution = qs.get("resolution", [None])[0]  # e.g. "256x128"
        
        has_aligned = False
        has_legacy = False
        prefix = f"removal-{resolution}-" if resolution else None
        
        try:
            for subdir in store_dir.iterdir():
                if subdir.is_dir():
                    if prefix and not subdir.name.startswith(prefix):
                        continue
                        
                    if (subdir / "_aligned_cache.json").exists():
                        has_aligned = True
                        break
                    # If we find any subdirectory, it could be a legacy cache variant
                    has_legacy = True
        except Exception:
            pass

        if has_aligned:
            return self._serve_json({"cache_type": "aligned"})
        
        # Only return legacy if we found actual subdirectories, otherwise the cache root is essentially empty/missing
        if has_legacy:
            return self._serve_json({"cache_type": "legacy"})
            
        return self._serve_json({"cache_type": "missing"})

    def _precompute_status(self):
        proc = getattr(self.server, "precompute_process", None)
        is_running = proc is not None and proc.poll() is None
        return {"running": is_running}
        
    def _start_precompute(self):
        proc = getattr(self.server, "precompute_process", None)
        if proc is not None and proc.poll() is None:
            return self._serve_json({"ok": False, "error": "Already running"}, status=HTTPStatus.CONFLICT)
            
        payload = self._read_json_body()
        config_name = payload.get("config_path")
        if not config_name:
            return self._serve_json({"ok": False, "error": "config_path required"}, status=HTTPStatus.BAD_REQUEST)
            
        manager = self._training_manager()
        if not manager:
            return self.send_error(HTTPStatus.NOT_IMPLEMENTED)
            
        config_path = manager.dashboard_configs_root / config_name
            
        # Spawn process
        import subprocess
        import threading
        
        script = manager.repo_root / "training" / "precompute_aligned_masks.py"
        seg_ckpt = manager.repo_root / "artifacts" / "checkpoints" / "segmentation"  # Use latest inside loop or let script find it
        # Try to find a seg checkpoint
        ckpts = manager.list_checkpoints(task_type="segmentation")
        if not ckpts:
            return self._serve_json({"ok": False, "error": "No segmentation checkpoint found."}, status=HTTPStatus.BAD_REQUEST)
        
        # Sort to get the latest best
        def sort_key(rec):
            return (rec.get("epoch", -1) or -1, 1 if rec.get("kind") == "best" else 0)
        best_ckpt = max(ckpts, key=sort_key)
        
        cmd = [
            manager.python_executable,
            str(script),
            "--config", str(config_path),
            "--seg-checkpoint", str(best_ckpt["checkpoint_path"])
        ]
        
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1, cwd=manager.repo_root)
        except Exception as e:
            return self._serve_json({"ok": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            
        self.server.precompute_process = p
        
        def _track():
            if p.stdout:
                for line in p.stdout:
                    text = line.strip()
                    # Look for [10/1000]
                    if text.startswith("[") and "/" in text and "]" in text:
                        try:
                            parts = text.split("]")[0][1:].split("/")
                            current = int(parts[0])
                            total = int(parts[1])
                            self.server.state.publish("precompute_progress", {"current": current, "total": total})
                        except Exception:
                            pass
            p.wait()
            self.server.state.publish("precompute_progress", {"current": -1, "total": -1}) # Done/cancelled
            
        threading.Thread(target=_track, daemon=True).start()
        return self._serve_json({"ok": True, "status": "started"})
        
    def _cancel_precompute(self):
        proc = getattr(self.server, "precompute_process", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return self._serve_json({"ok": True, "status": "cancelled"})
        return self._serve_json({"ok": True, "status": "not_running"})

    def _serve_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _serve_static(self, path: Path, content_type: str | None = None):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._safe_write(data)

    def _serve_web_asset(self, request_path: str):
        web_path = self._resolve_web_path(request_path)
        if web_path is not None:
            return self._serve_static(web_path)

        index_path = self.server.web_root / "index.html"
        if request_path in {"/", ""} or Path(request_path).suffix == "":
            return self._serve_static(index_path)

        self.send_error(HTTPStatus.NOT_FOUND)

    def _resolve_web_path(self, request_path: str) -> Path | None:
        relative = request_path.lstrip("/") or "index.html"
        candidate = (self.server.web_root / relative).resolve()
        web_root = self.server.web_root.resolve()
        try:
            candidate.relative_to(web_root)
        except Exception:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _serve_run_file(self):
        request_path = urlsplit(self.path).path
        _, _, tail = request_path.partition("/runs/")
        segments = tail.split("/", 1)
        if len(segments) != 2:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        run_id = unquote(segments[0])
        relative_path = unquote(segments[1])
        path = self.server.state.resolve_run_file(run_id, relative_path)
        if path is None:
            # State may not have loaded this run yet — try once from disk
            self.server.state.load_from_disk()
            path = self.server.state.resolve_run_file(run_id, relative_path)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return self._serve_static(path)

    def _serve_run_state(self):
        request_path = urlsplit(self.path).path
        _, _, tail = request_path.partition("/api/runs/")
        run_id = unquote(tail.strip("/"))
        if not run_id:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.server.state.load_from_disk()
        payload = self.server.state.run_snapshot(run_id)
        if payload is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return self._serve_json(payload)

    def _orchestration_snapshot(self) -> dict:
        orchestration = getattr(self.server, "orchestration_state", None)
        if orchestration is None:
            return {
                "phase": "idle",
                "selected_family": None,
                "draft_config_path": None,
                "active_run_id": None,
                "active_checkpoint": None,
                "last_error": None,
                "metadata": {},
            }
        return orchestration.snapshot()

    def _training_manager(self) -> TrainingManager | None:
        return getattr(self.server, "training_manager", None)

    def _training_manager_snapshot(self) -> dict:
        manager = self._training_manager()
        if manager is None:
            return {}
        return manager.snapshot()

    def _training_manager_list(self, method_name: str) -> list[dict]:
        manager = self._training_manager()
        if manager is None:
            return []
        method = getattr(manager, method_name, None)
        if method is None:
            return []
        return method()

    def _training_manager_checkpoints(self) -> list[dict]:
        manager = self._training_manager()
        if manager is None:
            return []
        query = parse_qs(urlsplit(self.path).query)
        return manager.list_checkpoints(
            compatible_for=query.get("compatible_for", [None])[0],
            task_type=query.get("task_type", [None])[0],
            family_name=query.get("family_name", [None])[0],
            run_id=query.get("run_id", [None])[0],
        )

    _MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            return {}
        if length <= 0:
            return {}
        if length > self._MAX_BODY_BYTES:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _serve_sse(self):
        subscriber = self.server.state.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self._safe_write(sse_message("tick", {}))
                    self._safe_flush()
                    continue
                self._safe_write(sse_message(event["type"], event["data"]))
                self._safe_flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            self.server.state.unsubscribe(subscriber)

    def _safe_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _safe_flush(self) -> None:
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
