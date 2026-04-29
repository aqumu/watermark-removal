from __future__ import annotations

import socket
import threading
import time
import webbrowser
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wm_shared.run_manifest import RunManifest

from .dashboard_events import NullDashboardEventSink
from ..dashboard.app import DashboardHTTPServer, DashboardRequestHandler
from ..dashboard.state import DashboardState


@dataclass
class DashboardAddresses:
    host: str
    port: int
    bind_host: str
    local_url: str
    lan_url: str | None


class LiveDashboardEventSink(NullDashboardEventSink):
    def __init__(self, *, state: DashboardState, manifest: RunManifest):
        self.state = state
        self.manifest = manifest
        self.state.register_run(manifest)

    def set_status(self, status: str) -> None:
        self.state.set_status(self.manifest, status)

    def log_model_overview(self, overview: dict) -> None:
        self.state.set_model_overview(self.manifest, overview)

    def log_train_metrics(self, row: dict) -> None:
        self.state.append_train_metrics(self.manifest, row)

    def log_val_metrics(self, row: dict) -> None:
        self.state.append_val_metrics(self.manifest, row)

    def log_preview(self, step_or_epoch: int, name_to_path: dict[str, Path]) -> None:
        preview_urls = {}
        for name, path in name_to_path.items():
            rel_path = path.relative_to(self.manifest.paths.run_dir).as_posix()
            preview_urls[name] = f"/runs/{self.manifest.identity.run_id}/{rel_path}"
        self.state.set_preview(self.manifest, step_or_epoch, preview_urls)

    def log_artifact(self, *, category: str, path: Path, name: str | None = None) -> None:
        rel_path = path.relative_to(self.manifest.paths.run_dir).as_posix()
        artifact = {
            "name": name or path.name,
            "category": category,
            "url": f"/runs/{self.manifest.identity.run_id}/{rel_path}",
        }
        self.state.add_artifact(self.manifest, artifact)


class DashboardRuntime:
    def __init__(self, *, family_root: Path, manifest: RunManifest | None, host: str, port: int, open_browser: bool):
        self.family_root = family_root
        self.manifest = manifest
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.state = DashboardState(family_root=family_root, current_run_id=manifest.identity.run_id if manifest else None)
        self.server: DashboardHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.addresses: DashboardAddresses | None = None

    def start(self) -> DashboardAddresses:
        self.state.load_from_disk()
        if self.manifest is not None:
            self.state.register_run(self.manifest)

        bind_host = self.host if self.host not in {"0.0.0.0", "::"} else "0.0.0.0"
        port = self._find_open_port(bind_host, self.port)
        web_root = Path(__file__).resolve().parent.parent / "dashboard" / "web"
        self.server = DashboardHTTPServer((bind_host, port), DashboardRequestHandler, state=self.state, web_root=web_root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        lan_ip = self._resolve_lan_ip()
        local_url = f"http://127.0.0.1:{port}/"
        lan_url = f"http://{lan_ip}:{port}/" if lan_ip else None
        self.addresses = DashboardAddresses(
            host=self.host,
            port=port,
            bind_host=bind_host,
            local_url=local_url,
            lan_url=lan_url,
        )

        print(f"[dashboard] listening on {local_url}")
        if lan_url:
            print(f"[dashboard] LAN URL {lan_url}")

        if self.open_browser:
            threading.Thread(target=self._open_browser_when_ready, args=(port, local_url), daemon=True).start()

        return self.addresses

    def create_sink(self) -> LiveDashboardEventSink | NullDashboardEventSink:
        if self.manifest is None:
            return NullDashboardEventSink()
        return LiveDashboardEventSink(state=self.state, manifest=self.manifest)

    def close(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.server = None
        self.thread = None

    @staticmethod
    def _find_open_port(host: str, preferred_port: int) -> int:
        probe_host = host if host != "0.0.0.0" else "127.0.0.1"
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((probe_host, preferred_port)) != 0:
                return preferred_port

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _resolve_lan_ip() -> str | None:
        for candidate in ("8.8.8.8", "1.1.1.1", "192.168.1.1"):
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                    sock.connect((candidate, 80))
                    ip = str(sock.getsockname()[0])
                    if ip and ip != "0.0.0.0":
                        return ip
            except OSError:
                continue
        return None

    @staticmethod
    def _open_browser_when_ready(port: int, url: str) -> None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                with closing(socket.create_connection(("127.0.0.1", port), timeout=0.5)):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.25)


def maybe_start_dashboard(*, cfg: dict[str, Any], manifest: RunManifest) -> DashboardRuntime | None:
    dashboard_cfg = cfg.get("dashboard", {})
    if not dashboard_cfg.get("enabled", False):
        return None

    runtime = DashboardRuntime(
        family_root=Path(cfg["logging"].get("family_dir", manifest.paths.root_dir)),
        manifest=manifest,
        host=str(dashboard_cfg.get("host", "0.0.0.0")),
        port=int(dashboard_cfg.get("port", 8765)),
        open_browser=bool(dashboard_cfg.get("open_browser", True)),
    )
    runtime.start()
    return runtime
