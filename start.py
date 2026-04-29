from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import threading
import webbrowser
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.dashboard.app import DashboardHTTPServer, DashboardRequestHandler
from training.src.dashboard.state import DashboardState
from training.src.common.orchestration_state import OrchestrationState
from training.src.common.training_manager import TrainingManager


def _reconcile_stale_running_statuses(runs_root: Path) -> None:
    """At startup the orchestration state is always idle, so any run.json with
    status='running' on disk is an orphan from a crashed/killed process.
    Rewrite those to 'unknown' so the UI does not show them as actively running."""
    for run_json in runs_root.rglob("meta/run.json"):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            changed = False
            if "identity" in payload and isinstance(payload["identity"], dict):
                if payload["identity"].get("status") == "running":
                    payload["identity"]["status"] = "unknown"
                    changed = True
            elif isinstance(payload, dict) and payload.get("status") == "running":
                payload["status"] = "unknown"
                changed = True
            if changed:
                run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="training/runs", help="root directory containing run families")
    parser.add_argument("--web-root", default="training/src/dashboard/web", help="directory containing built dashboard assets")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    web_root = Path(args.web_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    _reconcile_stale_running_statuses(runs_root)
    state = DashboardState(family_root=runs_root, current_run_id=None)
    state.load_from_disk()
    orchestration_state = OrchestrationState()
    training_manager = TrainingManager(
        repo_root=REPO_ROOT,
        runs_root=runs_root,
        configs_root=REPO_ROOT / "training" / "configs",
        orchestration_state=orchestration_state,
    )

    server = DashboardHTTPServer(
        (args.host, args.port),
        DashboardRequestHandler,
        state=state,
        web_root=web_root,
        orchestration_state=orchestration_state,
        training_manager=training_manager,
    )
    local_url = f"http://127.0.0.1:{args.port}/"
    print(f"[dashboard] listening on {local_url}")
    def _open_browser_when_ready() -> None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                with closing(socket.create_connection(("127.0.0.1", args.port), timeout=0.5)):
                    webbrowser.open(local_url)
                    return
            except OSError:
                time.sleep(0.25)

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
