"""
view.py  read-only viewer for training runs
-------------------------------------------
Shows metrics, previews, config, artifacts and logs of every run on disk.
It never starts, stops or configures training: run train.py / train_seg.py
from a terminal and leave this open.

Usage:
  python view.py                         # scans runs/ and training/runs/
  python view.py --runs path/to/runs     # one or more run roots
  python view.py --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.dashboard.app import ViewerHTTPServer, ViewerRequestHandler
from training.src.dashboard.runs import DEFAULT_STALE_AFTER_S, RunIndex

DEFAULT_ROOTS = (REPO_ROOT / "runs", REPO_ROOT / "training" / "runs")
WEB_ROOT = REPO_ROOT / "training" / "src" / "dashboard" / "web"


def _lan_ip() -> str | None:
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="run roots to scan (default: runs/ and training/runs/)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address; use 127.0.0.1 to keep it off the LAN")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER_S,
                        help="seconds without any activity before a 'running' run is shown as stopped")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.runs] if args.runs else list(DEFAULT_ROOTS)
    index = RunIndex(roots, stale_after=args.stale_after)
    index.start()

    server = ViewerHTTPServer((args.host, args.port), ViewerRequestHandler, index=index, web_root=WEB_ROOT)
    local_url = f"http://127.0.0.1:{args.port}/"
    print(f"[viewer] watching {', '.join(str(r) for r in index.roots)}")
    print(f"[viewer] {len(index.snapshot()['runs'])} runs found")
    print(f"[viewer] open {local_url}")
    if args.host in ("0.0.0.0", "::"):
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"[viewer] LAN  http://{lan_ip}:{args.port}/")
    if not (WEB_ROOT / "index.html").is_file():
        print("[viewer] frontend not built: cd training/src/dashboard/frontend && npm install && npm run build")

    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(local_url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        index.stop()


if __name__ == "__main__":
    main()
