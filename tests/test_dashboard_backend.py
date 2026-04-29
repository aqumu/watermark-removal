from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from training.src.dashboard.app import DashboardHTTPServer, DashboardRequestHandler
from training.src.dashboard.state import DashboardState
from wm_shared.run_manifest import create_run_manifest


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_manifest(tmp_path: Path):
    return create_run_manifest(
        project_run="removal_512",
        task_name="removal",
        root_dir=tmp_path / "runs" / "removal_512",
        start_mode="fresh",
        history_mode="fork",
        config_fingerprint="cfg1234567890abcd",
        code_fingerprint="code1234567890ab",
        device="cpu",
    )


def test_dashboard_state_loads_preview_history_from_disk(tmp_path: Path):
    manifest = _make_manifest(tmp_path)

    _write_text(
        manifest.paths.train_csv,
        "epoch,step,global_step,total\n1,1,1,0.5\n1,2,2,0.4\n",
    )
    _write_text(
        manifest.paths.val_csv,
        "epoch,global_step,psnr_masked\n1,2,25.4\n",
    )
    _write_text(
        manifest.paths.meta_dir / "model_overview.json",
        json.dumps({"model_name": "Net"}),
    )
    _write_bytes(manifest.paths.preview_dir / "000001" / "sample.png", b"png-one")
    _write_bytes(manifest.paths.preview_dir / "000002" / "sample.png", b"png-two")
    _write_bytes(manifest.paths.artifact_dir / "checkpoints" / "best.pth", b"weights")

    state = DashboardState(family_root=manifest.paths.root_dir, current_run_id=manifest.identity.run_id)
    state.load_from_disk()

    snapshot = state.snapshot()
    current = snapshot["current_run"]

    assert current is not None
    assert len(current["train_metrics"]) == 2
    assert len(current["val_metrics"]) == 1
    assert current["model_overview"]["model_name"] == "Net"
    assert [entry["label"] for entry in current["preview_history"]] == ["000001", "000002"]
    assert current["previews"]["sample"].endswith("/previews/000002/sample.png")
    assert current["artifacts"][0]["url"].endswith("/artifacts/checkpoints/best.pth")


def test_dashboard_state_and_api_expose_live_preview_history(tmp_path: Path):
    manifest = _make_manifest(tmp_path)
    state = DashboardState(family_root=manifest.paths.root_dir, current_run_id=manifest.identity.run_id)
    state.register_run(manifest)
    state.set_preview(manifest, 12, {"sample": f"/runs/{manifest.identity.run_id}/previews/000012/sample.png"})

    web_root = tmp_path / "web"
    _write_text(web_root / "index.html", "<!doctype html><title>ok</title>")
    _write_text(web_root / "app.js", "console.log('ok')")
    _write_text(web_root / "styles.css", "body{}")

    server = DashboardHTTPServer(("127.0.0.1", 0), DashboardRequestHandler, state=state, web_root=web_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base_url}/api/runs/{manifest.identity.run_id}") as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["manifest"]["run_id"] == manifest.identity.run_id
    assert payload["preview_history"] == [
        {
            "step": 12,
            "label": "000012",
            "previews": {"sample": f"/runs/{manifest.identity.run_id}/previews/000012/sample.png"},
        }
    ]
    assert payload["previews"]["sample"].endswith("/previews/000012/sample.png")


def test_dashboard_serves_run_files_with_cache_busting_query(tmp_path: Path):
    manifest = _make_manifest(tmp_path)
    state = DashboardState(family_root=manifest.paths.root_dir, current_run_id=manifest.identity.run_id)
    state.register_run(manifest)

    preview_path = manifest.paths.preview_dir / "000001" / "sample.png"
    _write_bytes(preview_path, b"png-data")

    web_root = tmp_path / "web"
    _write_text(web_root / "index.html", "<!doctype html><title>ok</title>")
    _write_text(web_root / "app.js", "console.log('ok')")
    _write_text(web_root / "styles.css", "body{}")

    server = DashboardHTTPServer(("127.0.0.1", 0), DashboardRequestHandler, state=state, web_root=web_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(
            f"{base_url}/runs/{manifest.identity.run_id}/previews/000001/sample.png?t=123"
        ) as response:
            body = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert body == b"png-data"
