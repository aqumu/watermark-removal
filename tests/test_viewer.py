from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from training.src.dashboard.app import ViewerHTTPServer, ViewerRequestHandler
from training.src.dashboard.runs import RunIndex, resolve_status
from wm_shared.experiment import ExperimentSession
from wm_shared.run_manifest import create_run_manifest


def _make_run(root: Path, family: str = "removal_256"):
    return create_run_manifest(
        project_run=family,
        task_name="removal",
        root_dir=root / family,
        start_mode="fresh",
        history_mode="fork",
        config_fingerprint="cfg1234567890abcd",
        code_fingerprint="code1234567890ab",
        device="cpu",
    )


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _age(path: Path, seconds: float) -> None:
    ts = time.time() - seconds
    os.utime(path, (ts, ts))


def test_index_reads_run_files(tmp_path: Path):
    manifest = _make_run(tmp_path)
    run_dir = manifest.paths.run_dir
    _write(manifest.paths.train_csv, "epoch,step,global_step,total,lr\n1,10,10,0.5,1e-3\n1,20,20,0.4,1e-3\n")
    _write(manifest.paths.val_csv, "epoch,global_step,psnr,psnr_masked\n1,20,30.1,25.2\n")
    _write(run_dir / "previews" / "000001" / "blended_output.png", b"png")
    _write(run_dir / "meta" / "config.yaml", "training:\n  epochs: 50\n")

    index = RunIndex([tmp_path])
    index.scan()

    snapshot = index.snapshot()
    assert [r["manifest"]["run_id"] for r in snapshot["runs"]] == [manifest.identity.run_id]
    summary = snapshot["runs"][0]
    assert len(summary["train_metrics"]) == 1  # the list only carries the latest row
    assert summary["manifest"]["progress"]["epochs_total"] == 50

    detail = index.run_detail(manifest.identity.run_id)
    assert len(detail["train_metrics"]) == 2
    assert detail["val_metrics"][0]["psnr_masked"] == 25.2
    assert detail["preview_history"][0]["previews"]["blended_output"].endswith("/previews/000001/blended_output.png")


def test_scan_reports_only_changed_runs(tmp_path: Path):
    first = _make_run(tmp_path, "removal_256")
    time.sleep(1.1)  # run ids carry a seconds timestamp
    second = _make_run(tmp_path, "segmentation")
    index = RunIndex([tmp_path])
    changed, list_changed = index.scan()
    assert changed == {first.identity.run_id, second.identity.run_id} and list_changed

    assert index.scan() == (set(), False)

    _write(second.paths.train_csv, "epoch,step,global_step,total\n1,10,10,0.5\n")
    changed, list_changed = index.scan()
    assert changed == {second.identity.run_id} and not list_changed


def test_running_without_heartbeat_is_shown_as_stopped(tmp_path: Path):
    manifest = _make_run(tmp_path)
    run_dir = manifest.paths.run_dir
    _write(run_dir / "meta" / "status.json", json.dumps({
        "status": "running", "epoch": 3, "updated_at": "2020-01-01T00:00:00+00:00",
    }))
    _age(run_dir / "meta" / "run.json", 3600)

    live = resolve_status(run_dir, "running", now=time.time(), stale_after=600)
    assert live["status"] == "stopped"
    assert live["progress"] == {"epoch": 3}

    _write(manifest.paths.train_csv, "epoch,step\n3,10\n")  # fresh activity
    assert resolve_status(run_dir, "running", now=time.time(), stale_after=600)["status"] == "running"


def test_finished_statuses_are_not_touched_by_staleness(tmp_path: Path):
    manifest = _make_run(tmp_path)
    run_dir = manifest.paths.run_dir
    _write(run_dir / "meta" / "status.json", json.dumps({"status": "completed"}))
    _age(run_dir / "meta" / "run.json", 3600)
    assert resolve_status(run_dir, "running", now=time.time(), stale_after=1)["status"] == "completed"


def test_experiment_session_writes_status_and_heartbeat(tmp_path: Path):
    manifest = _make_run(tmp_path)
    cfg = {"logging": {"dir": str(manifest.paths.run_dir)}, "device": "cpu"}
    experiment = ExperimentSession(task_name="removal", cfg=cfg, manifest=manifest)
    status_path = manifest.paths.meta_dir / "status.json"
    assert json.loads(status_path.read_text())["status"] == "running"

    experiment.heartbeat(epoch=2, step=30, global_step=130)
    payload = json.loads(status_path.read_text())
    assert (payload["epoch"], payload["step"], payload["global_step"]) == (2, 30, 130)

    experiment.set_status("paused")
    payload = json.loads(status_path.read_text())
    assert payload["status"] == "paused" and payload["epoch"] == 2
    assert json.loads((manifest.paths.meta_dir / "run.json").read_text())["identity"]["status"] == "paused"


def test_hidden_runs_disappear_and_star_is_kept(tmp_path: Path):
    manifest = _make_run(tmp_path)
    index = RunIndex([tmp_path])
    index.scan()
    run_id = manifest.identity.run_id

    assert index.set_star(run_id, True, "best so far")
    assert index.run_detail(run_id)["manifest"]["starred"] is True

    assert index.hide_run(run_id)
    assert index.snapshot()["runs"] == []
    assert manifest.paths.run_dir.exists()


def test_log_tail_is_bounded(tmp_path: Path):
    manifest = _make_run(tmp_path)
    _write(manifest.paths.run_dir / "training.log", "".join(f"line {i}\n" for i in range(10000)))
    index = RunIndex([tmp_path])
    index.scan()
    tail = index.log_tail(manifest.identity.run_id, max_bytes=1000)
    assert tail["truncated"]
    assert tail["text"].endswith("line 9999\n")
    assert tail["text"].startswith("line ")


@pytest.fixture()
def server(tmp_path: Path):
    manifest = _make_run(tmp_path)
    _write(manifest.paths.train_csv, "epoch,step,global_step,total\n1,10,10,0.5\n")
    _write(manifest.paths.run_dir / "training.log", "hello\n")
    web_root = tmp_path / "web"
    _write(web_root / "index.html", "<html></html>")
    index = RunIndex([tmp_path], poll_interval=0.1)
    index.start()
    httpd = ViewerHTTPServer(("127.0.0.1", 0), ViewerRequestHandler, index=index, web_root=web_root)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", manifest
    httpd.shutdown()
    httpd.server_close()
    index.stop()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def test_http_routes(server):
    base, manifest = server
    run_id = manifest.identity.run_id

    status, body = _get(f"{base}/api/state")
    assert status == 200 and json.loads(body)["runs"][0]["manifest"]["run_id"] == run_id

    status, body = _get(f"{base}/api/runs/{run_id}")
    assert json.loads(body)["train_metrics"][0]["total"] == 0.5

    status, body = _get(f"{base}/api/runs/{run_id}/log")
    assert json.loads(body)["text"] == "hello\n"

    status, body = _get(f"{base}/runs/{run_id}/train.csv")
    assert body.startswith(b"epoch,step")

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/runs/{run_id}/../../web/index.html")
    assert exc.value.code == 404

    status, body = _get(f"{base}/some/client/route")
    assert body == b"<html></html>"


def test_cross_origin_annotation_is_rejected(server):
    base, manifest = server
    request = urllib.request.Request(
        f"{base}/api/runs/{manifest.identity.run_id}",
        method="DELETE",
        headers={"Origin": "http://evil.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 403


def test_events_stream_reports_changes(server):
    base, manifest = server
    response = urllib.request.urlopen(f"{base}/api/events", timeout=5)
    assert response.readline().startswith(b"event: hello")
    response.readline()
    response.readline()

    with open(manifest.paths.train_csv, "a", encoding="utf-8") as f:
        f.write("1,20,20,0.4\n")

    line = response.readline()
    assert line.startswith(b"event: runs_changed")
    data = json.loads(response.readline().split(b"data: ", 1)[1])
    assert data["run_ids"] == [manifest.identity.run_id]
    response.close()
