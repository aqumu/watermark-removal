from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import yaml

from training.src.common.orchestration_state import OrchestrationState
from training.src.common.training_control import clear_pause_request, pause_requested
from training.src.common.training_manager import ActiveJobRecord, RunStartRequest, TrainingManager


def test_pause_run_writes_control_file_and_marks_state(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    manager = TrainingManager(
        repo_root=tmp_path,
        runs_root=runs_root,
        configs_root=configs_root,
        orchestration_state=OrchestrationState(),
        python_executable="python",
    )

    run_dir = runs_root / "demo_family" / "demo_family__20260408_000001"
    job = ActiveJobRecord(
        command=["python", "train.py"],
        process=SimpleNamespace(poll=lambda: None),
        family_name="demo_family",
        template_id="train_256",
        task_type="removal",
        draft_path=configs_root / "dashboard" / "demo_family.yaml",
        launch_config_path=configs_root / "dashboard" / ".generated" / "demo_family__launch.yaml",
        mode="fresh",
        run_id=run_dir.name,
        run_dir=run_dir,
    )

    with manager._lock:
        manager._active_job = job

    result = manager.pause_run()

    assert result["family_name"] == "demo_family"
    assert manager.orchestration_state.snapshot()["phase"] == "pausing"
    assert pause_requested(run_dir)

    clear_pause_request(run_dir)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_manager(tmp_path: Path) -> TrainingManager:
    runs_root = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    _write_yaml(
        configs_root / "train_256.yaml",
        {
            "seed": 42,
            "dataset": {
                "root": "dataset",
                "image_width": 256,
                "image_height": 256,
            },
            "training": {
                "epochs": 2,
                "batch_size": 1,
            },
            "logging": {},
        },
    )
    _write_yaml(
        configs_root / "dashboard" / "demo_family.yaml",
        {
            "seed": 42,
            "dataset": {
                "root": "dataset",
                "image_width": 256,
                "image_height": 256,
            },
            "training": {
                "epochs": 2,
                "batch_size": 1,
            },
            "logging": {},
            "dashboard": {
                "task_type": "removal",
            },
        },
    )
    manager = TrainingManager(
        repo_root=tmp_path,
        runs_root=runs_root,
        configs_root=configs_root,
        orchestration_state=OrchestrationState(),
        python_executable="python",
    )
    manager._track_job = lambda job: None
    manager._capture_output = lambda job: None
    return manager


def test_start_run_uses_headless_web_launch_config_for_fresh_runs(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(
            poll=lambda: None,
            wait=lambda: 0,
            stdout=[],
        )

    with patch("training.src.common.training_manager.subprocess.Popen", side_effect=_fake_popen):
        result = manager.start_run(RunStartRequest(family_name="demo_family", mode="fresh"))

    assert result["mode"] == "fresh"
    assert captured["command"] == [
        "python",
        str(tmp_path / "training" / "train.py"),
        "--config",
        result["launch_config_path"],
    ]
    launch_cfg = yaml.safe_load(Path(result["launch_config_path"]).read_text(encoding="utf-8"))
    assert launch_cfg["dashboard"]["enabled"] is False
    assert launch_cfg["dashboard"]["use_legacy_plotter"] is False


def test_start_run_maps_checkpoint_launch_modes_to_expected_cli_flags(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    commands: list[list[str]] = []

    def _fake_popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            poll=lambda: None,
            wait=lambda: 0,
            stdout=[],
        )

    checkpoint_path = str(tmp_path / "runs" / "demo_family" / "demo_run" / "artifacts" / "checkpoints" / "epoch_0002.pth")
    with patch("training.src.common.training_manager.subprocess.Popen", side_effect=_fake_popen):
        manager.start_run(
            RunStartRequest(
                family_name="demo_family",
                mode="load_weights",
                checkpoint_path=checkpoint_path,
            )
        )
        manager._active_job = None
        manager.start_run(
            RunStartRequest(
                family_name="demo_family",
                mode="resume",
                checkpoint_path=checkpoint_path,
            )
        )

    assert commands[0][-2:] == ["--load-weights", checkpoint_path]
    assert commands[1][-2:] == ["--resume", checkpoint_path]
