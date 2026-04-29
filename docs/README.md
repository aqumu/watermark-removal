# Operator Workflow

This document summarizes the recommended dashboard-first workflow for training and inspection.

## Start The Dashboard

```bash
python start.py
```

This boots the dashboard server, orchestration backend, and launcher shell against `training/runs/`.

## In The UI

- Open an existing run to inspect history.
- Create a draft config for a new run family.
- Start a fresh run or load weights from a compatible checkpoint.
- Pause the active job when you need a safe stop.
- Resume a paused job from the header controls.

## Standalone Inspect Mode

```bash
python training/serve_dashboard.py --family-dir training/runs
```

Use this when you only need to review run history and do not want to launch training.

## Notes

- The standalone `training/serve_dashboard.py` launcher is inspect-only on restart and does not reattach to orphaned jobs.
- Live training remains server-owned; the dashboard does not reattach to orphaned jobs.
