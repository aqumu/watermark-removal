# Operator Workflow

## Train From The Command Line

```bash
cd training
python train.py --config configs/train_512.yaml          # removal
python train_seg.py --config configs/seg.yaml            # segmentation
python train.py --resume runs/removal_512/<run_id>/artifacts/checkpoints/epoch_0010.pth
python train.py --load-weights path/to/best.pth          # new run from existing weights
```

Press Ctrl+C to pause: the trainer saves a checkpoint, marks the run `paused` and
exits. Resume it with `--resume` on that checkpoint. If the config is unchanged the
run continues in the same folder; otherwise a new run is created with the old one
as its parent.

## Watch Runs

```bash
python view.py                    # scans runs/ and training/runs/
python view.py --runs some/other/runs --port 8770 --no-browser
python view.py --host 127.0.0.1   # keep it off the LAN
```

The viewer is read-only. It can be started and stopped at any time, before,
during or after training, and any number of runs can train while it is open.

## Run Folder Contract

The viewer depends only on these files. Anything the trainers write here shows
up automatically, including new loss columns in `train.csv`.

```text
<runs root>/<family>/<run_id>/
  meta/run.json              identity, lineage, final status
  meta/status.json           status + heartbeat: epoch, step, global_step, updated_at
  meta/config.yaml           resolved config
  meta/model_overview.json   model name, parameter count, optimizer
  train.csv                  one row every log_every steps
  val.csv                    one row per validation
  previews/<epoch>/*.png     preview images
  artifacts/checkpoints/     checkpoints
  training.log               console output (stdout + stderr)
  meta/star.json             viewer: starred + note
  meta/deleted.json          viewer: hidden from the list
```

Status values: `running`, `paused` (Ctrl+C), `completed`, `failed` (exception),
and `stopped`, which the viewer shows for a `running` run with no file activity
for `--stale-after` seconds (default 900), i.e. a killed or hard-crashed process.
