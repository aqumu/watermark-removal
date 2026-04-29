import csv
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn


def _safe_delete(path: Path, max_retries: int = 5):
    """Attempt to delete a file with retries for Windows locking issues."""
    for i in range(max_retries):
        try:
            if path.exists():
                path.unlink()
            return True
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    return False


def _save_ckpt(ckpt_dir: Path, epoch: int, model: nn.Module,
               optimizer, scheduler, best_psnr: float, keep_last: int,
               ema=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:04d}.pth"

    # Remove any orphaned temp files left by a previous crashed save
    for stale_tmp in ckpt_dir.glob("*.pth.tmp"):
        _safe_delete(stale_tmp)

    import tempfile
    fd, temp_path_str = tempfile.mkstemp(suffix=".pth.tmp")
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_psnr": best_psnr,
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()

        torch.save(payload, temp_path)
        time.sleep(0.2)
        _safe_delete(path)

        for i in range(10):
            try:
                shutil.move(str(temp_path), str(path))
                break
            except (PermissionError, OSError):
                time.sleep(0.5 * (i + 1))
    except Exception as exc:
        _safe_delete(temp_path)
        print(f"  [WARNING] Failed to save checkpoint at epoch {epoch}: {exc}")
        return None

    ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
    while len(ckpts) > keep_last:
        _safe_delete(ckpts[0])
        ckpts = ckpts[1:]
    return path


def _save_ckpt_to(path: Path, epoch: int, model: nn.Module,
                  optimizer, scheduler, best_psnr: float,
                  ema=None):
    """Write a checkpoint to an explicit path with no rotation."""
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".pth.tmp")
    os.close(fd)

    saved_state = (
        {k: v.to(next(model.parameters()).device) for k, v in ema.shadow.items()}
        if ema is not None else model.state_dict()
    )

    try:
        torch.save({
            "epoch": epoch,
            "model": saved_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_psnr": best_psnr,
        }, tmp)
        time.sleep(0.1)
        _safe_delete(path)
        shutil.move(tmp, str(path))
    except Exception as exc:
        _safe_delete(Path(tmp))
        print(f"  [WARNING] Failed to save best checkpoint: {exc}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None,
                    ema=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and ckpt.get("ema"):
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


class CSVLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fields = None

    def log(self, row: dict):
        if self._fields is None:
            self._fields = list(row.keys())
            if not self.path.exists() or self.path.stat().st_size == 0:
                with open(self.path, "w", newline="") as f:
                    csv.DictWriter(f, self._fields).writeheader()

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self._fields).writerow(row)
