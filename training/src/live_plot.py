"""
live_plot.py — Real-time training visualisation
------------------------------------------------
Three side-by-side panels updated during training:

  Left   — EMA-smoothed total training loss (left axis) +
            validation PSNR full-image and masked-region (right axis)

  Centre — EMA-smoothed weighted contribution (λ × raw_loss) of each
            individual loss term

  Right  — Three stacked images from a fixed sample, updated each epoch:
              watermarked input  |  model output  |  weighted loss heatmap

Threading model
---------------
Training runs on a background thread.  The main thread owns the matplotlib
event loop (plt.show(block=True)).  Communication is via a thread-safe
queue: training threads call log_step / log_val / log_images which put
messages on the queue; a matplotlib timer fires every 200 ms, drains the
queue on the main thread, and redraws if anything changed.

This keeps the window fully interactive at all times — the GUI event loop
is never starved by training compute.

Usage (train.py):
    plotter = LivePlotter(log_dir, total_epochs, loss_cfg)
    thread  = threading.Thread(target=_train_worker, args=(trainer, plotter))
    thread.start()
    plotter.run_event_loop()   # blocks main thread; returns when done/closed
    thread.join()
    plotter.save()
"""

from __future__ import annotations
import queue
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


# ── EMA ───────────────────────────────────────────────────────────────────────

_EMA_ALPHA = 0.05   # ≈ 14-call half-life; at log_every=10 ≈ 2 epochs of smoothing


def _ema_update(state: dict, key: str, value: float) -> float:
    prev = state.get(key)
    new  = value if prev is None else _EMA_ALPHA * value + (1.0 - _EMA_ALPHA) * prev
    state[key] = new
    return new


# ── colours ───────────────────────────────────────────────────────────────────

_COMP_COLORS = {
    "l1_masked":          "#2196F3",   # blue
    "perceptual":         "#9C27B0",   # purple
    "saturation":         "#E91E63",   # pink
    "color_moment":       "#00BCD4",   # cyan
    "border":             "#F44336",   # red
    "interior_tv":        "#4CAF50",   # green
    "bg_tv":              "#795548",   # brown
    "bg_delta":           "#607D8B",   # blue-grey
}


# ── plotter ───────────────────────────────────────────────────────────────────

class LivePlotter:
    """
    Thread-safe real-time training plot.

    log_* methods are safe to call from any thread — they only put messages
    on a queue.  All matplotlib work happens on the main thread via a timer.
    """

    # Timer interval in milliseconds — how often the queue is drained and
    # the figure redrawn.  200 ms gives 5 fps, plenty for training feedback.
    _POLL_MS = 200

    def __init__(self, log_dir: str | Path, total_epochs: int, loss_cfg: dict, 
                 log_every: int = 10):
        self._log_dir      = Path(log_dir)
        self._total_epochs = total_epochs
        self._log_every    = log_every
        self._loss_cfg     = loss_cfg
        self._save_path    = self._log_dir / "training_curves.png"

        # Thread-safe message queue — training threads write, main thread reads
        self._queue: queue.SimpleQueue = queue.SimpleQueue()

        # EMA running state (main-thread only)
        self._ema_state: dict[str, float] = {}

        # History buffers (main-thread only)
        self._gs:               list[int]   = []
        self._total_ema:        list[float] = []
        self._comp_ema:         dict[str, list[float]] = {k: [] for k in _COMP_COLORS}

        self._val_gs:           list[int]   = []
        self._val_psnr:         list[float] = []
        self._val_psnr_masked:  list[float] = []

        # Set to True once the figure exists (even if the window was later closed)
        self._has_figure = False
        # Set to True while the window is open and receiving updates
        self._active     = False

        self._setup()

    # ── setup (main thread) ───────────────────────────────────────────────────


    def _setup(self) -> None:
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
            self._plt = plt
        except ImportError:
            print("[LivePlotter] matplotlib not available — live plot disabled")
            return

        try:
            plt.style.use("dark_background")
            plt.ion()

            self.fig = plt.figure(figsize=(16, 7))
            self.fig.suptitle("Training Progress", fontsize=11, fontweight="bold", y=0.99)
            gs = gridspec.GridSpec(
                3, 3, figure=self.fig,
                width_ratios=[2.8, 2.8, 2.2],
                wspace=0.32, hspace=0.08,
            )
            self.fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.08)

            # ── left: loss + PSNR ─────────────────────────────────────────
            self._ax_loss = self.fig.add_subplot(gs[:, 0])
            self._ax_psnr = self._ax_loss.twinx()
            self._ax_loss.set_ylabel("Total Loss (EMA)")
            self._ax_loss.set_xlabel("Training Step")
            self._ax_loss.set_title("Loss & Validation PSNR", pad=4)
            self._ax_psnr.set_ylabel("PSNR  (dB)", color="#ef9a9a")
            self._ax_psnr.tick_params(axis="y", labelcolor="#ef9a9a")

            # ── centre: loss components ────────────────────────────────────
            self._ax_comp = self.fig.add_subplot(gs[:, 1])
            self._ax_comp.set_ylabel("Weighted Contribution  (λ × loss, EMA)")
            self._ax_comp.set_xlabel("Training Step")
            self._ax_comp.set_title("Individual Loss Terms", pad=4)

            # ── right: three stacked sample images ─────────────────────────
            _labels    = ["watermarked", "model output", "weighted loss"]
            self._ax_imgs = [self.fig.add_subplot(gs[r, 2]) for r in range(3)]
            _blank_rgb  = np.zeros((4, 4, 3), dtype=np.uint8)
            _blank_gray = np.zeros((4, 4),    dtype=np.float32)
            self._im_wm    = self._ax_imgs[0].imshow(_blank_rgb)
            self._im_pred  = self._ax_imgs[1].imshow(_blank_rgb)
            self._im_wloss = self._ax_imgs[2].imshow(_blank_gray, cmap="jet",
                                                      vmin=0, vmax=1)
            for ax, lbl in zip(self._ax_imgs, _labels):
                ax.set_title(lbl, fontsize=8, pad=2)
                ax.axis("off")

            # ── line objects — created once, updated in-place ──────────────
            self._line_total, = self._ax_loss.plot(
                [], [], color="#b0bec5", lw=1.6, label="total loss",
                antialiased=True, solid_capstyle="round", solid_joinstyle="round")

            self._line_psnr, = self._ax_psnr.plot(
                [], [], color="#ef9a9a", lw=1.6,
                marker="o", markersize=3, label="PSNR",
                antialiased=True, solid_capstyle="round", solid_joinstyle="round")

            self._line_psnr_m, = self._ax_psnr.plot(
                [], [], color="#ffcdd2", lw=1.4, linestyle="--",
                marker="o", markersize=3, label="PSNR (masked)",
                antialiased=True, solid_capstyle="round", solid_joinstyle="round")

            self._comp_lines: dict = {}
            for comp, color in _COMP_COLORS.items():
                line, = self._ax_comp.plot(
                    [], [], color=color, lw=1.4, label=comp, alpha=0.8,
                    antialiased=True, solid_capstyle="round", solid_joinstyle="round")
                self._comp_lines[comp] = line

            self._ax_loss.legend(loc="upper left",  fontsize=8)
            self._ax_psnr.legend(loc="upper right", fontsize=8)
            self._ax_comp.legend(loc="upper right", fontsize=8, ncol=2)

            # ── timer: polls the queue from the main thread ────────────────
            self._timer = self.fig.canvas.new_timer(interval=self._POLL_MS)
            self._timer.add_callback(self._poll_queue)
            self._timer.start()

            # ── window close handler ───────────────────────────────────────
            self.fig.canvas.mpl_connect("close_event", self._on_close)

            self._has_figure = True
            self._active     = True

        except Exception as exc:
            print(f"[LivePlotter] could not open window ({exc}) — live plot disabled")

    # ── public API (thread-safe — safe to call from training thread) ──────────

    def log_step(self, global_step: int, breakdown: dict) -> None:
        if not self._active:
            return
        self._queue.put(("step", global_step, breakdown))

    def log_val(self, global_step: int, psnr: float, psnr_masked: float) -> None:
        if not self._active:
            return
        self._queue.put(("val", global_step, psnr, psnr_masked))

    def log_images(self, wm_bgr, pred_bgr, wloss_f32) -> None:
        """
        wm_bgr    : HxWx3 uint8 BGR  — watermarked input  (debug_0)
        pred_bgr  : HxWx3 uint8 BGR  — raw model output   (debug_3)
        wloss_f32 : HxW   float32    — weighted loss map   (debug_6)
        """
        if not self._active:
            return
        self._queue.put(("images", wm_bgr, pred_bgr, wloss_f32))

    def run_event_loop(self) -> None:
        """Block the calling (main) thread running the GUI event loop.
        Returns when training finishes (stop signal) or the window is closed.
        """
        if not self._has_figure:
            self._setup()
        
        if not self._active:
            return
        self._plt.show(block=True)

    def save(self) -> None:
        """Write the figure to disk.  Safe to call after run_event_loop returns."""
        if not self._has_figure:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self._save_path, dpi=120, bbox_inches="tight")
            print(f"[LivePlotter] saved → {self._save_path}")
        except Exception as exc:
            print(f"[LivePlotter] could not save figure: {exc}")

    # ── main-thread internals (called only from timer / close handler) ────────

    def _poll_queue(self) -> None:
        """Drain the queue and redraw if anything changed.  Runs on main thread."""
        updated = False
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "step":
                    self._apply_step(item[1], item[2])
                    updated = True
                elif kind == "val":
                    self._apply_val(item[1], item[2], item[3])
                    updated = True
                elif kind == "images":
                    self._apply_images(item[1], item[2], item[3])
                    updated = True
                elif kind == "stop":
                    # Training finished — close the window
                    self._active = False
                    try:
                        self._plt.close(self.fig)
                    except Exception:
                        pass
                    return
        except queue.Empty:
            pass
        if updated:
            self._redraw()

    def _apply_step(self, global_step: int, breakdown: dict) -> None:
        self._gs.append(global_step)
        self._total_ema.append(
            _ema_update(self._ema_state, "_total", breakdown["total"])
        )
        lc = self._loss_cfg
        for comp in _COMP_COLORS:
            raw = breakdown.get(comp, 0.0)
            if raw == 0.0:
                # If loss was skipped, don't update EMA with 0.0, just keep previous
                val = self._ema_state.get(comp, 0.0)
                self._comp_ema[comp].append(val)
                continue

            w = breakdown.get(f"weight_{comp}", lc.get(comp, 0.0))
            if isinstance(w, (list, tuple)):
                w = float(w[0])
            self._comp_ema[comp].append(
                _ema_update(self._ema_state, comp, w * raw)
            )

    def _apply_val(self, global_step: int, psnr: float, psnr_masked: float) -> None:
        self._val_gs.append(global_step)
        self._val_psnr.append(psnr)
        self._val_psnr_masked.append(psnr_masked)

    def _apply_images(self, wm_bgr, pred_bgr, wloss_f32) -> None:
        import cv2
        self._im_wm.set_data(cv2.cvtColor(wm_bgr,   cv2.COLOR_BGR2RGB))
        self._im_pred.set_data(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB))
        self._im_wloss.set_data(wloss_f32)
        self._im_wloss.set_clim(0, max(float(wloss_f32.max()), 1e-6))
        for ax in self._ax_imgs:
            ax.relim()
            ax.autoscale_view()

    def _redraw(self) -> None:
        gs = self._gs
        
        sx, sy = _smooth_line(gs, self._total_ema)
        self._line_total.set_xdata(sx)
        self._line_total.set_ydata(sy)

        if self._val_gs:
            self._line_psnr.set_xdata(self._val_gs)
            self._line_psnr.set_ydata(self._val_psnr)
            
            self._line_psnr_m.set_xdata(self._val_gs)
            self._line_psnr_m.set_ydata(self._val_psnr_masked)

        for comp, line in self._comp_lines.items():
            sx, sy = _smooth_line(gs, self._comp_ema[comp])
            line.set_xdata(sx)
            line.set_ydata(sy)

        for ax in (self._ax_loss, self._ax_psnr, self._ax_comp):
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()

    def _on_close(self, event) -> None:
        """User closed the window — disable the plotter, stop the timer."""
        self._active = False
        try:
            self._timer.stop()
        except Exception:
            pass


# ── seg plotter ───────────────────────────────────────────────────────────────

_SEG_COMP_COLORS = {
    "bce":   "#2196F3",   # blue
    "focal": "#FF9800",   # amber
    "l1":    "#9C27B0",   # purple
    "ms":    "#4CAF50",   # green
    "dice":  "#F44336",   # red
}


class SegLivePlotter:
    """
    Real-time training plot for the segmentation model.

    Left panel  — EMA total loss (left axis) + validation IoU (right axis),
                  with focal and dice breakdown lines.
    Right panel — Three stacked images updated each epoch:
                    watermarked input | predicted mask | ground-truth mask

    Same threading model as LivePlotter: training on a background thread, GUI
    on the main thread, communication via a thread-safe queue.
    """

    _POLL_MS = 200

    def __init__(self, log_dir: str | Path, total_epochs: int):
        self._log_dir      = Path(log_dir)
        self._total_epochs = total_epochs
        self._save_path    = self._log_dir / "training_curves_seg.png"

        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._ema_state: dict[str, float] = {}

        self._gs:         list[int]   = []
        self._total_ema:  list[float] = []
        self._comp_ema:   dict[str, list[float]] = {k: [] for k in _SEG_COMP_COLORS}

        self._val_gs:  list[int]   = []
        self._val_iou: list[float] = []

        self._has_figure = False
        self._active     = False

    def _setup(self) -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
            self._plt = plt
        except ImportError:
            print("[SegLivePlotter] matplotlib not available — live plot disabled")
            return

        try:
            plt.style.use("dark_background")
            plt.ion()

            self.fig = plt.figure(figsize=(16, 7))
            self.fig.suptitle("Segmentation Training Progress",
                              fontsize=11, fontweight="bold", y=0.99)
            gs = gridspec.GridSpec(
                3, 2, figure=self.fig,
                width_ratios=[3.5, 2.0],
                wspace=0.30, hspace=0.08,
            )
            self.fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08)

            # ── left: loss + IoU ──────────────────────────────────────────
            self._ax_loss = self.fig.add_subplot(gs[:, 0])
            self._ax_iou  = self._ax_loss.twinx()
            self._ax_loss.set_ylabel("Loss (EMA)")
            self._ax_loss.set_xlabel("Training Step")
            self._ax_loss.set_title("Loss & Validation IoU", pad=4)
            self._ax_iou.set_ylabel("IoU", color="#a5d6a7")
            self._ax_iou.tick_params(axis="y", labelcolor="#a5d6a7")

            # ── right: three stacked sample images ────────────────────────
            _labels = ["watermarked", "predicted mask", "ground truth"]
            self._ax_imgs = [self.fig.add_subplot(gs[r, 1]) for r in range(3)]
            _blank_rgb  = np.zeros((4, 4, 3), dtype=np.uint8)
            _blank_gray = np.zeros((4, 4),    dtype=np.float32)
            self._im_wm   = self._ax_imgs[0].imshow(_blank_rgb)
            self._im_pred = self._ax_imgs[1].imshow(_blank_gray, cmap="hot", vmin=0, vmax=1)
            self._im_gt   = self._ax_imgs[2].imshow(_blank_gray, cmap="hot", vmin=0, vmax=1)
            for ax, lbl in zip(self._ax_imgs, _labels):
                ax.set_title(lbl, fontsize=8, pad=2)
                ax.axis("off")

            # ── line objects ──────────────────────────────────────────────
            self._line_total, = self._ax_loss.plot(
                [], [], color="#b0bec5", lw=1.6, label="total loss",
                antialiased=True, solid_capstyle="round", solid_joinstyle="round")

            self._comp_lines: dict = {}
            for comp, color in _SEG_COMP_COLORS.items():
                line, = self._ax_loss.plot(
                    [], [], color=color, lw=1.2, linestyle="--", label=comp, alpha=0.7,
                    antialiased=True, solid_capstyle="round", solid_joinstyle="round")
                self._comp_lines[comp] = line

            self._line_iou, = self._ax_iou.plot(
                [], [], color="#a5d6a7", lw=1.6,
                marker="o", markersize=3, label="IoU",
                antialiased=True, solid_capstyle="round", solid_joinstyle="round")

            self._ax_loss.legend(loc="upper left",  fontsize=8)
            self._ax_iou. legend( loc="upper right", fontsize=8)

            self._timer = self.fig.canvas.new_timer(interval=self._POLL_MS)
            self._timer.add_callback(self._poll_queue)
            self._timer.start()
            self.fig.canvas.mpl_connect("close_event", self._on_close)

            self._has_figure = True
            self._active     = True

        except Exception as exc:
            print(f"[SegLivePlotter] could not open window ({exc}) — live plot disabled")

    # ── public API ────────────────────────────────────────────────────────────

    def log_step(self, global_step: int, breakdown: dict) -> None:
        if not self._active:
            return
        self._queue.put(("step", global_step, breakdown))

    def log_val(self, global_step: int, iou_score: float) -> None:
        if not self._active:
            return
        self._queue.put(("val", global_step, iou_score))

    def log_images(self, wm_bgr, pred_mask_f32, gt_mask_f32) -> None:
        """
        wm_bgr       : HxWx3 uint8 BGR  — watermarked input
        pred_mask_f32: HxW   float32    — predicted mask in [0, 1]
        gt_mask_f32  : HxW   float32    — ground-truth mask in [0, 1]
        """
        if not self._active:
            return
        self._queue.put(("images", wm_bgr, pred_mask_f32, gt_mask_f32))

    def run_event_loop(self) -> None:
        if not self._has_figure:
            self._setup()
        
        if not self._active:
            return
        self._plt.show(block=True)

    def save(self) -> None:
        if not self._has_figure:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self._save_path, dpi=120, bbox_inches="tight")
            print(f"[SegLivePlotter] saved → {self._save_path}")
        except Exception as exc:
            print(f"[SegLivePlotter] could not save figure: {exc}")

    # ── main-thread internals ─────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        updated = False
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "step":
                    self._apply_step(item[1], item[2])
                    updated = True
                elif kind == "val":
                    self._apply_val(item[1], item[2])
                    updated = True
                elif kind == "images":
                    self._apply_images(item[1], item[2], item[3])
                    updated = True
                elif kind == "stop":
                    self._active = False
                    try:
                        self._plt.close(self.fig)
                    except Exception:
                        pass
                    return
        except queue.Empty:
            pass
        if updated:
            self._redraw()

    def _apply_step(self, global_step: int, breakdown: dict) -> None:
        self._gs.append(global_step)
        self._total_ema.append(
            _ema_update(self._ema_state, "_total", breakdown["total"])
        )
        for comp in _SEG_COMP_COLORS:
            raw = breakdown.get(comp, 0.0)
            if raw == 0.0:
                val = self._ema_state.get(comp, 0.0)
                self._comp_ema[comp].append(val)
                continue
                
            self._comp_ema[comp].append(
                _ema_update(self._ema_state, comp, raw)
            )

    def _apply_val(self, global_step: int, iou_score: float) -> None:
        self._val_gs.append(global_step)
        self._val_iou.append(iou_score)

    def _apply_images(self, wm_bgr, pred_mask_f32, gt_mask_f32) -> None:
        import cv2
        self._im_wm.set_data(cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2RGB))
        self._im_pred.set_data(pred_mask_f32)
        self._im_pred.set_clim(0, 1)
        self._im_gt.set_data(gt_mask_f32)
        for ax in self._ax_imgs:
            ax.relim()
            ax.autoscale_view()

    def _redraw(self) -> None:
        gs = self._gs
        
        sx, sy = _smooth_line(gs, self._total_ema)
        self._line_total.set_xdata(sx)
        self._line_total.set_ydata(sy)
        
        for comp, line in self._comp_lines.items():
            sx, sy = _smooth_line(gs, self._comp_ema[comp])
            line.set_xdata(sx)
            line.set_ydata(sy)
            
        if self._val_gs:
            self._line_iou.set_xdata(self._val_gs)
            self._line_iou.set_ydata(self._val_iou)
            
        for ax in (self._ax_loss, self._ax_iou):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def _on_close(self, event) -> None:
        self._active = False
        try:
            self._timer.stop()
        except Exception:
            pass


def _smooth_line(x, y, samples=200):
    """Cubic spline interpolation for a smoother aesthetic."""
    if len(x) < 4:
        return x, y
    try:
        # Cast to numpy just to be sure
        xa = np.array(x)
        ya = np.array(y)
        # Unique points only (spline fails on duplicates)
        _, idx = np.unique(xa, return_index=True)
        xa, ya = xa[idx], ya[idx]
        
        if len(xa) < 4:
            return x, y
            
        x_new = np.linspace(xa.min(), xa.max(), samples)
        spl = make_interp_spline(xa, ya, k=3)
        y_new = spl(x_new)
        return x_new, y_new
    except Exception:
        return x, y
