# Avito small watermark: reverse engineering

Scripts that measure the small Avito mark (the "●Avito" logo in the bottom-right
corner) from the photos the parser stores, and test how accurate the measured
model is. Outputs go to `out/` (gitignored). The template is the artwork of the
large mark, `data_gen/watermark.png` (8001×2032 alpha): the small mark is the
same logo.

## Findings (2026-10-24, 25,968 stored Avito photos)

- **Input the service gets:** JPEG, libjpeg q85 (60%) / q80 (40%), always 4:2:0.
  Sizes: 1280×960 (27%), 720×960 (27%), 1280×853, 1280×720, 540×960, ...
- **Position and size are fixed in pixels.** The mark is stamped after the photo
  is resized, at the same offset from the bottom-right corner on every image size
  (432×960 to 1280×960): the artwork at **94.04 px wide**, its top-left
  **(W − 106.03, H − 37.78)**. Per-photo localisation spread over 25.5k photos:
  ±0.05 px in x, −0.2..+0.07 px in y (p1..p99). No jitter, no scaling, so no
  localiser is needed: the frame is the same pixels of every photo.
- **Present in every photo** (NCC ≥ 0.2 in all 25,949 readable ones, ≥ 0.5 in 98.5%).
- **Layers** (fitted per pixel and channel, see `fit_model.py`):
  - logo: white-ish `c = (255.5, 249.7, 246.0)`, **opacity 0.514** (blue ×1.026),
    edge blur σ 0.34 px — crisp;
  - black drop shadow: opacity 0.33, offset **0.93 px down**, blur σ 1.12 px,
    **knocked out under the logo** (inside the glyphs the mark is a plain white
    overlay): alpha ≈ 0.11 / 0.07 / 0.04 / 0.01 at 0.5 / 1 / 1.5 / 2.5 px outside the edge;
  - a background-independent darkening around it (does not scale with the
    background, like the Cian logo's shadow): ≈ 3.9 / 2.2 / 2.0 levels (R/G/B)
    at the edge, 1.7 / 1.0 / 0.9 at 2–3 px, gone by ~6 px.
  Blend: `out = (b·(1 − s) − d)·(1 − α) + α·c`, i.e. per pixel
  `out − b = A − Alpha·b` with `Alpha = 1 − (1 − s)(1 − α)`, `A = α·c − d·(1 − α)`.
  This is the same structure as the large mark's model in `data_gen` (white
  logo + offset blurred black shadow), at a much higher opacity (0.51 vs ~0.10).
- **Raster detail:** a free-form per-pixel correction on top of the parametric
  model (`freeform.py`) carries over to JPEG grid phases it was not fitted on,
  so part of the remaining difference is the real mark's own raster, not JPEG.
- **Double marks:** ~0.4% of photos carry a second mark (re-uploaded photos,
  sometimes visibly offset). Saturated colours (channels at 255) clip the overlay.

## Accuracy

1. **Generative model** (what training needs): the model stamped on mark-free
   regions of the same photos, JPEG-encoded at the real grid phase and quality,
   estimated exactly like the real marks (`estimate.py --mode synth`), against
   the real estimate. RMS over the mark's support, levels on a background of
   40 / 128 / 220 (the mark itself is 52 / 30 / 13 there):

   | model | dominant grid phase (15k photos) | phases not used for fitting |
   |---|---|---|
   | guessed mark of the first `avito_small_v1` run (104×26 px, 11%) | 46 / 28 / 13 | — |
   | single blurred shadow | 7–10 | — |
   | layered, parametric (`model`) | 2.8 / 1.9 / 2.6 | 3.8 / 2.0 / 3.6 |
   | + free-form correction (`ff`) | 1.7 / 0.9 / 1.7 | 3.3 / 1.25 / 3.1 |

   Estimator noise (split halves): 0.1–0.2 levels on the dominant phase, 0.3–1.4
   on the small held-out phase groups.

2. **Inverse on held-out photos** (2,607 photos never used for any fit,
   `accuracy.py`): each frame inverted with the per-pixel map, then the leftover
   mark measured as a least-squares amplitude of the mark's pattern, relative to
   the input (input = 100%):

   | inverse map | leftover median | p90 | p99 | > 17% |
   |---|---|---|---|---|
   | none | 100% | | | |
   | guessed mark (first run) | 104% | | | |
   | layered parametric model | 2.4% | 6.4% | 15% | 0.6% |
   | mean real effect after JPEG (`realest`) | 1.8% | 6.0% | 14% | 0.7% |
   | noise floor (same measure, mark-free area) | 0.3% | 2.7% | | |

   By background: 1.5% median on dark frames, 2.4% mid, 4–6% on bright ones.
   The > 17% photos are double marks (~+60%) and saturated colours (~−24%).
   What an exact inverse leaves on typical photos is JPEG: a faint ringing
   outline with 4:2:0 colour fringes at ×5 contrast. That is the removal
   model's job; the pre-JPEG mark (`model` / `ff`) is what to stamp in training.

## Order

1. `survey.py`: index the Avito photos (`photo-lab/manifest_new.jsonl` +
   `photo-lab/images`), mean bottom-right corners per size.
2. `codec.py`: JPEG quality and subsampling.
3. `fit_seed.py`: artwork width/position on the mean corner.
4. `locate.py`: per-photo position, presence and scale.
5. `estimate.py --mode real [--phase H%16,W%16]`: per-pixel `Alpha`, `A` (IV fit).
6. `fit_model.py`, then iterate `estimate.py --mode synth` + `fit_model.py --bias ...`
   (JPEG bias removed from the target): the layered parametric model.
7. `freeform.py`: phase-averaged free-form correction (`out/ff_*.npy`).
8. `compare.py`, `profile.py`: real vs synthetic per phase / by edge distance.
9. `accuracy.py`: held-out inverse test.
