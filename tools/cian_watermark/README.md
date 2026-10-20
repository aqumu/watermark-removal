# Cian watermark: reverse engineering

Scripts that turned served Cian photos into `assets/marks/cian/` (the mark
profile used by training and inference). Outputs go to `out/` (gitignored).

## Findings (2026-10-20)

- **Input the service gets:** the parser stores Cian photos as JPEG, 91% libjpeg
  q80 and 8% q85, 4:2:0, no larger than 1280×960. The CDN serves browsers a WebP
  transcode (q≈80) of the same master; that is not the service's input.
- **Mark:** logo raster (pin, roof, "циан") plus a text line `ID <listing id>`
  in DejaVu Sans, either **bold 24.25 px (~82%)** or **regular 23.6 px (~18%)**,
  centred at frame x≈159. 8-digit IDs exist. Text and logo share one opacity
  (≈0.136 in solid strokes); overlay grey ≈234: `out = bg + m·(234 − bg)`.
- **Position:** fixed pixel size, sub-pixel placement (fractional offsets are
  uniform). Frame origin = `(W − (0.099·W + 247.2), H − (0.0985·H + 115.1))`,
  jitter ≈ ±12.7 px in x and ±6.7 px in y.
- **ID ≠ listing:** about 6% of photos with a visible logo don't fit their
  listing's ID better than a random ID (re-used photos carry the original
  listing's ID; some of these are just hidden text). About 79% verifiably
  match, and the rest are too faint to tell.

## Order

1. `collect.py`: index the parser's stored Cian photos with their listing IDs
   (parser dump + `photo-lab/images`). Copy `out/cian_index.csv` to
   `data_gen/backgrounds/` for training.
2. `locate.py`, `stack.py`, `relocate.py`, `refine.py`: first template from
   the 89 WebP samples in `deduper/dumps/cian_samples` (seed only).
3. `build_template.py`: locate + sub-pixel registration by model fit on
   ~3000 JPEGs, restacked into `out/m_jpeg.npy`.
4. `per_listing.py`: per-listing stacks, used for the text-line fits.
5. `fit_font.py`, `fit_style_photos.py`: font, size and position of each text
   style. `fit_unexplained.py` was a check for a third style; there isn't one.
6. `classify.py`: style frequencies and the ID-mismatch rate
   (`out/style_stats.json`).
7. `export_assets.py`: writes `assets/marks/cian/mark.{json,npz}`.
