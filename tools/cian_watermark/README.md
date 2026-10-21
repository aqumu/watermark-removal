# Cian watermark: reverse engineering

Scripts that turned served Cian photos into `assets/marks/cian/` (the mark
profile used by training and inference). Outputs go to `out/` (gitignored).

## Findings (2026-10-20)

- **Input the service gets:** the parser stores Cian photos as JPEG, 91% libjpeg
  q80 and 8% q85, 4:2:0, no larger than 1280×960. The CDN serves browsers a WebP
  transcode (q≈80) of the same master; that is not the service's input.
- **Mark:** logo raster (pin, roof, "циан") plus a text line `ID <listing id>`
  in DejaVu Sans, **bold 24.25 px (~80%)**, **regular 23.6 px (~19%)**, or no
  text line at all (~1%), centred under the logo. 8-digit IDs exist.
- **Blend:** close to a white overlay at ~11–13% opacity (`out = bg + m·(c − bg)`,
  c fitted 248–259 on 700 JPEGs; the first WebP fit, 234, left a
  brightness-dependent error). On near-white backgrounds real marks show a
  faint 1–2 level darkening that this model does not produce (accepted).
- **Per-photo variation:** strength ±20%, edge sharpness (mostly softer than
  the template), **size 97–103%** (regular-style photos often ~98%), and the
  text line's placement relative to the logo by a few px. All consistent with
  the mark being stamped before the photo is resized.
- **No dark halo on photos (2026-10-21, `fit_halo.py`, `edge_profile.py`):**
  real marks on flat backgrounds match halo-free synthetic marks outside the
  glyph edges within ±0.5 levels on dark and mid backgrounds. The method
  recovers an injected black halo of that size, so one would have shown. The
  "dark ring" a plain stack shows is a measurement artefact (it appears on
  synthetic marks too). Pure-white graphics (floor plans) do show a ~1.7 level
  dark outline 1–4 px out that JPEG and WebP do not produce; likely the mark
  stamped on a transparent PNG before it is flattened onto white. Graphics only.
- **Edge profile differs from the template:** real marks are brighter just
  inside the glyph edges (1–2 px in: about +1.2–2.4 levels on dark backgrounds,
  scaling with the mark's contrast) with no dark undershoot outside. Training's
  sharpness augmentation (`m + k·(m − blur(m))`) brightens the inner edge but
  also adds an undershoot real marks don't have, and its range was mostly on
  the soft side (k < 0) while real edges sit near k ≈ 0.5–1 inside.
- **Position:** sub-pixel placement (fractional offsets are uniform). Frame
  origin = `(W − (0.099·W + 247.2), H − (0.0985·H + 115.1))`, jitter ≈ ±12.7 px
  in x and ±6.7 px in y.
- **ID ≠ listing:** re-used photos can carry another listing's ID, sometimes
  next to the new listing's own mark (double marks). Rarer than first
  estimated: most "mismatches" were regular-style marks that the first
  classifier missed.

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
7. `refit_text.py`: text-line geometry and style shares on photos registered
   on the logo alone; `fit_scale.py`: per-photo size (noisy population-wide,
   clear on single photos).
8. `export_assets.py`: writes `assets/marks/cian/mark.{json,npz}`.
9. `fit_halo.py`: per-pixel opacity and colour of the mark across backgrounds
   (instrumental-variable fit; synthetic control and halo injection test).
10. `edge_profile.py`: model-free edge profile on flat backgrounds, real vs
   synthetic at several edge sharpness values.
