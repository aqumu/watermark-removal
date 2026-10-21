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
  brightness-dependent error). Plus the logo shadow below.
- **Per-photo variation:** strength ±20%, edge sharpness (mostly softer than
  the template), **size 97–103%** (regular-style photos often ~98%), and the
  text line's placement relative to the logo by a few px. All consistent with
  the mark being stamped before the photo is resized.
- **Shadow around the logo, bolder text (2026-10-21, `fit_halo.py --profile`):**
  real marks compared with synthetic ones (the template stamped on clean parts
  of the same photos, JPEG, same localisation and measurement), separately for
  the logo and the text line, per distance to the glyph edge. Fitting
  `out - bg = a - alpha * bg` with the crop's overall level as instrument:
  - logo: a darkening that does not depend on the background (no extra
    opacity): ~3.1 levels just outside the edge, 2.2 / 1.4 / 1.2 at 1–3 px,
    ~0.9 at 4–6 px, ~0.4 out to 12 px; green ~15% stronger than red. The
    outline on pure-white graphics is the same thing. Both halves of the
    listings agree to 0.1 level; an injected black halo is recovered.
  - logo opacity ~8% below the stacked template, edge a little crisper.
  - text line bolder than the font rendering: ~8% more opacity plus a faint
    outer glow (white) up to ~3 px.
  - per character the text differs too (`--align text --style S --id-len 9`
    stacks the constant "ID " prefix pixel-exactly): relative to the digits
    the "I" is ~20% (bold) / ~35% (regular) fainter, the "D" ~10%; no
    position offset (< 0.5 px), no extra blur. The over-drawn "I" was the
    tick left at "ID" on real outputs, worst on regular-style marks.
  Neither was in the training data; they match the outline ghosts around
  "циан" and the tick at "ID" on real outputs. Fitted model in mark.json
  ("shadow", "edge"; `out/mark_model.json` for export_assets.py): synthetic
  marks then match real ones within ~0.3–0.5 levels (logo) and ~0.7–1 (text),
  from 1.2–1.8 before. A plain stack or whole-mark profile hides this: the
  background estimate next to a bright mark is biased (control shows a fake
  ring), and logo (darker) and text (brighter) average out.
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
9. `fit_halo.py`: opacity and colour of the mark across backgrounds
   (instrumental-variable fit; synthetic control, halo injection test);
   `--profile` per logo/text and edge distance, with candidate mark models
   (`--shadow`, `--text-gain`, `--text-glow`, `--logo-gain`, `--logo-rim`);
   `--align text` stacks on the text line (per style / ID length).
10. `edge_profile.py`: edge profile on flat backgrounds, real vs synthetic.
