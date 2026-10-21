"""Step 9: is there a dark halo around the mark, and what are its spread, colour and intensity?

Result (2026-10-21): no halo on photos; see edge_profile.py and the README.

Per frame pixel and colour channel, a mark of any construction composites as

    out = b * (1 - alpha) + a          (alpha: total opacity, a: premultiplied colour)

so with b the background under the pixel, D = out - b = a - alpha * b. Over many
aligned real marks on different backgrounds this is a straight line in b, and a
per-pixel fit (see solve()) gives alpha and a; their ratio C = a / alpha
is the colour of whatever covers that pixel. A white layer alone (the current
template, overlay 248) gives C = 248 everywhere, also in resampling/JPEG ringing
lobes (alpha < 0 there, same C). A dark halo shows up as alpha > 0 with a low C
outside the glyph edges.

Modes:
  real   aligned real marks from the parser-stored photos
  synth  control: the current template (no halo) stamped on clean regions of the
         same photos, JPEG q80, then the same localise/align/fit
  inject like synth, plus a known black halo (--beta, --sigma) to check that the
         fit recovers it

Writes out/halo_<mode>.npz (per-pixel alpha, a, standard errors, the two halves
fitted separately) and prints the edge-distance profile.

Usage:
  python fit_halo.py --mode real --n 8000
  python fit_halo.py --mode synth --n 8000
  python fit_halo.py --mode inject --beta 0.01 --sigma 2.0 --n 8000
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from wm_shared.marks.localize import localize  # noqa: E402
from wm_shared.marks.profile import load_mark  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
PAD = 32                 # crop margin around the frame
HALO_REACH = 18          # px beyond the glyphs kept out of the background estimate
EDGE_LEVEL = 0.06        # template opacity taken as the glyph edge
MIN_SCORE = 0.35

_S: dict = {}


def glyph_mask(m: np.ndarray) -> np.ndarray:
    return (m > EDGE_LEVEL).astype(np.uint8)


def halo_map(mask: np.ndarray, beta: float, sigma: float) -> np.ndarray:
    """Synthetic black halo for the injection test: blurred glyph mask."""
    return beta * cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) if beta > 0 else np.zeros(mask.shape, np.float32)


def _init(mode: str, beta: float, sigma: float, sharp: float = 0.0) -> None:
    cv2.setNumThreads(1)
    p = load_mark("cian")
    fw, fh = p.frame_size
    logo = np.zeros((fh + 2 * PAD, fw + 2 * PAD), np.float32)
    logo[PAD:PAD + fh, PAD:PAD + fw] = p.logo
    _S.update(p=p, mode=mode, beta=beta, sigma=sigma, sharp=sharp, logo=logo)


def _stamp(rgb: np.ndarray, lid: str, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]] | None:
    """synth/inject: a stamped, JPEG-compressed window built from a clean region of the photo."""
    p = _S["p"]
    ww, wh = p.window
    h, w = rgb.shape[:2]
    if h < 2 * wh or w < ww + 8:
        return None
    # bottom-left (floors, like the mark's own corner) when the photo is wide enough, else top-left
    clean = (rgb[h - wh - 8:h - 8, :ww] if w >= 2 * ww + 40 else rgb[:wh, :ww]).copy()
    style = "bold" if rng.random() < 0.8 else "regular"
    x, y = 40 + rng.uniform(-8, 8), 40 + rng.uniform(-5, 5)
    scale = rng.uniform(0.98, 1.02)
    m = p.render(lid, x, y, size=(ww, wh), style=style, scale=scale)
    if _S["sharp"]:
        m = m + _S["sharp"] * (m - cv2.GaussianBlur(m, (0, 0), 1.0))   # as in training (data.py)
    out = clean
    if _S["mode"] == "inject":
        hl = halo_map(glyph_mask(m), _S["beta"], _S["sigma"])
        out = out * (1 - hl[..., None])
    out = out + m[..., None] * (p.overlay - out)
    ok, enc = cv2.imencode(".jpg", np.clip(np.rint(out), 0, 255).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    win = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    return win, (int(round(x - 40)), int(round(y - 40)))


def _crop(row: dict) -> tuple | None:
    """Frame-aligned RGB crop of the mark, background estimate, per-pixel weights, overall
    background level, the aligned template and the text style."""
    p = _S["p"]
    bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = bgr[..., ::-1].astype(np.float32)
    ww, wh = p.window
    lid = row["listing_id"]
    if _S["mode"] == "real":
        h, w = rgb.shape[:2]
        x0, y0 = p.window_origin(w, h)
        win = rgb[y0:y0 + wh, x0:x0 + ww]
        if win.shape[:2] != (wh, ww):
            return None
        ex, ey = p.expected_origin(w, h)
        prior = (ex - x0, ey - y0)
    else:
        made = _stamp(rgb, lid, random.Random(zlib.crc32(row["path"].encode())))
        if made is None:
            return None
        win, _ = made
        prior = (40.0, 40.0)
    try:
        x, y, score, style, offset, scale = localize(win, p, lid, prior=prior)
    except cv2.error:
        return None
    if abs(score) < MIN_SCORE or style == "none":
        return None
    fw, fh = p.frame_size
    cx, cy = fw / 2, fh / 2
    mat = np.array([[scale, 0, -PAD * scale + cx * (1 - scale) + x],
                    [0, scale, -PAD * scale + cy * (1 - scale) + y]], np.float32)
    size = (fw + 2 * PAD, fh + 2 * PAD)
    crop = cv2.warpAffine(win, mat, size, flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    m = p.render(lid, PAD, PAD, size=size, style=style, text_offset=offset)
    support = cv2.dilate(glyph_mask(np.maximum(m, _S["logo"])) | (m > 0.003).astype(np.uint8),
                         np.ones((2 * HALO_REACH + 1, 2 * HALO_REACH + 1), np.uint8))
    keep = (1 - support).astype(np.float32)
    if keep.mean() < 0.2:
        return None
    b = np.stack([cv2.inpaint(crop[..., k], support, 5, cv2.INPAINT_TELEA) for k in range(3)], -1)
    b = np.where(support[..., None] > 0, cv2.GaussianBlur(b, (0, 0), 2.0), b)
    # flatness: residual variance of the surroundings, spread into the support
    g = crop.mean(-1)
    r2 = (g - cv2.GaussianBlur(g, (0, 0), 3.0)) ** 2 * keep
    var = cv2.GaussianBlur(r2, (0, 0), 12) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 12), 1e-3)
    wgt = 1.0 / (var + 1.0)
    # backgrounds near the clipping limits carry clipped JPEG ringing, not the blend
    valid = (b > 6) & (b < 249) & (crop > 1) & (crop < 254)
    z = (crop * keep[..., None]).sum((0, 1)) / keep.sum()          # the crop's overall background level
    return crop, b, np.where(valid, wgt[..., None], 0).astype(np.float32), z, m, style


def _work(rows: list[dict]):
    sums = None
    n = 0
    for row in rows:
        got = _crop(row)
        if got is None:
            continue
        crop, b, w, z, _, _ = got
        d = crop - b
        half = zlib.crc32(row["listing_id"].encode()) & 1
        if sums is None:
            sums = np.zeros((2, 6) + crop.shape, np.float64)
        s = sums[half]
        s[0] += w
        s[1] += w * b
        s[2] += w * d
        s[3] += w * z
        s[4] += w * z * b
        s[5] += w * z * d
        n += 1
    return n, sums


def solve(s: np.ndarray):
    """Per pixel/channel fit of D = a - alpha * b, with the crop's overall background level z
    as the instrument for b.

    Plain least squares on the estimated background is biased: where the estimate
    is noisy (deep inside the inpainted support) its errors regress towards the
    mean and fake a positive alpha whose colour is the average background. The
    instrument follows b but not the local estimation error, so
    alpha = -cov(z, D) / cov(z, b) is free of that bias.
    """
    sw, sb, sd, sz, szb, szd = s
    sw = np.maximum(sw, 1e-9)
    mb, md, mz = sb / sw, sd / sw, sz / sw
    czb = szb / sw - mz * mb
    czd = szd / sw - mz * md
    slope = czd / np.where(np.abs(czb) > 1e-6, czb, 1e-6)
    return -slope, md - slope * mb, czb


def edge_profile(alpha: np.ndarray, a: np.ndarray, logo: np.ndarray, rows: slice) -> list[tuple]:
    """Mean alpha, a and C = a/alpha by signed distance to the logo's glyph edge (logo rows only)."""
    inside = glyph_mask(logo)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    dist = np.where(inside > 0, -d_in + 0.5, d_out - 0.5)
    sel = np.zeros(logo.shape, bool)
    sel[rows] = True
    out = []
    for lo in range(-6, 20):
        s = sel & (dist >= lo) & (dist < lo + 1)
        if s.sum() < 20:
            continue
        al = alpha[s].mean(0)
        aa = a[s].mean(0)
        out.append((lo, int(s.sum()), al, aa))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "synth", "inject"), default="real")
    ap.add_argument("--n", type=int, default=8000, help="photos to try")
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--index", default=str(REPO / "data_gen/backgrounds/cian_index.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))
    random.Random(17).shuffle(rows)
    rows = rows[:args.n]
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    total, sums = 0, None
    with Pool(args.workers, initializer=_init, initargs=(args.mode, args.beta, args.sigma)) as pool:
        for n, s in pool.imap_unordered(_work, chunks):
            total += n
            if s is not None:
                sums = s if sums is None else sums + s
    print(f"{args.mode}: {total} aligned marks")

    p = load_mark("cian")
    fw, fh = p.frame_size
    logo = np.zeros((fh + 2 * PAD, fw + 2 * PAD), np.float32)
    logo[PAD:PAD + fh, PAD:PAD + fw] = p.logo
    logo_rows = slice(0, PAD + int(np.nonzero(p.logo.max(1) > 0.02)[0].max()) + 1)

    both = solve(sums.sum(0))
    halves = [solve(sums[k]) for k in range(2)]
    OUT_DIR.mkdir(exist_ok=True)
    tag = args.mode if args.mode != "inject" else f"inject_b{args.beta}_s{args.sigma}"
    np.savez_compressed(OUT_DIR / f"halo_{tag}.npz", alpha=both[0], a=both[1], czb=both[2],
                        alpha_h0=halves[0][0], a_h0=halves[0][1], alpha_h1=halves[1][0], a_h1=halves[1][1],
                        n=total, logo=logo)

    print("signed distance to glyph edge (px, <0 inside) | alpha R G B (x1000) | a R G B | C = a/alpha"
          " | alpha half0/half1 (luma, x1000) | injected halo (x1000)")
    prof = edge_profile(both[0], both[1], logo, logo_rows)
    inj = halo_map(glyph_mask(logo), args.beta, args.sigma)[..., None].repeat(3, -1)
    pin = {d: al for d, _, al, _ in edge_profile(inj, inj, logo, logo_rows)}
    p0 = {d: al for d, _, al, _ in edge_profile(halves[0][0], halves[0][1], logo, logo_rows)}
    p1 = {d: al for d, _, al, _ in edge_profile(halves[1][0], halves[1][1], logo, logo_rows)}
    for d, cnt, al, aa in prof:
        c = aa / np.where(np.abs(al) > 1e-4, al, np.nan)
        print(f"{d:+3d} n={cnt:5d} | " + " ".join(f"{v * 1000:+6.2f}" for v in al) + " | "
              + " ".join(f"{v:+6.3f}" for v in aa) + " | " + " ".join(f"{v:6.0f}" for v in c)
              + f" | {p0[d].mean() * 1000:+6.2f} {p1[d].mean() * 1000:+6.2f} | {pin[d].mean() * 1000:5.2f}")


if __name__ == "__main__":
    main()
