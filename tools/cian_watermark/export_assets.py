"""Step 7: write assets/marks/cian/{mark.npz, mark.json} from the estimates.

- logo: the JPEG-stack estimate with the text line removed (rows >= TEXT_ROW) and
  the background noise floor cleared;
- text styles: size / centre / top fitted with the profile's own renderer against
  per-listing stacks (NCC over the text band, summed over listings);
- anchor, jitter and image sizes from the registered JPEG positions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common import *  # noqa: E402
from stack import FRAME_X0, FRAME_Y0  # noqa: E402
from wm_shared.marks.profile import TextStyle, text_coverage, text_ink_width  # noqa: E402

ASSET_DIR = REPO / "assets" / "marks" / "cian"
TEXT_ROW = 103
BAND = (96, 130, 40, 280)
STYLE_LISTINGS = {
    "bold": ["325371418", "333382852", "318479656", "312255687", "333450398", "325690988"],
    "regular": ["326545824", "333743056", "333159316", "333808851"],
}
INIT = {"bold": ("DejaVuSans-Bold.ttf", 24.25, 0.135), "regular": ("DejaVuSans.ttf", 23.625, 0.11)}


def clean_logo(m: np.ndarray) -> np.ndarray:
    logo = np.clip(m, 0, 1).copy()
    logo[TEXT_ROW:] = 0
    support = cv2.dilate((cv2.GaussianBlur(logo, (0, 0), 1.0) > 0.006).astype(np.uint8), np.ones((5, 5), np.uint8))
    return (logo * support).astype(np.float32)


def text_band(text: str, st: TextStyle, frame_shape) -> np.ndarray:
    canvas = np.zeros(frame_shape, np.float32)
    cov, ix, iy = text_coverage(text, st, st.center_x - text_ink_width(text, st) / 2, st.top)
    h, w = cov.shape
    canvas[iy:iy + h, ix:ix + w] += cov
    y0, y1, x0, x1 = BAND
    return canvas[y0:y1, x0:x1]


def score(stacks: dict, st: TextStyle) -> float:
    y0, y1, x0, x1 = BAND
    total = 0.0
    for lid, stack in stacks.items():
        t = text_band("ID " + lid, st, stack.shape).ravel()
        s = stack[y0:y1, x0:x1].ravel()
        total += float(np.corrcoef(t, s)[0, 1])
    return total / len(stacks)


def fit_style(name: str) -> dict:
    font, size, opacity = INIT[name]
    stacks = {lid: np.load(OUT_DIR / f"mj_{lid}.npy") for lid in STYLE_LISTINGS[name]}
    st = TextStyle(font=str(ASSET_DIR / "fonts" / font), size=size, opacity=opacity, center_x=159.2, top=103.5)
    best = score(stacks, st)
    for step in (0.5, 0.2, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for attr, d in (("center_x", step), ("top", step), ("size", step / 2)):
                for sign in (-1, 1):
                    trial = TextStyle(**{**st.__dict__, attr: getattr(st, attr) + sign * d})
                    s = score(stacks, trial)
                    if s > best + 1e-5:
                        st, best, improved = trial, s, True
    # opacity by least squares over all listings
    y0, y1, x0, x1 = BAND
    num = den = 0.0
    for lid, stack in stacks.items():
        t = text_band("ID " + lid, st, stack.shape)
        num += float((t * stack[y0:y1, x0:x1]).sum()); den += float((t * t).sum())
    st.opacity = num / den
    print(f"{name}: size={st.size:.3f} center_x={st.center_x:.2f} top={st.top:.2f} "
          f"opacity={st.opacity:.4f} mean ncc={best:.4f}")
    return {"font": f"fonts/{font}", "size": round(st.size, 3), "opacity": round(st.opacity, 4),
            "center_x": round(st.center_x, 2), "top": round(st.top, 2)}


def anchor_fit(positions: list[dict]) -> tuple[dict, dict]:
    good = [p for p in positions if p["gain"] >= 0.2]
    W = np.array([p["size"][0] for p in good], float); H = np.array([p["size"][1] for p in good], float)
    ox = np.array([p["x"] for p in good]) + FRAME_X0       # frame origin
    oy = np.array([p["y"] for p in good]) + FRAME_Y0
    rx, ry = W - ox, H - oy
    ax, bx = np.linalg.lstsq(np.stack([W, np.ones_like(W)], 1), rx, rcond=None)[0]
    ay, by = np.linalg.lstsq(np.stack([H, np.ones_like(H)], 1), ry, rcond=None)[0]
    jx = -(rx - (ax * W + bx)); jy = -(ry - (ay * H + by))  # origin offset from the expected origin
    print(f"anchor: right = {ax:.4f} W + {bx:.2f}, bottom = {ay:.4f} H + {by:.2f}  (n={len(good)})")
    print(f"jitter x: std {jx.std():.2f} pct0.5/99.5 {np.percentile(jx, [0.5, 99.5]).round(2)}; "
          f"y: std {jy.std():.2f} pct0.5/99.5 {np.percentile(jy, [0.5, 99.5]).round(2)}")
    jitter = {"x": [round(float(np.percentile(jx, 0.5)), 1), round(float(np.percentile(jx, 99.5)), 1)],
              "y": [round(float(np.percentile(jy, 0.5)), 1), round(float(np.percentile(jy, 99.5)), 1)]}
    return {"ax": round(float(ax), 5), "bx": round(float(bx), 2),
            "ay": round(float(ay), 5), "by": round(float(by), 2)}, jitter


if __name__ == "__main__":
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    # overlay grey and opacity fitted jointly on ~700 registered JPEGs (stack.estimate with a free
    # overlay level); c = 234 from the first WebP samples left a brightness-dependent mismatch
    OVERLAY = 248.0
    logo = clean_logo(np.load(OUT_DIR / "m_jpeg_cfree.npy"))
    np.savez_compressed(ASSET_DIR / "mark.npz", logo=logo)
    cv2.imwrite(str(ASSET_DIR / "logo_preview.png"), np.clip(logo / logo.max() * 255, 0, 255).astype(np.uint8))
    positions = json.loads((OUT_DIR / "jpeg_positions.json").read_text())
    anchor, jitter = anchor_fit(positions)
    styles = {name: fit_style(name) for name in ("bold", "regular")}
    # Text is drawn at the logo's opacity (text and logo peaks match in every per-listing
    # stack); least squares on stacks underestimates it for thin strokes, so use the
    # opacity inside thick logo strokes for both styles.
    core = cv2.erode((logo > 0.07).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    solid = round(float(np.median(logo[core])), 4)
    for v in styles.values():
        v["opacity"] = solid
    # The regular style's geometry from a joint fit over 250 single photos
    # (fit_style_photos.py) is better constrained than the 4 noisy listing stacks.
    styles["regular"].update({"center_x": 159.0, "top": 103.25})
    print(f"text opacity set to the logo's solid opacity {solid}")
    style_probs = json.loads((OUT_DIR / "style_stats.json").read_text())["probs"] \
        if (OUT_DIR / "style_stats.json").exists() else {"bold": 0.75, "regular": 0.25}
    # geometry and shares re-fitted on photos registered on the logo alone (refit_text.py);
    # the placement still varies by a few pixels per photo, which the localiser searches for.
    # Marks without a text line ("none", ~1%) are handled by the localiser / training config.
    refit = OUT_DIR / "text_refit.json"
    if refit.exists():
        r = json.loads(refit.read_text())
        for k, v in r["styles"].items():
            styles[k].update(v)
        style_probs = r["probs"]
    for k, v in styles.items():
        v["prob"] = style_probs.get(k, 0.0)
    sizes = sorted({tuple(p["size"]) for p in positions})
    sizes_all = [p["size"] for p in positions]  # with repeats: the empirical distribution
    meta = {
        "description": "Cian watermark: logo raster + 'ID <listing id>' in DejaVu Sans (bold or regular), "
                       "centred under the logo; fixed pixel size, bottom-right anchor with jitter; "
                       "served as JPEG q80 (some q85) 4:2:0.",
        "frame": [int(logo.shape[1]), int(logo.shape[0])],
        "overlay": OVERLAY,
        "text": {"prefix": "ID ", "styles": styles, "id_lengths": {"9": 0.97, "8": 0.03}},
        "anchor": anchor,
        "jitter": jitter,
        "window": [352, 224],
        "grid": 16,
        "codec": {"jpeg_quality": {"80": 0.91, "85": 0.09}, "subsampling": "4:2:0"},
        "sizes": sizes_all,
    }
    # logo shadow and text-line edge model fitted against synthetic marks (fit_halo.py --profile)
    model = OUT_DIR / "mark_model.json"
    if model.exists():
        meta.update(json.loads(model.read_text()))
    (ASSET_DIR / "mark.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {ASSET_DIR} ({len(sizes)} distinct sizes)")
