"""Step 12: the worst real photos with a single (ordinary) mark.

Like worst_cases.py, but drops photos with a second mark (same position: the exact
inverse leaves k >= 0.25; displaced: second_mark.py NCC >= 0.35) and clipped
backgrounds, and adds local measures that the global leftover k misses:
  leftover   |k| of the model output over the whole logo
  local      the largest |k| over the logo's left / middle / right thirds
  patch      the largest smoothed |model - exact inverse| inside the logo (a tinted or
             shifted patch; the inverse is unbiased on unclipped single marks)
  smudge     texture lost under the logo: high-pass level inside vs around the logo
             (photos with a textured surround only)
Writes out/single_<criterion>.jpg and out/single.json.

Usage: python worst_single.py --config ... --checkpoint ... [--raw]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import OUT, index, split
from eval_model import load_runner, load_mark
from accuracy import FH, FW, _S, _init, amplitude, hp
from worst_cases import CTX_H, CTX_W, read, sheet


def k_of(img, sup):
    al, a = _S["ref"]
    E = a - al * cv2.GaussianBlur(img, (0, 0), 3.0)
    he, hr = hp(E)[sup], hp(img)[sup]
    return float((he * hr).sum() / max((he * he).sum(), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    profile = load_mark("avito")
    clean = load_runner(args, profile)
    _init(["none"])
    al, A = profile.alpha, profile.a
    logo = profile.coverage() > 0.5
    near = cv2.dilate(logo.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ring = cv2.dilate(logo.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~near
    cols = np.where(_S["sup"].any(0))[0]
    thirds = []
    for c in np.array_split(cols, 3):
        s = _S["sup"].copy()
        s[:, :c[0]] = False
        s[:, c[-1] + 1:] = False
        thirds.append(s)

    worst = {x["sha"]: x for x in json.loads((OUT / "worst.json").read_text())}
    second = {x["sha"]: x["ncc"] for x in json.loads((OUT / "second_mark.json").read_text())}
    k0 = float(np.median([x["k_in"] for x in worst.values()]))      # already normalised there: ~1
    ok = {s for s, x in worst.items() if x["k_inv"] < 0.25 and x["clip"] < 0.02 and second.get(s, 0) < 0.35}
    rows = [r for r in index(CTX_W, CTX_H) if r["sha256"] in ok]
    print(f"single-mark, unclipped photos: {len(rows)} of {len(worst)}")
    res, keep = [], {}
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(rows), 256):
            chunk = [(r, c) for r, c in pool.map(read, rows[i:i + 256]) if c is not None]
            outs = clean(np.stack([c[-64:, -128:] for _, c in chunk]))
            for (r, ctx), out_win in zip(chunk, outs):
                out = ctx.copy()
                out[-64:, -128:] = out_win
                frame = ctx[-FH:, -FW:].astype(np.float32)
                mod = out[-FH:, -FW:].astype(np.float32)
                inv = np.clip((frame - A) / np.maximum(1 - al, 0.05), 0, 255)
                k_in = k_of(frame, _S["sup"])
                d = cv2.GaussianBlur(mod - inv, (0, 0), 3.0)
                hf = np.abs(mod - cv2.GaussianBlur(mod, (0, 0), 1.5)).mean(-1)
                rec = {"sha": r["sha256"], "path": r["path"], "split": split(r["sha256"]),
                       "k_in": k_in, "leftover": k_of(mod, _S["sup"]),
                       "local": max((k_of(mod, s) for s in thirds), key=abs),
                       "patch": float(np.abs(d)[logo].max()),
                       "hf_ring": float(hf[ring].mean()), "hf_logo": float(hf[logo].mean()),
                       "bg": float(frame.mean())}
                rec["smudge"] = 1 - rec["hf_logo"] / rec["hf_ring"] if rec["hf_ring"] > 4 else 0.0
                res.append(rec)
                keep[r["sha256"]] = (ctx, out)
            if len(keep) > 3000:
                wanted = set()
                for key in ("leftover", "local", "patch", "smudge"):
                    wanted |= {x["sha"] for x in sorted(res, key=lambda x: -abs(x[key]))[:150]}
                keep = {k: v for k, v in keep.items() if k in wanted}
            print(f"{len(res)}/{len(rows)}", end="\r", flush=True)
    k0 = float(np.median([x["k_in"] for x in res]))
    for x in res:
        for key in ("k_in", "leftover", "local"):
            x[key] /= k0
    (OUT / "single.json").write_text(json.dumps(res))

    def lab(x):
        return (f"{x['sha'][:10]} ({x['split']})  leftover {x['leftover']:+.1%}  local {x['local']:+.1%}  "
                f"patch {x['patch']:.1f}  smudge {x['smudge']:.0%}  bg {x['bg']:.0f}")

    print()
    for key in ("leftover", "local", "patch", "smudge"):
        lst = sorted(res, key=lambda x: -abs(x[key]))
        v = np.abs([x[key] for x in res])
        print(f"{key:8s} median {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}  p99 {np.percentile(v, 99):.3f}  "
              f"max {v.max():.3f}")
        items = [{"ctx": keep[x["sha"]][0], "out": keep[x["sha"]][1], "label": lab(x)}
                 for x in lst if x["sha"] in keep][:args.top]
        cv2.imwrite(str(OUT / f"single_{key}.jpg"), sheet(items, "single mark, worst " + key)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
