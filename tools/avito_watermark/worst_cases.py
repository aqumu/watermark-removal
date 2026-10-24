"""Step 10: the worst real photos for a trained removal model, by several criteria.

Runs the model on every stored photo's corner window (training only used regions
away from the photos' own marks, so the corners are unseen) and ranks:
  leftover   |k| of the model output (accuracy.py's measure, relative to the input)
  double     photos with a second mark (the exact inverse leaves k >= 0.25)
  clipped    saturated backgrounds under the mark (channels at 255)
  disagree   largest difference between the model and the exact inverse inside
             the logo (where the inverse is reliable: neither clipped nor doubled)
Writes out/worst_<criterion>.jpg (input, output and both at x5 contrast, with
context around the window) and out/worst.json.

Usage: python worst_cases.py --config ... --checkpoint ... [--raw] [--split all|holdout]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import OUT, index, split
from eval_model import contrast, load_runner, load_mark
from accuracy import FH, FW, _S, _init, amplitude

CTX_W, CTX_H = 224, 96          # context shown around the 128 x 64 window


def read(row):
    img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    return row, None if img is None else np.ascontiguousarray(img[-CTX_H:, -CTX_W:, ::-1])


def sheet(items, title):
    rows = []
    for it in items:
        a, b = it["ctx"].astype(np.float32), it["out"].astype(np.float32)
        row = np.concatenate([a, b, contrast(a), contrast(b)], 1)
        row = cv2.resize(row, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        label = np.full((22, row.shape[1], 3), 255, np.float32)
        cv2.putText(label, it["label"], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows += [label, row, np.full((6, row.shape[1], 3), 255, np.float32)]
    head = np.full((30, rows[0].shape[1], 3), 255, np.float32)
    cv2.putText(head, title + "   |   input | model | input x5 contrast | model x5 contrast", (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([head] + rows, 0).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--split", default="all")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    profile = load_mark("avito")
    clean = load_runner(args, profile)
    _init(["none"])
    al, A = profile.alpha, profile.a
    logo = profile.coverage() > 0.5                       # glyph interiors (frame coordinates)
    rows = [r for r in index(CTX_W, CTX_H) if args.split == "all" or split(r["sha256"]) == args.split]
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
                k_in, k_inv, k_mod = amplitude(frame)[0], amplitude(inv)[0], amplitude(mod)[0]
                clip = float((frame[logo] >= 254).any(-1).mean())
                d = np.abs(mod - inv)[logo].mean(-1)
                rec = {"sha": r["sha256"], "path": r["path"], "split": split(r["sha256"]),
                       "k_in": k_in, "k_inv": k_inv, "k_mod": k_mod, "clip": clip,
                       "disagree": float(np.percentile(d, 95)), "bg": float(frame.mean())}
                res.append(rec)
                keep[r["sha256"]] = (ctx, out)
            # keep only what can still make a top list (memory)
            if len(keep) > 4000:
                wanted = set()
                for key in ("k_mod", "k_inv", "clip", "disagree"):
                    wanted |= {x["sha"] for x in sorted(res, key=lambda x: -abs(x[key]))[:200]}
                keep = {k: v for k, v in keep.items() if k in wanted}
            print(f"{len(res)}/{len(rows)}", end="\r", flush=True)
    k0 = float(np.median([x["k_in"] for x in res]))
    for x in res:
        for key in ("k_in", "k_inv", "k_mod"):
            x[key] /= k0
    (OUT / "worst.json").write_text(json.dumps(res))

    def lab(x):
        return (f"{x['sha'][:10]} ({x['split']})  leftover {x['k_mod']:+.1%}  exact inverse {x['k_inv']:+.1%}  "
                f"clipped {x['clip']:.0%}  bg {x['bg']:.0f}")

    single = [x for x in res if x["k_inv"] < 0.25 and x["clip"] < 0.02]
    lists = {
        "leftover": sorted(res, key=lambda x: -abs(x["k_mod"])),
        "double": sorted([x for x in res if x["k_inv"] >= 0.25], key=lambda x: -abs(x["k_mod"])),
        "clipped": sorted([x for x in res if x["clip"] >= 0.02], key=lambda x: -abs(x["k_mod"])),
        "disagree": sorted(single, key=lambda x: -x["disagree"]),
    }
    print(f"\nphotos: {len(res)}; double marks (exact inverse leaves >= 25%): {len(lists['double'])}; "
          f"clipped under the logo: {len(lists['clipped'])}")
    for name, lst in lists.items():
        items = [{"ctx": keep[x["sha"]][0], "out": keep[x["sha"]][1], "label": lab(x)}
                 for x in lst if x["sha"] in keep][:args.top]
        if items:
            cv2.imwrite(str(OUT / f"worst_{name}.jpg"), sheet(items, name)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        km = np.abs([x["k_mod"] for x in lst]) if lst else np.zeros(1)
        print(f"{name:9s} n={len(lst):5d}  leftover median {np.median(km):.1%}  p90 {np.percentile(km, 90):.1%}  "
              f"max {km.max():.1%}")


if __name__ == "__main__":
    main()
