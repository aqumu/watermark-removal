"""Step 1: index the Avito photos in the parser store and average their corners.

Writes out/avito_index.csv (sha, path, w, h) and out/corner_mean_<w>x<h>.png for the
most common sizes: the mean of the bottom-right 320x160 corner, contrast-stretched.
A mark at a fixed pixel offset from the corner shows up in the mean; one that scales
with the image or moves around does not.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

STORE = Path(r"C:\Users\aqumu\PycharmProjects\photo-lab\images")
MANIFEST = Path(r"C:\Users\aqumu\PycharmProjects\photo-lab\manifest_new.jsonl")
OUT = Path(__file__).resolve().parent / "out"
CW, CH = 320, 160


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for line in MANIFEST.open(encoding="utf-8"):
        r = json.loads(line)
        if "img.avito.st" not in r["url"]:
            continue
        sha = r["sha256"]
        p = STORE / sha[:2] / sha[2:4] / f"{sha}.jpg"
        if p.exists():
            rows.append((sha, p, r["url"]))
    print(f"avito rows with a local file: {len(rows)}")
    sums: dict = {}
    sizes = Counter()
    with open(OUT / "avito_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "path", "w", "h", "url"])
        for i, (sha, p, url) in enumerate(rows):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, wd = img.shape[:2]
            w.writerow([sha, str(p), wd, h, url])
            sizes[(wd, h)] += 1
            if h >= CH and wd >= CW:
                g = img[h - CH:, wd - CW:].astype(np.float64)
                s = sums.setdefault((wd, h), [np.zeros_like(g), 0])
                s[0] += g; s[1] += 1
            if i % 5000 == 0:
                print(i)
    print("top sizes:", sizes.most_common(15))
    allsum = sum(v[0] for v in sums.values()); n = sum(v[1] for v in sums.values())
    for key, (s, k) in [(("all",), (allsum, n))] + [(sz, sums[sz]) for sz, _ in sizes.most_common(4)]:
        m = s / k
        st = (m - m.min()) / (m.max() - m.min() + 1e-9) * 255
        cv2.imwrite(str(OUT / f"corner_mean_{'x'.join(map(str, key))}.png"), cv2.resize(st.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        print(key, k)


if __name__ == "__main__":
    main()
