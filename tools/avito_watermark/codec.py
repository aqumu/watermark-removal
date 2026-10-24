"""JPEG quality (IJG-equivalent, from the luma quantisation table) and chroma subsampling of the stored Avito photos."""
from collections import Counter

import numpy as np
from PIL import Image

from common import OUT, index

STD = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def ijg_quality(q):
    best = None
    for qual in range(1, 101):
        sc = 5000 / qual if qual < 50 else 200 - 2 * qual
        t = np.clip((STD * sc + 50) // 100, 1, 255)
        err = np.abs(np.sort(t) - np.sort(np.array(q))).sum()
        if best is None or err < best[0]:
            best = (err, qual)
    return best[1] if best[0] < 8 else f"~{best[1]}(nonstd)"


rows = index(1, 1)
rng = np.random.default_rng(1)
qs, subs = Counter(), Counter()
for i in rng.choice(len(rows), 1500, replace=False):
    im = Image.open(rows[i]["path"])
    qs[ijg_quality(im.quantization[0])] += 1
    from PIL import JpegImagePlugin
    subs[JpegImagePlugin.get_sampling(im)] += 1
print("quality:", qs.most_common(10))
print("subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0):", subs)
