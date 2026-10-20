"""Per-listing stacks from the parser JPEGs (all photos of a few listings) to study the ID line."""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

from common import *
from build_template import _init, process
from stack import aligned_crop, estimate, save_vis

if __name__ == "__main__":
    n_listings = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rows = list(csv.DictReader(open(OUT_DIR / "cian_index.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[r["listing_id"]].append(r)
    lids = [l for l, rs in by.items() if len(rs) >= 15]
    random.Random(5).shuffle(lids)
    lids = lids[:n_listings]
    m = np.load(OUT_DIR / "m_jpeg.npy")
    todo = [r for l in lids for r in by[l]]
    with Pool(12, initializer=_init, initargs=(m,)) as pool:
        res = [r for r in pool.imap_unordered(process, todo, chunksize=4) if r is not None]
    out, stacks = [], []
    for l in lids:
        good = [r for r in res if r["listing_id"] == l and r["gain"] >= 0.15]
        if len(good) < 5:
            continue
        crops = [aligned_crop(luma(load_rgb(Path(r["path"]))), r["x"], r["y"]) for r in good]
        ml, _, _ = estimate(crops, c_fixed=234.0, iters=3)
        np.save(OUT_DIR / f"mj_{l}.npy", ml)
        stacks.append(ml[95:135])
        sizes = sorted({tuple(r["size"]) for r in good})
        out.append({"listing_id": l, "n": len(good), "sizes": sizes})
        print(l, len(good), sizes[:4])
    save_vis("mj_listings.png", np.concatenate(stacks, axis=0))
    (OUT_DIR / "per_listing.json").write_text(json.dumps({"listings": out, "positions": res}))
