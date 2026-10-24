"""Photos with a second, displaced Avito mark in the 320x160 corner (re-uploaded after cropping /
resizing): high-pass NCC of the logo at a few scales away from the nominal position, with a
noise floor in the normalisation so flat regions do not match. Writes out/second_mark.json."""
import sys, json

from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
from common import OUT, index, render_logo, luma, detail, split
CW, CH = 320, 160
params = json.loads((OUT / "model.json").read_text())
nom = (CW - 124 + params["x"], CH - 56 + params["y"])          # nominal logo top-left in the corner crop
tmpls = {}
for s in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4):
    w = params["width"] * s
    t = render_logo(w, 2, 2, frame=(int(w) + 5, int(w * 2032 / 8001) + 5))
    tmpls[s] = detail(t, 2.0).astype(np.float32)
def one(r):
    img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
    if img is None or img.shape[0] < CH or img.shape[1] < CW: return None
    g = detail(luma(img[-CH:, -CW:, ::-1].astype(np.float32)), 2.0).astype(np.float32)
    best = (0, None, None)
    for s, t in tmpls.items():
        t0 = t - t.mean()
        num = cv2.matchTemplate(g, t0, cv2.TM_CCORR)
        e = cv2.boxFilter(g * g, -1, t.shape[::-1], normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        e = e[:num.shape[0], :num.shape[1]]
        ncc = num / np.sqrt((e + 4.0 * t.size) * (t0 * t0).sum())     # +2 levels rms of noise floor
        ys, xs = np.mgrid[:ncc.shape[0], :ncc.shape[1]]
        far = (np.abs(xs + 2 - nom[0]) > 4) | (np.abs(ys + 2 - nom[1]) > 4) | (s != 1.0)
        if s != 1.0:
            far &= (np.abs(xs + 2 - nom[0]) > 4 * s) | (np.abs(ys + 2 - nom[1]) > 4 * s)
        v = np.where(far, ncc, -1)
        i = np.unravel_index(v.argmax(), v.shape)
        if v[i] > best[0]: best = (float(v[i]), s, (int(i[1] + 2 - nom[0]), int(i[0] + 2 - nom[1])))
    return {"sha": r["sha256"], "split": split(r["sha256"]), "ncc": best[0], "scale": best[1], "offset": best[2]}
rows = index(CW, CH)
with ThreadPoolExecutor(10) as ex:
    res = [x for x in ex.map(one, rows) if x]
(OUT / "second_mark.json").write_text(json.dumps(res))
n = np.array([x["ncc"] for x in res])
print(len(res), {t: int((n > t).sum()) for t in (0.4, 0.5, 0.6, 0.7)})
for x in sorted(res, key=lambda x: -x["ncc"])[:40]:
    print(x["sha"][:10], round(x["ncc"], 2), x["scale"], x["offset"])
