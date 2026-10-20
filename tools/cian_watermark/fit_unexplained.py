"""Fit text style freely per photo for the photos no known style explains (style_stats 'mismatch')."""
import json
import sys

sys.path.insert(0, "../..")
from common import *  # noqa: E402
from fit_style_photos import prepare, total_gain  # noqa: E402
from wm_shared.marks import load_mark  # noqa: E402
from wm_shared.marks.profile import TextStyle  # noqa: E402

prof = load_mark("cian")
stats = json.loads((OUT_DIR / "style_stats.json").read_text())
pos = {(p["listing_id"], round(p["gain"], 6)): p for p in json.loads((OUT_DIR / "jpeg_positions.json").read_text())}
sel = [pos[(r["listing_id"], round(r["gain_logo"], 6))] for r in stats["photos"] if r["gains"][r["best"]] < 0.1]
sel = sorted(sel, key=lambda p: -p["gain"])[:int(sys.argv[1]) if len(sys.argv) > 1 else 40]
data = prepare(sel, prof.logo)
fw, fh = prof.frame_size
F = "../../assets/marks/cian/fonts/"
out = []
for item, p in zip(data, sel):
    best = (-9, None)
    for font in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        for size in (21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0):
            for top in np.arange(96.0, 110.1, 1.0):
                for cx in np.arange(146.0, 172.1, 2.0):
                    st = TextStyle(font=F + font, size=size, opacity=0.12, center_x=cx, top=top)
                    g = total_gain([item], st, (fh, fw))
                    if g > best[0]:
                        best = (g, ("bold" if "Bold" in font else "reg", size, float(top), float(cx)))
    out.append({"path": p["path"], "listing_id": item[0], "gain": best[0], "fit": best[1]})
    print(item[0], round(best[0], 3), best[1], flush=True)
(OUT_DIR / "unexplained_fits.json").write_text(json.dumps(out, indent=1))
