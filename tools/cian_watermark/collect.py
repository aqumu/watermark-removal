"""Index real Cian photos that the parser stored locally, with their listing IDs.

Streams a parser dump (pg_dump, gzip) and joins public.notice_images (url, path)
with public.notices (listing url -> Cian listing id). Keeps rows whose stored
file exists under the image store. Writes out/cian_index.csv:
    listing_id, position, path, url, size
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

from common import OUT_DIR

DUMP = Path(r"C:\Users\aqumu\PycharmProjects\deduper\dumps\parser-dump-2026-10-01.sql.gz")
STORE = Path(r"C:\Users\aqumu\PycharmProjects\photo-lab\images")
LISTING_RE = re.compile(r"cian\.ru/(?:sale|rent)/[a-z]+/(\d+)/?")


def read_copy_blocks(dump: Path, tables: set[str]):
    """Yield (table, columns, row_fields) for COPY blocks of the requested tables."""
    current = None
    columns: list[str] = []
    with gzip.open(dump, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if current is None:
                if line.startswith("COPY "):
                    m = re.match(r"COPY ([\w.]+) \((.*)\) FROM stdin;", line)
                    if m and m.group(1) in tables:
                        current = m.group(1)
                        columns = [c.strip().strip('"') for c in m.group(2).split(",")]
                continue
            if line.startswith("\\."):
                current = None
                continue
            yield current, columns, line.rstrip("\n").split("\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    notices: dict[str, str] = {}
    images: list[dict] = []
    for table, cols, row in read_copy_blocks(args.dump, {"public.notices", "public.notice_images"}):
        rec = dict(zip(cols, row))
        if table == "public.notices":
            m = LISTING_RE.search(rec.get("url", ""))
            if m:
                notices[rec["id"]] = m.group(1)
        elif "cdn-cian.ru" in rec.get("url", "") and rec.get("path") not in (None, "", "\\N"):
            images.append(rec)

    OUT_DIR.mkdir(exist_ok=True)
    kept = 0
    with open(OUT_DIR / "cian_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["listing_id", "position", "path", "url", "size"])
        for rec in images:
            lid = notices.get(rec["notice_id"])
            path = args.store / rec["path"]
            if lid is None or not path.exists():
                continue
            w.writerow([lid, rec["position"], str(path), rec["url"], rec.get("size", "")])
            kept += 1
    print(f"cian listings in notices: {len(set(notices.values()))}; cian image rows: {len(images)}; "
          f"stored locally with listing id: {kept}")


if __name__ == "__main__":
    main()
