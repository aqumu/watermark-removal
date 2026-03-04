from __future__ import annotations

import csv
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


def _url_to_filename(url: str) -> str:
    return Path(url.split("?")[0]).name


def _download_one(url: str, dest: Path, timeout: int = 15) -> tuple:
    """Download a single URL to dest. Returns (url, success, error_message)."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return url, True, ""
    except Exception as e:
        try:
            dest.unlink()
        except FileNotFoundError:
            pass
        return url, False, str(e)


def download_images(config: dict) -> None:
    """Ensure the clean_images dir has at least num_images images, downloading as needed.

    Downloads in rounds. URLs that fail are collected and written to failed_urls_path;
    each subsequent round pulls fresh URLs from the remaining pool until the target is
    reached or the pool is exhausted.
    """
    dl_cfg = config.get("download", {})
    csv_path = Path(dl_cfg.get("csv_path", "./data/sql.csv"))
    num_images = int(dl_cfg.get("num_images", 100))
    workers = int(dl_cfg.get("workers", 8))
    out_dir = Path(config["paths"]["clean_images_dir"])
    failed_path = Path(dl_cfg.get("failed_urls_path", str(csv_path.parent / "failed_urls.txt")))

    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    existing = {p.name for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts}
    already = len(existing)

    if already >= num_images:
        print(f"[downloader] {already} images present — nothing to download.")
        return

    print(f"[downloader] {already} images present, need {num_images - already} more (target: {num_images})")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_urls = [row["url"].strip() for row in reader if row.get("url", "").strip()]

    random.shuffle(all_urls)

    # Skip URLs whose output filename is already on disk
    url_pool = [u for u in all_urls if _url_to_filename(u) and _url_to_filename(u) not in existing]

    all_failed: list[str] = []
    round_num = 0

    while already < num_images and url_pool:
        round_num += 1
        needed = num_images - already
        batch, url_pool = url_pool[:needed], url_pool[needed:]

        to_download = [(url, out_dir / _url_to_filename(url)) for url in batch]
        round_failed: list[str] = []
        round_ok = 0

        desc = "Downloading" if round_num == 1 else f"Downloading (retry {round_num - 1})"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, url, dest): url for url, dest in to_download}
            with tqdm(total=len(to_download), unit="img", desc=desc) as bar:
                for fut in as_completed(futures):
                    url, ok, _ = fut.result()
                    if ok:
                        round_ok += 1
                        already += 1
                    else:
                        round_failed.append(url)
                    bar.update(1)

        print(f"[downloader] Round {round_num}: {round_ok} downloaded, {len(round_failed)} failed.")

        if round_failed:
            all_failed.extend(round_failed)

    if all_failed:
        failed_set = set(all_failed)

        # Append failed URLs to failed_urls.txt
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "a", encoding="utf-8") as f:
            for url in all_failed:
                f.write(url + "\n")
        print(f"[downloader] {len(all_failed)} failed URLs written to {failed_path}")

        # Remove failed URLs from sql.csv
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [row for row in reader if row.get("url", "").strip() not in failed_set]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[downloader] Removed {len(all_failed)} failed URLs from {csv_path}")

    if already < num_images:
        print(f"[downloader] Warning: only {already}/{num_images} images available after exhausting URL pool.")
    else:
        print(f"[downloader] Done. {already} images ready.")
