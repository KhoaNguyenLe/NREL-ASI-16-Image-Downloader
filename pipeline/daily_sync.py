"""
Daily sync: upload yesterday's images and meteorological data to HuggingFace.

Designed to run as a daily cron job or systemd timer on Jetstream2.
Performs three operations:
  1. Upload any un-uploaded image days to HF (using hfuploader logic)
  2. Download yesterday's BMS meteorological data from NREL MIDC
  3. Upload meteorological data to HF

Usage:
    # Upload everything not yet synced (catches up if missed days)
    python daily_sync.py

    # Sync a specific date range
    python daily_sync.py --start 2025-06-01 --end 2025-06-15

    # Dry run (check what would be synced)
    python daily_sync.py --dry-run

Environment:
    HF_TOKEN          - HuggingFace write token (required)
    SRRL_IMAGE_DIR    - Local image root (default: /media/volume/Primary-Dataset)
    SRRL_METEO_DIR    - Local meteo root (default: /media/volume/Primary-Dataset/meteorological)
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from config import (
    BMS_MAX_DAYS_PER_REQUEST,
    BMS_SITE,
    DEFAULT_IMAGE_DIR,
    DEFAULT_METEO_DIR,
    HF_REPO_ID,
    HF_REPO_TYPE,
    NREL_BMS_API_URL,
    TRACK_FILE,
    UPLOAD_BATCH_SLEEP_SEC,
    get_hf_token,
)


# ── Helpers ───────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_uploaded_days() -> set:
    if not TRACK_FILE.exists():
        return set()
    return set(TRACK_FILE.read_text().splitlines())


def mark_day_uploaded(day_str: str):
    with open(TRACK_FILE, "a") as f:
        f.write(day_str + "\n")


# ── Image upload ──────────────────────────────────────────────────────


def find_unuploaded_days(image_dir: Path, uploaded: set) -> list:
    """Find day directories that haven't been uploaded yet."""
    days = []
    for year_dir in sorted(image_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue
                day_str = f"{year_dir.name}/{month_dir.name}/{day_dir.name}"
                if day_str not in uploaded:
                    days.append((day_str, day_dir))
    return days


def upload_image_day(api, day_str: str, day_dir: Path, dry_run: bool = False):
    """Deduplicate and upload a single day's images to HF."""
    seen_hashes = set()
    unique_files = []

    for f in sorted(day_dir.glob("*.jpg")):
        fhash = sha256_file(f)
        if fhash not in seen_hashes:
            seen_hashes.add(fhash)
            unique_files.append(f)

    if not unique_files:
        print(f"  {day_str}: no unique images, skipping")
        return

    if dry_run:
        print(f"  {day_str}: {len(unique_files)} unique images (dry run)")
        return

    # Stage unique files in a temp directory for upload
    tmp = Path("_tmp_upload")
    tmp.mkdir(exist_ok=True)
    for f in unique_files:
        dest = tmp / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    try:
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            path_in_repo=day_str,
        )
        mark_day_uploaded(day_str)
        print(f"  {day_str}: uploaded {len(unique_files)} images")
        time.sleep(UPLOAD_BATCH_SLEEP_SEC)
    except Exception as e:
        print(f"  {day_str}: upload FAILED — {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Meteorological data ──────────────────────────────────────────────


def download_bms_day(target_date: date, meteo_dir: Path) -> Path | None:
    """Download one day of BMS data, select curated columns, and save as CSV."""
    from download_and_upload_meteo import (
        fetch_bms_data,
        split_and_save,
    )

    day_dir = meteo_dir / f"{target_date.year:04d}" / f"{target_date.month:02d}"
    day_csv = day_dir / f"{target_date.strftime('%Y%m%d')}.csv"

    if day_csv.exists():
        return day_csv  # already have it

    try:
        df = fetch_bms_data(target_date, target_date)
        files = split_and_save(df, meteo_dir)
        return files[0] if files else None
    except Exception as e:
        print(f"  meteo {target_date}: download FAILED — {e}")
        return None


def upload_meteo_to_hf(api, meteo_dir: Path, dry_run: bool = False):
    """Upload the entire meteorological directory to HF."""
    if dry_run:
        csvs = list(meteo_dir.glob("*/*/*.csv"))
        print(f"  meteo: {len(csvs)} CSV files (dry run)")
        return

    try:
        api.upload_folder(
            folder_path=str(meteo_dir),
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            path_in_repo="meteorological",
        )
        print("  meteo: uploaded to HuggingFace")
    except Exception as e:
        print(f"  meteo: upload FAILED — {e}")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Daily sync: images + meteo → HuggingFace"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help="Start date for meteo download (default: yesterday)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="End date for meteo download (default: yesterday)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(DEFAULT_IMAGE_DIR),
        help="Local image root directory",
    )
    parser.add_argument(
        "--meteo-dir",
        type=Path,
        default=Path(DEFAULT_METEO_DIR),
        help="Local meteorological data directory",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image upload (only sync meteo)",
    )
    parser.add_argument(
        "--skip-meteo",
        action="store_true",
        help="Skip meteo download/upload (only sync images)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without uploading",
    )
    args = parser.parse_args()

    yesterday = date.today() - timedelta(days=1)
    start = args.start or yesterday
    end = args.end or yesterday

    print(f"=== SRRL Daily Sync — {date.today()} ===")

    # Initialize HF API
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=get_hf_token())
    else:
        api = None

    # ── 1. Upload un-uploaded image days ──
    if not args.skip_images:
        print("\n[1/3] Checking for un-uploaded image days...")
        uploaded = load_uploaded_days()
        pending = find_unuploaded_days(args.image_dir, uploaded)
        print(f"  Found {len(pending)} day(s) to upload")
        for day_str, day_dir in pending:
            upload_image_day(api, day_str, day_dir, dry_run=args.dry_run)

    # ── 2. Download meteorological data ──
    if not args.skip_meteo:
        print(f"\n[2/3] Downloading BMS data {start} → {end}...")
        current = start
        meteo_files = []
        while current <= end:
            csv_path = download_bms_day(current, args.meteo_dir)
            if csv_path:
                meteo_files.append(csv_path)
            current += timedelta(days=1)
            time.sleep(0.5)  # be polite to API
        print(f"  {len(meteo_files)} daily CSV(s) ready")

    # ── 3. Upload meteo to HF ──
    if not args.skip_meteo:
        print("\n[3/3] Uploading meteorological data to HuggingFace...")
        upload_meteo_to_hf(api, args.meteo_dir, dry_run=args.dry_run)

    print("\n=== Sync complete ===")


if __name__ == "__main__":
    main()
