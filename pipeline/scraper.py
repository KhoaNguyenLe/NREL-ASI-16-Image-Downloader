"""
Continuous NREL ASI-16 sky image scraper.

Polls the real-time NREL MIDC image feed every ~60 seconds, deduplicates
by MD5 hash, filters blank nighttime frames, and saves to a date-organized
directory structure (YYYY/MM/DD/YYYYMMDD-HHMMSS.jpg).

Designed to run as a systemd service on Jetstream2.

Usage:
    python scraper.py                          # default output dir from config
    python scraper.py /path/to/output          # custom output dir
"""

import datetime
import hashlib
import logging
import os
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import requests

from config import (
    BLANK_IMAGE_MD5,
    DEFAULT_IMAGE_DIR,
    NREL_IMAGE_URL,
    SCRAPE_INTERVAL_SEC,
    SCRAPE_WAIT_OFFSET_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MST = ZoneInfo("America/Denver")


def wait_until_next_minute(seconds=0):
    """Sleep until `seconds` past the next minute boundary."""
    now = datetime.datetime.now()
    next_min = (now + datetime.timedelta(minutes=1)).replace(second=seconds, microsecond=0)
    time.sleep(max(0, (next_min - now).total_seconds()))


def get_and_save_image(out_dir):
    """Main scraping loop: fetch, deduplicate, save."""
    previous_hash = None

    while True:
        try:
            response = requests.get(NREL_IMAGE_URL, timeout=15)
            response.raise_for_status()

            current_hash = hashlib.md5(response.content).hexdigest()

            # Skip blank nighttime placeholder
            if current_hash == BLANK_IMAGE_MD5:
                time.sleep(SCRAPE_WAIT_OFFSET_SEC)
                continue

            # Timestamp: image was captured ~1 minute ago at :33s
            capture_time = (
                datetime.datetime.now(tz=MST)
                - datetime.timedelta(minutes=1)
            ).replace(second=33, microsecond=0)

            filepath = os.path.join(
                out_dir,
                capture_time.strftime("%Y"),
                capture_time.strftime("%m"),
                capture_time.strftime("%d"),
                capture_time.strftime("%Y%m%d-%H%M%S.jpg"),
            )

            if previous_hash != current_hash:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                previous_hash = current_hash
                log.info("Saved %s", filepath)
                wait_until_next_minute(seconds=SCRAPE_WAIT_OFFSET_SEC)
            else:
                time.sleep(SCRAPE_WAIT_OFFSET_SEC)

        except requests.exceptions.RequestException as e:
            log.warning("Network error: %s", e)
            time.sleep(15)

        except Exception:
            log.error("Unexpected error:\n%s", traceback.format_exc())
            time.sleep(15)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    log.info("Starting scraper, output: %s", output_dir)
    get_and_save_image(output_dir)
