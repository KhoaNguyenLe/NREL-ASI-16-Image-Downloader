"""
Shared configuration for the SRRL sky imagery data pipeline.

All paths, constants, and credentials are centralized here.
Credentials MUST come from environment variables, never hardcoded.
"""

import os
from pathlib import Path

# === HuggingFace ===
HF_REPO_ID = "knl2366/NREL_Sky_Imagery"
HF_REPO_TYPE = "dataset"

def get_hf_token() -> str:
    """Get HF token from environment. Raises if not set."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Then: export HF_TOKEN=hf_..."
        )
    return token

# === NREL MIDC API ===
NREL_IMAGE_URL = "https://midcdmz.nrel.gov/data/rt/srrlasi.jpg"
NREL_BMS_API_URL = "https://midcdmz.nrel.gov/apps/data_api.pl"
BMS_SITE = "BMS"
BMS_MAX_DAYS_PER_REQUEST = 30  # MIDC API limit

# === Image dedup ===
# MD5 hash of the blank nighttime placeholder image served by NREL
BLANK_IMAGE_MD5 = "604c77fd179dd033f129c4397a8095eb"

# === Paths (Jetstream2 defaults, override via env vars) ===
DEFAULT_IMAGE_DIR = os.environ.get(
    "SRRL_IMAGE_DIR", "/media/volume/Primary-Dataset"
)
DEFAULT_METEO_DIR = os.environ.get(
    "SRRL_METEO_DIR", "/media/volume/Primary-Dataset/meteorological"
)

# === Timing ===
SCRAPE_INTERVAL_SEC = 60       # Poll NREL image every 60s
SCRAPE_WAIT_OFFSET_SEC = 10    # Wait until :10s past the minute
UPLOAD_BATCH_SLEEP_SEC = 5     # Pause between HF upload API calls
DAILY_SYNC_HOUR_UTC = 10       # Run daily sync at 10:00 UTC (03:00 MST)

# === Dataset metadata ===
SITE_LATITUDE = 39.742
SITE_LONGITUDE = -105.180
SITE_ELEVATION_M = 1829
SITE_NAME = "NREL Solar Radiation Research Laboratory (SRRL)"
CAMERA_MODEL = "EKO ASI-16"
IMAGE_RESOLUTION = (1920, 1920)
TIMEZONE = "America/Denver"  # MST/MDT

# === Upload tracking ===
TRACK_FILE = Path(__file__).parent / "uploaded_days.txt"
