"""
Download all images from a Hugging Face dataset.
Preserves directory structure and avoids duplicate downloads.
"""

import os
from huggingface_hub import HfApi, hf_hub_download

# ==============================
# CONFIGURATION
# ==============================

HF_REPO_ID = "knl2366/NREL_Sky_Imagery"
REPO_TYPE = "dataset"
LOCAL_OUTPUT_DIR = "./NREL_Sky_Imagery"
HF_TOKEN = os.getenv("HF_TOKEN")  # Required only if dataset is private

# ==============================
# INITIALIZE API
# ==============================

api = HfApi(token=HF_TOKEN)

# ==============================
# FETCH FILE LIST
# ==============================

print("Fetching file list from Hugging Face...")
repo_files = api.list_repo_files(
    repo_id=HF_REPO_ID,
    repo_type=REPO_TYPE,
)

print(f"Found {len(repo_files)} total files.")

# ==============================
# DOWNLOAD FILES
# ==============================

downloaded = 0
skipped = 0

for file_path in repo_files:

    # Only download image files
    if not file_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    local_path = os.path.join(LOCAL_OUTPUT_DIR, file_path)

    # Skip if already exists locally
    if os.path.exists(local_path):
        skipped += 1
        continue

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file_path,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )

        # Move from cache to desired location
        os.replace(cached_path, local_path)

        downloaded += 1
        print(f"Downloaded: {file_path}")

    except Exception as e:
        print(f"Failed to download {file_path}: {e}")

print("Done.")
print(f"Downloaded: {downloaded}")
print(f"Skipped (already existed): {skipped}")
