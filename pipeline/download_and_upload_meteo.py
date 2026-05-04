"""
Download SRRL BMS meteorological data from the NREL MIDC API.

Downloads 1-minute meteorological measurements from the NREL Solar Radiation
Research Laboratory (SRRL) Baseline Measurement System (BMS). The BMS reports
253 columns per minute; this script saves a curated subset of ~25 key variables
most relevant for solar irradiance forecasting and sky-image-based ML research.

The full 253-column dataset is always available from the NREL MIDC API at:
    https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin=YYYYMMDD&end=YYYYMMDD

Output structure:
    <out-dir>/YYYY/MM/YYYYMMDD.csv   (one file per day, ~220 KB each)

Usage:
    # Download Jan 2025 through today
    python download_and_upload_meteo.py --start 2025-01-01 --end 2026-03-04

    # Download and upload to HuggingFace
    python download_and_upload_meteo.py --start 2025-01-01 --end 2026-03-04 --upload

    # Keep all 253 columns instead of the curated subset
    python download_and_upload_meteo.py --start 2025-01-01 --end 2026-03-04 --all-columns

    # Dry run (show what would be downloaded, don't write files)
    python download_and_upload_meteo.py --start 2025-01-01 --end 2026-03-04 --dry-run
"""

import argparse
import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from config import (
    BMS_SITE,
    HF_REPO_ID,
    HF_REPO_TYPE,
    NREL_BMS_API_URL,
)

# ── Column selection ─────────────────────────────────────────────────
# The BMS reports 253 columns. Most are redundant instrument readings
# (multiple pyranometers, raw mV signals, case temperatures). We select
# the ~25 variables most useful for ML-based solar forecasting.
#
# Reference instruments chosen per NREL SRRL best-practice guidance:
#   GHI  → Global CMP22 (ventilated, corrected) — research-grade thermopile
#   DNI  → Direct CHP1-1  — pyrheliometer on tracker
#   DHI  → Diffuse CM22-1 (ventilated, corrected) — shaded thermopile

CURATED_COLUMNS = [
    # ── Time identification ──
    "Year",
    "DOY",
    "MST",
    # ── Solar irradiance (W/m^2) ──
    "Global CMP22 (vent/cor) [W/m^2]",       # GHI — reference
    "Direct NIP [W/m^2]",                     # DNI — normal incidence pyrheliometer
    "Direct CHP1-1 [W/m^2]",                 # DNI — companion pyrheliometer
    "Diffuse CM22-1 (vent/cor) [W/m^2]",     # DHI — reference
    # ── Solar geometry ──
    "Zenith Angle [degrees]",
    "Azimuth Angle [degrees]",
    "Airmass",
    "Global Extraterrestrial (calc) [W/m^2]",
    # ── Atmospheric conditions ──
    "Tower Dry Bulb Temp [deg C]",
    "Tower RH [%]",
    "Tower Dew Point Temp [deg C]",
    "Tower Wet Bulb Temp [deg C]",
    "Station Pressure [mBar]",
    "Sea-Level Pressure (Est) [mBar]",
    # ── Wind ──
    "Avg Wind Speed @ 19ft [m/s]",
    "Avg Wind Direction @ 19ft [deg from N]",
    "Peak Wind Speed @ 19ft [m/s]",
    # ── Cloud & sky ──
    "Total Cloud Cover [%]",
    "Opaque Cloud Cover [%]",
    # ── Surface & precipitation ──
    "Albedo (CMP11)",
    "Snow Depth [cm]",
    "Precipitation [mm]",
]

# Short aliases for cleaner CSV headers (optional rename map)
COLUMN_ALIASES = {
    "Global CMP22 (vent/cor) [W/m^2]": "GHI [W/m^2]",
    "Direct NIP [W/m^2]": "DNI [W/m^2]",
    "Direct CHP1-1 [W/m^2]": "DNI_CHP1 [W/m^2]",
    "Diffuse CM22-1 (vent/cor) [W/m^2]": "DHI [W/m^2]",
    "Global Extraterrestrial (calc) [W/m^2]": "ETR [W/m^2]",
    "Tower Dry Bulb Temp [deg C]": "Temperature [deg C]",
    "Tower RH [%]": "Relative Humidity [%]",
    "Tower Dew Point Temp [deg C]": "Dew Point [deg C]",
    "Tower Wet Bulb Temp [deg C]": "Wet Bulb [deg C]",
    "Station Pressure [mBar]": "Pressure [mBar]",
    "Sea-Level Pressure (Est) [mBar]": "Sea Level Pressure [mBar]",
    "Avg Wind Speed @ 19ft [m/s]": "Wind Speed [m/s]",
    "Avg Wind Direction @ 19ft [deg from N]": "Wind Direction [deg]",
    "Peak Wind Speed @ 19ft [m/s]": "Peak Wind Speed [m/s]",
    "Total Cloud Cover [%]": "Cloud Cover [%]",
    "Opaque Cloud Cover [%]": "Opaque Cloud Cover [%]",
    "Albedo (CMP11)": "Albedo",
    "Snow Depth [cm]": "Snow Depth [cm]",
    "Precipitation [mm]": "Precipitation [mm]",
    "Zenith Angle [degrees]": "Zenith [deg]",
    "Azimuth Angle [degrees]": "Azimuth [deg]",
}

# MIDC API allows up to ~365 days per request, but large requests are slow.
# 30-day chunks balance speed and reliability.
from config import BMS_MAX_DAYS_PER_REQUEST as MAX_DAYS_PER_REQUEST

# Missing data sentinel in BMS data
BMS_MISSING = -7999


# ── Core functions ───────────────────────────────────────────────────


def fetch_bms_data(start: date, end: date) -> pd.DataFrame:
    """Fetch BMS data from the NREL MIDC API for a date range.

    The API returns CSV text with 253 columns. The first column is unnamed
    (always 0) and is dropped. The API follows redirects automatically.

    Raises requests.HTTPError on failure.
    """
    url = (
        f"{NREL_BMS_API_URL}?site={BMS_SITE}"
        f"&begin={start.strftime('%Y%m%d')}"
        f"&end={end.strftime('%Y%m%d')}"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    # Drop the unnamed first column (always 0)
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])

    return df


def select_columns(df: pd.DataFrame, all_columns: bool = False) -> pd.DataFrame:
    """Select curated columns and rename to shorter aliases."""
    if all_columns:
        return df

    available = [c for c in CURATED_COLUMNS if c in df.columns]
    missing = [c for c in CURATED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns (API may have changed): {missing}")

    df = df[available].copy()
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    df = df.rename(columns=rename)
    return df


def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace BMS missing-data sentinel (-7999) with NaN."""
    return df.replace(BMS_MISSING, pd.NA)


def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ISO 8601 datetime column from Year, DOY, MST."""
    # MST is HHMM as an integer (0-2359)
    hours = df["MST"] // 100 if "MST" in df.columns else df["Hour"] // 100
    minutes = df["MST"] % 100 if "MST" in df.columns else df["Hour"] % 100

    dt = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" + df["DOY"].astype(int).astype(str),
        format="%Y-%j",
    )
    df = df.copy()
    df.insert(
        0,
        "Datetime (MST)",
        dt + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m"),
    )
    return df


def split_and_save(
    df: pd.DataFrame,
    out_dir: Path,
    all_columns: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Split a multi-day DataFrame into daily CSV files.

    Returns list of paths written.
    """
    df = select_columns(df, all_columns=all_columns)
    df = replace_missing(df)
    df = add_datetime_column(df)

    files = []
    for (year, doy), group in df.groupby(["Year", "DOY"]):
        try:
            dt = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(
                days=int(doy) - 1
            )
        except Exception as e:
            print(f"  [WARN] Skipping Year={year}, DOY={doy}: {e}")
            continue

        day_dir = out_dir / f"{dt.year:04d}" / f"{dt.month:02d}"
        day_path = day_dir / f"{dt.strftime('%Y%m%d')}.csv"

        if dry_run:
            print(f"  Would write {day_path} ({len(group)} rows)")
            files.append(day_path)
            continue

        day_dir.mkdir(parents=True, exist_ok=True)

        # Drop Year/DOY (redundant with Datetime column) for cleaner output
        out = group.drop(columns=["Year", "DOY"], errors="ignore")
        out.to_csv(day_path, index=False, float_format="%.6g")
        files.append(day_path)

    return files


# ── Download orchestration ───────────────────────────────────────────


def download_range(
    start: date,
    end: date,
    out_dir: Path,
    all_columns: bool = False,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> list[Path]:
    """Download BMS data in chunks and save as daily CSVs.

    Splits the date range into MAX_DAYS_PER_REQUEST-day chunks to keep
    API requests reliable. Skips days that already have a CSV on disk.
    """
    all_files = []
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end)

        # Check if all days in this chunk already exist
        if skip_existing:
            all_exist = True
            check = current
            while check <= chunk_end:
                csv = out_dir / f"{check.year:04d}" / f"{check.month:02d}" / f"{check.strftime('%Y%m%d')}.csv"
                if not csv.exists():
                    all_exist = False
                    break
                check += timedelta(days=1)
            if all_exist:
                print(
                    f"  {current} → {chunk_end}: all files exist, skipping"
                )
                # Collect existing paths
                check = current
                while check <= chunk_end:
                    csv = out_dir / f"{check.year:04d}" / f"{check.month:02d}" / f"{check.strftime('%Y%m%d')}.csv"
                    all_files.append(csv)
                    check += timedelta(days=1)
                current = chunk_end + timedelta(days=1)
                continue

        print(
            f"  {current} → {chunk_end}: downloading..."
        )

        try:
            df = fetch_bms_data(current, chunk_end)
            files = split_and_save(df, out_dir, all_columns=all_columns, dry_run=dry_run)
            all_files.extend(files)
            print(f"  {current} → {chunk_end}: {len(files)} daily files")
        except requests.HTTPError as e:
            print(f"  {current} → {chunk_end}: HTTP error — {e}")
        except Exception as e:
            print(f"  {current} → {chunk_end}: FAILED — {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # rate-limit: be polite to NREL

    return all_files


def upload_to_huggingface(local_dir: Path):
    """Upload meteorological directory to HuggingFace."""
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[ERROR] Set HF_TOKEN environment variable before uploading")
            return

        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            path_in_repo="meteorological",
        )
        print("Uploaded meteorological data to HuggingFace")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download SRRL BMS meteorological data from NREL MIDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --start 2025-01-01 --end 2026-03-04
  %(prog)s --start 2025-01-01 --end 2026-03-04 --upload
  %(prog)s --start 2025-01-01 --end 2026-03-04 --all-columns
  %(prog)s --start 2025-01-01 --end 2026-03-04 --dry-run
        """,
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=Path, default=Path("meteorological"),
                        help="Output directory (default: meteorological)")
    parser.add_argument("--all-columns", action="store_true",
                        help="Keep all 253 BMS columns instead of curated subset")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace after download")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without writing files")
    args = parser.parse_args()

    n_days = (args.end - args.start).days + 1
    col_mode = "all 253 columns" if args.all_columns else f"curated {len(CURATED_COLUMNS)} columns"
    print(f"Downloading BMS data: {args.start} → {args.end} ({n_days} days, {col_mode})")
    print(f"Output: {args.out_dir.resolve()}")

    files = download_range(
        args.start, args.end, args.out_dir,
        all_columns=args.all_columns,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    )

    print(f"\nTotal: {len(files)} daily CSV files")

    if args.upload and not args.dry_run:
        upload_to_huggingface(args.out_dir)


if __name__ == "__main__":
    main()
