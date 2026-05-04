"""
Compute dataset statistics for the paper (Section 5d).

Usage:
    python dataset_stats.py --image-dir /path/to/images --meteo-dir /path/to/meteorological
"""

import argparse
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


def compute_image_stats(image_dir: str):
    """Compute statistics about the image dataset."""
    root = Path(image_dir)
    images = list(root.glob("*/*/*/**.jpg"))

    if not images:
        print("No images found")
        return

    # Count by date
    daily_counts = defaultdict(int)
    total_size = 0
    sizes = []

    for img in images:
        try:
            parts = img.relative_to(root).parts
            if len(parts) >= 3:
                d = date(int(parts[0]), int(parts[1]), int(parts[2]))
                daily_counts[d] += 1
            size = img.stat().st_size
            total_size += size
            sizes.append(size)
        except (ValueError, IndexError, OSError):
            continue

    all_dates = sorted(daily_counts.keys())
    counts = list(daily_counts.values())

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nImages:")
    print(f"  Total images: {len(images):,}")
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"  Total calendar days: {(all_dates[-1] - all_dates[0]).days + 1}")
    print(f"  Days with data: {len(daily_counts)}")
    print(f"  Completeness: {len(daily_counts) / ((all_dates[-1] - all_dates[0]).days + 1) * 100:.1f}%")
    print(f"\n  Images/day statistics:")
    print(f"    Mean: {np.mean(counts):.1f}")
    print(f"    Median: {np.median(counts):.1f}")
    print(f"    Min: {np.min(counts)}")
    print(f"    Max: {np.max(counts)}")
    print(f"    Std: {np.std(counts):.1f}")
    print(f"\n  File sizes:")
    print(f"    Total: {total_size / 1e9:.2f} GB")
    print(f"    Mean per image: {np.mean(sizes) / 1e3:.1f} KB")
    print(f"    Growth rate: ~{np.mean(counts) * np.mean(sizes) / 1e6:.1f} MB/day")


def compute_meteo_stats(meteo_dir: str):
    """Compute statistics about the meteorological data."""
    import pandas as pd

    root = Path(meteo_dir)
    csvs = list(root.glob("*/*/*.csv"))

    if not csvs:
        print("\nNo meteorological CSVs found")
        return

    total_rows = 0
    all_columns = set()

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, nrows=5)
            all_columns.update(df.columns)
            df_full = pd.read_csv(csv_path)
            total_rows += len(df_full)
        except Exception:
            continue

    print(f"\nMeteorological Data:")
    print(f"  CSV files: {len(csvs)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Variables: {len(all_columns)}")
    print(f"  Key columns: {sorted([c for c in all_columns if any(k in c.lower() for k in ['global', 'direct', 'diffuse', 'temp', 'wind', 'humid', 'zenith'])])[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--image-dir", required=True, help="Root image directory")
    parser.add_argument("--meteo-dir", default=None, help="Meteorological data directory")
    args = parser.parse_args()

    compute_image_stats(args.image_dir)
    if args.meteo_dir:
        compute_meteo_stats(args.meteo_dir)


if __name__ == "__main__":
    main()
