"""
Generate a calendar heatmap showing the number of images captured per day.
Produces Figure 2 for the DMLR paper.

Usage:
    python completeness_heatmap.py --image-dir /path/to/images --output figures/completeness_heatmap.pdf
"""

import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def count_images_per_day(image_dir: str) -> dict:
    """Count JPEG images per day in YYYY/MM/DD/ directory structure."""
    counts = defaultdict(int)
    root = Path(image_dir)

    for jpg in root.glob("*/*/*/**.jpg"):
        try:
            # Extract date from directory structure: YYYY/MM/DD/
            parts = jpg.relative_to(root).parts
            if len(parts) >= 3:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                d = date(year, month, day)
                counts[d] += 1
        except (ValueError, IndexError):
            continue

    return dict(counts)


def plot_calendar_heatmap(counts: dict, output_path: str):
    """Plot a GitHub-style calendar heatmap of image counts per day."""
    if not counts:
        print("No data to plot")
        return

    all_dates = sorted(counts.keys())
    start_date = all_dates[0]
    end_date = all_dates[-1]

    # Create figure
    # Calculate number of weeks
    total_days = (end_date - start_date).days + 1
    n_weeks = (total_days + start_date.weekday()) // 7 + 2

    fig, ax = plt.subplots(figsize=(max(12, n_weeks * 0.3), 3))

    # Build grid
    max_count = max(counts.values()) if counts else 1
    # Expected ~720 images/day at equinox; normalize to this
    norm = mcolors.Normalize(vmin=0, vmax=min(max_count, 800))
    cmap = plt.cm.YlOrRd

    current = start_date
    from datetime import timedelta

    while current <= end_date:
        week = (current - start_date).days // 7
        weekday = current.weekday()  # 0=Monday

        count = counts.get(current, 0)
        color = cmap(norm(count)) if count > 0 else "#ebedf0"

        rect = plt.Rectangle(
            (week, 6 - weekday), 1, 1,
            facecolor=color, edgecolor="white", linewidth=0.5
        )
        ax.add_patch(rect)
        current += timedelta(days=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")

    # Day labels
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_yticks([6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels(day_labels, fontsize=8)

    # Month labels
    month_positions = {}
    current = start_date
    while current <= end_date:
        if current.day == 1 or current == start_date:
            week = (current - start_date).days // 7
            month_key = current.strftime("%b %Y")
            if month_key not in month_positions:
                month_positions[month_key] = week
        current += timedelta(days=1)

    ax.set_xticks(list(month_positions.values()))
    ax.set_xticklabels(list(month_positions.keys()), fontsize=8, rotation=45, ha="right")

    ax.set_title("Sky Images Captured Per Day", fontsize=12, pad=10)
    ax.tick_params(left=False, bottom=False)
    ax.spines[:].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Images/day", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")

    # Print summary stats
    total_images = sum(counts.values())
    days_with_data = len(counts)
    total_days = (end_date - start_date).days + 1
    completeness = days_with_data / total_days * 100 if total_days > 0 else 0

    print(f"\nDataset Statistics:")
    print(f"  Date range: {start_date} to {end_date} ({total_days} days)")
    print(f"  Days with data: {days_with_data} ({completeness:.1f}%)")
    print(f"  Total images: {total_images:,}")
    print(f"  Avg images/day (data days): {total_images / days_with_data:.0f}")
    print(f"  Max images/day: {max(counts.values())}")
    print(f"  Min images/day (data days): {min(counts.values())}")


def main():
    parser = argparse.ArgumentParser(description="Generate completeness heatmap")
    parser.add_argument("--image-dir", required=True, help="Root image directory")
    parser.add_argument("--output", default="completeness_heatmap.pdf", help="Output file path")
    args = parser.parse_args()

    counts = count_images_per_day(args.image_dir)
    plot_calendar_heatmap(counts, args.output)


if __name__ == "__main__":
    main()
