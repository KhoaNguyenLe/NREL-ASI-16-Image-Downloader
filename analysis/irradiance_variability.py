"""
Plot GHI time series showing ramp events that occur on sub-10-minute timescales.
Produces Figure 4 for the DMLR paper.

Usage:
    python irradiance_variability.py --meteo-dir ./meteorological --output figures/ramp_events.pdf
    python irradiance_variability.py --csv bms_data.csv --output figures/ramp_events.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


GHI_COL = "Global CMP22 (vent/cor) [W/m^2]"
# Fallback column names
GHI_ALTERNATIVES = [
    "Global CMP22 (vent/cor) [W/m^2]",
    "Global PSP (vent/cor) [W/m^2]",
    "Global Horiz [W/m^2]",
]


def load_meteo_day(meteo_dir: str, year: int, month: int, day: int) -> pd.DataFrame:
    """Load a single day of BMS data."""
    csv_path = Path(meteo_dir) / f"{year:04d}" / f"{month:02d}" / f"{year:04d}{month:02d}{day:02d}.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Build datetime index
    if "Year" in df.columns and "DOY" in df.columns:
        if "Hour" in df.columns and "Minute" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["Year"].astype(int).astype(str) + "-" + df["DOY"].astype(int).astype(str),
                format="%Y-%j",
            ) + pd.to_timedelta(df["Hour"].astype(int), unit="h") + pd.to_timedelta(
                df["Minute"].astype(int), unit="min"
            )
        df = df.set_index("datetime")
    return df


def find_ghi_column(df: pd.DataFrame) -> str:
    """Find the GHI column in the dataframe."""
    for col in GHI_ALTERNATIVES:
        if col in df.columns:
            return col
    # Try partial match
    for col in df.columns:
        if "global" in col.lower() and "w/m" in col.lower():
            return col
    raise ValueError(f"No GHI column found. Available: {list(df.columns)}")


def find_ramp_days(meteo_dir: str, n_days: int = 3) -> list:
    """Find days with the largest GHI ramp events (most variability)."""
    root = Path(meteo_dir)
    day_variability = []

    for csv_path in sorted(root.glob("*/*/*.csv")):
        try:
            df = pd.read_csv(csv_path)
            ghi_col = find_ghi_column(df)
            ghi = df[ghi_col].dropna()

            if len(ghi) < 60:  # skip days with very little data
                continue

            # Calculate 1-minute ramp rate (absolute difference)
            ramps = ghi.diff().abs()
            max_ramp = ramps.max()
            std_ramp = ramps.std()

            # Extract date from filename
            stem = csv_path.stem  # e.g., '20250115'
            day_variability.append((stem, max_ramp, std_ramp, csv_path))
        except Exception:
            continue

    # Sort by ramp variability (std of ramp rate)
    day_variability.sort(key=lambda x: x[2], reverse=True)

    return day_variability[:n_days]


def plot_ramp_events(meteo_dir: str, output_path: str, days: list = None):
    """Plot GHI time series for days with high variability."""
    if days is None:
        ramp_days = find_ramp_days(meteo_dir, n_days=3)
        if not ramp_days:
            print("No suitable days found")
            return
        csv_paths = [d[3] for d in ramp_days]
        day_labels = [d[0] for d in ramp_days]
    else:
        csv_paths = days
        day_labels = [Path(d).stem for d in days]

    fig, axes = plt.subplots(len(csv_paths), 1, figsize=(10, 3 * len(csv_paths)), sharex=False)
    if len(csv_paths) == 1:
        axes = [axes]

    for ax, csv_path, label in zip(axes, csv_paths, day_labels):
        df = pd.read_csv(csv_path)
        ghi_col = find_ghi_column(df)

        # Build time axis
        if "Hour" in df.columns and "Minute" in df.columns:
            hours = df["Hour"] + df["Minute"] / 60.0
        elif "MST" in df.columns:
            mst = df["MST"].astype(str).str.zfill(4)
            hours = mst.str[:2].astype(float) + mst.str[2:].astype(float) / 60.0
        else:
            hours = np.arange(len(df)) / 60.0

        ghi = df[ghi_col].values

        # Filter to daytime (GHI > 0)
        mask = ghi > 0
        hours_day = hours[mask]
        ghi_day = ghi[mask]

        # Plot 1-minute data
        ax.plot(hours_day, ghi_day, "b-", linewidth=0.8, alpha=0.9, label="1-min GHI")

        # Overlay 10-minute subsampled
        subsample_idx = np.arange(0, len(ghi_day), 10)
        ax.plot(
            hours_day.iloc[subsample_idx] if hasattr(hours_day, "iloc") else hours_day[subsample_idx],
            ghi_day[subsample_idx],
            "ro-", markersize=4, linewidth=1.0, alpha=0.7, label="10-min samples"
        )

        # Add vertical lines at 10-min intervals for reference
        for t in np.arange(hours_day.min() if hasattr(hours_day, "min") else hours_day[0],
                          hours_day.max() if hasattr(hours_day, "max") else hours_day[-1],
                          10 / 60):
            ax.axvline(t, color="gray", alpha=0.15, linewidth=0.5)

        # Format date label
        try:
            date_str = f"{label[:4]}-{label[4:6]}-{label[6:8]}"
        except (IndexError, ValueError):
            date_str = label

        ax.set_ylabel("GHI (W/m²)", fontsize=10)
        ax.set_title(date_str, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(5, 19)
        ax.set_xlabel("Hour (MST)", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ramp event plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GHI variability and ramp events")
    parser.add_argument("--meteo-dir", required=True, help="Directory with meteorological CSVs")
    parser.add_argument("--output", default="ramp_events.pdf", help="Output file path")
    args = parser.parse_args()

    plot_ramp_events(args.meteo_dir, args.output)


if __name__ == "__main__":
    main()
