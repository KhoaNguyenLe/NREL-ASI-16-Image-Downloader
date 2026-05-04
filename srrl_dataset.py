"""
PyTorch Dataset for SRRL minute-resolution sky imagery with paired
meteorological data from the NREL Baseline Measurement System.

Usage:
    from srrl_dataset import SRRLDataset
    from torch.utils.data import DataLoader

    dataset = SRRLDataset(
        image_dir="./nrel_sky_imagery",
        meteo_dir="./meteorological",
        seq_length=5,
        forecast_horizon=15,
        target_variable="Global CMP22 (vent/cor) [W/m^2]",
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, meteo_features, target in loader:
        # images: (batch, seq_length, C, H, W)
        # meteo_features: (batch, seq_length, num_meteo_vars)
        # target: (batch,) - GHI at t + forecast_horizon
        pass
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Default GHI column name in BMS data
DEFAULT_TARGET = "Global CMP22 (vent/cor) [W/m^2]"

# Meteorological feature columns (common BMS variables)
DEFAULT_METEO_COLS = [
    "Zenith Angle [degrees]",
    "Azimuth Angle [degrees]",
    "Tower Dry Bulb Temp [deg C]",
    "Tower RH [%]",
    "Avg Wind Speed @ 6ft [m/s]",
    "Avg Wind Direction @ 6ft [deg from N]",
    "Station Pressure [mBar]",
]


class SRRLDataset(Dataset):
    """PyTorch Dataset for SRRL sky imagery + BMS meteorological data.

    Each sample consists of:
        - A sequence of `seq_length` consecutive sky images
        - Corresponding meteorological features for each timestep
        - A target irradiance value at `forecast_horizon` minutes ahead

    Args:
        image_dir: Root directory containing images in YYYY/MM/DD/ structure.
        meteo_dir: Directory containing meteorological CSVs in YYYY/MM/ structure.
        seq_length: Number of consecutive images per sample (default: 5).
        forecast_horizon: Minutes ahead to forecast (default: 15).
        target_variable: BMS column name for the forecast target.
        meteo_columns: List of BMS column names to include as features.
            If None, uses DEFAULT_METEO_COLS.
        image_size: Resize images to this (H, W) tuple (default: (224, 224)).
        transform: Optional torchvision transform for images. If None, uses
            default normalization (ImageNet stats).
        min_ghi: Minimum GHI threshold to include a sample (filters nighttime).
    """

    def __init__(
        self,
        image_dir: str,
        meteo_dir: str,
        seq_length: int = 5,
        forecast_horizon: int = 15,
        target_variable: str = DEFAULT_TARGET,
        meteo_columns: list = None,
        image_size: tuple = (224, 224),
        transform=None,
        min_ghi: float = 10.0,
    ):
        self.image_dir = Path(image_dir)
        self.meteo_dir = Path(meteo_dir)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.target_variable = target_variable
        self.meteo_columns = meteo_columns or DEFAULT_METEO_COLS
        self.image_size = image_size
        self.min_ghi = min_ghi

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        # Build index of valid samples
        self.samples = self._build_index()

    def _parse_image_timestamp(self, filename: str) -> datetime:
        """Parse timestamp from image filename like '20250115-123433.jpg'."""
        stem = Path(filename).stem  # '20250115-123433'
        return datetime.strptime(stem, "%Y%m%d-%H%M%S")

    def _load_meteo_for_date(self, dt: datetime) -> pd.DataFrame:
        """Load meteorological CSV for a given date."""
        csv_path = self.meteo_dir / dt.strftime("%Y") / dt.strftime("%m") / f"{dt.strftime('%Y%m%d')}.csv"
        if not csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(csv_path)

        # Build datetime index from Year, DOY, Hour, Minute columns
        if "Year" in df.columns and "DOY" in df.columns:
            if "Hour" in df.columns and "Minute" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Year"].astype(int).astype(str) + "-" +
                    df["DOY"].astype(int).astype(str),
                    format="%Y-%j",
                ) + pd.to_timedelta(df["Hour"].astype(int), unit="h") + pd.to_timedelta(
                    df["Minute"].astype(int), unit="min"
                )
            elif "MST" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Year"].astype(int).astype(str) + "-" +
                    df["DOY"].astype(int).astype(str),
                    format="%Y-%j",
                ) + pd.to_timedelta(df["MST"].astype(str).str.zfill(4).str[:2].astype(int), unit="h") + pd.to_timedelta(
                    df["MST"].astype(str).str.zfill(4).str[2:].astype(int), unit="min"
                )
            df = df.set_index("datetime")

        return df

    def _build_index(self) -> list:
        """Build list of valid (image_paths, meteo_rows, target) tuples."""
        samples = []

        # Find all image files
        image_files = sorted(self.image_dir.glob("*/*/*/**.jpg"))
        if not image_files:
            print(f"[WARN] No images found in {self.image_dir}")
            return samples

        # Group by date
        images_by_date = {}
        for img_path in image_files:
            try:
                ts = self._parse_image_timestamp(img_path.name)
                date_key = ts.date()
                if date_key not in images_by_date:
                    images_by_date[date_key] = []
                images_by_date[date_key].append((ts, img_path))
            except ValueError:
                continue

        # For each date, build sequences
        for date_key in sorted(images_by_date.keys()):
            day_images = sorted(images_by_date[date_key], key=lambda x: x[0])
            meteo_df = self._load_meteo_for_date(datetime.combine(date_key, datetime.min.time()))

            if meteo_df.empty:
                continue

            # Build timestamp -> index mapping for images
            ts_to_idx = {ts: i for i, (ts, _) in enumerate(day_images)}

            for i in range(len(day_images) - self.seq_length - self.forecast_horizon):
                seq_timestamps = [day_images[i + j][0] for j in range(self.seq_length)]
                target_ts = seq_timestamps[-1] + timedelta(minutes=self.forecast_horizon)

                # Check consecutive minutes
                is_consecutive = all(
                    (seq_timestamps[j + 1] - seq_timestamps[j]).total_seconds() <= 90
                    for j in range(self.seq_length - 1)
                )
                if not is_consecutive:
                    continue

                # Check target timestamp exists in meteo data
                target_ts_rounded = target_ts.replace(second=0, microsecond=0)
                if target_ts_rounded not in meteo_df.index:
                    continue

                # Check min GHI
                target_row = meteo_df.loc[target_ts_rounded]
                if self.target_variable in target_row and target_row[self.target_variable] < self.min_ghi:
                    continue

                seq_paths = [day_images[i + j][1] for j in range(self.seq_length)]
                samples.append(
                    (seq_paths, seq_timestamps, target_ts_rounded, date_key)
                )

        print(f"Built {len(samples)} valid samples from {len(images_by_date)} days")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_paths, seq_timestamps, target_ts, date_key = self.samples[idx]

        # Load and transform images
        images = []
        for path in seq_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)  # (seq_length, C, H, W)

        # Load meteorological features
        meteo_df = self._load_meteo_for_date(
            datetime.combine(date_key, datetime.min.time())
        )
        meteo_features = []
        for ts in seq_timestamps:
            ts_rounded = ts.replace(second=0, microsecond=0)
            if ts_rounded in meteo_df.index:
                row = meteo_df.loc[ts_rounded]
                features = []
                for col in self.meteo_columns:
                    if col in row:
                        val = row[col]
                        features.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        features.append(0.0)
                meteo_features.append(features)
            else:
                meteo_features.append([0.0] * len(self.meteo_columns))

        meteo_features = torch.tensor(meteo_features, dtype=torch.float32)

        # Get target
        target_row = meteo_df.loc[target_ts]
        target = float(target_row[self.target_variable]) if self.target_variable in target_row else 0.0
        target = torch.tensor(target, dtype=torch.float32)

        return images, meteo_features, target
