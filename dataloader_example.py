"""
Minimal example: loading the SRRL sky imagery dataset for ML training.

Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Download data: python download.py

This script demonstrates how to:
    - Create an SRRLDataset with image sequences + meteorological features
    - Wrap it in a DataLoader for batched training
    - Access images, meteorological ground truth, and targets
"""

import torch
from torch.utils.data import DataLoader
from srrl_dataset import SRRLDataset

# --- Configuration ---
IMAGE_DIR = "./Dataset/images"        # Downloaded sky images
METEO_DIR = "./Dataset/meteorological"  # Downloaded BMS CSVs
SEQUENCE_LENGTH = 5                    # Number of consecutive minutes per sample
BATCH_SIZE = 8
NUM_WORKERS = 2

# --- Create Dataset ---
dataset = SRRLDataset(
    image_dir=IMAGE_DIR,
    meteo_dir=METEO_DIR,
    sequence_length=SEQUENCE_LENGTH,
    # target_col="GHI_Wm2",           # Default target: Global Horizontal Irradiance
    # transform=None,                  # Uses ImageNet normalization by default
)

print(f"Dataset size: {len(dataset)} sequences")

# --- Create DataLoader ---
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# --- Iterate one batch ---
batch = next(iter(loader))
images = batch["images"]       # (B, seq_len, C, H, W) sky image sequence
meteo = batch["meteo"]         # (B, num_features) meteorological measurements
target = batch["target"]       # (B,) prediction target (e.g. GHI)

print(f"Image batch shape: {images.shape}")
print(f"Meteo features shape: {meteo.shape}")
print(f"Target shape: {target.shape}")
print(f"Target values (GHI W/m2): {target[:4].tolist()}")

# --- Training loop skeleton ---
# model = YourModel(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = torch.nn.MSELoss()
#
# for epoch in range(num_epochs):
#     for batch in loader:
#         images = batch["images"].to(device)
#         meteo = batch["meteo"].to(device)
#         target = batch["target"].to(device)
#
#         pred = model(images, meteo)
#         loss = criterion(pred.squeeze(), target)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
