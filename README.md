# SRRL Minute-Resolution Sky Imagery Dataset

A continuously growing, minute-resolution sky imagery dataset from NREL's Solar
Radiation Research Laboratory (SRRL), paired with comprehensive meteorological
ground-truth measurements. The dataset is hosted on
[HuggingFace](https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery) and
described in an accompanying DMLR paper.

## Dataset Overview

| Property | Value |
| -------- | ----- |
| **Source** | EKO ASI-16 all-sky imager at NREL SRRL, Golden, CO |
| **Image resolution** | 1920 x 1920 px JPEG |
| **Temporal resolution** | 1 minute (vs. 10 min in the public SRRL gallery) |
| **Ground truth** | SRRL BMS 1-min data: GHI, DNI, DHI + 130 meteorological variables |
| **Collection start** | January 2025, ongoing |
| **Naming convention** | `YYYY/MM/DD/YYYYMMDD-HHMMSS.jpg` (MST timestamps) |

## Quick Start

### Download the dataset

```bash
conda env create -f environment.yml
conda activate mrd
python download.py
```

Or with pip:

```bash
pip install -r requirements.txt
python download.py
```

### Use with PyTorch

```python
from srrl_dataset import SRRLDataset
from torch.utils.data import DataLoader

dataset = SRRLDataset(
    image_dir="Dataset/images",
    meteo_dir="Dataset/meteorological",
    sequence_length=5,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    images = batch["images"]   # (B, seq_len, C, H, W)
    meteo = batch["meteo"]     # (B, num_features)
    target = batch["target"]   # (B,) GHI in W/m2
    # ... train your model
```

See [dataloader_example.py](dataloader_example.py) for a complete working example.

## Repository Structure

```
.
|-- download.py                # Download dataset from HuggingFace
|-- srrl_dataset.py            # PyTorch Dataset class (images + meteo)
|-- dataloader_example.py      # Minimal training example
|-- HF_README.md               # HuggingFace dataset card
|-- environment.yml            # Conda environment spec
|-- requirements.txt           # pip dependencies
|-- pipeline/                  # Data collection & upload pipeline
|   |-- config.py              # Shared config (paths, constants, env vars)
|   |-- scraper.py             # Continuous image scraper (Jetstream2 service)
|   |-- daily_sync.py          # Daily upload: images + meteo to HF
|   |-- download_and_upload_meteo.py  # BMS meteorological data downloader
|   |-- deploy/                # Jetstream2 systemd configs & setup script
|   |-- README.md              # Full pipeline documentation
|-- analysis/                  # Paper figure generation & benchmarks
|   |-- benchmark_model.py     # CNN benchmark: 1-min vs 10-min prediction
|   |-- completeness_heatmap.py
|   |-- dataset_stats.py
|   |-- irradiance_variability.py
|-- manuscript/                # DMLR paper (LaTeX)
|   |-- paper.tex, references.bib, dmlr2e.sty
|   |-- figures/               # Generated figures
|-- Dataset/                   # Local data mirror (gitignored)
    |-- images/                # Sky images (YYYY/MM/DD/...)
    |-- meteorological/        # BMS CSVs (YYYY/MM/YYYYMMDD.csv)
```

## Pipeline Architecture

The scraping pipeline runs on [IU Jetstream2](https://jetstream-cloud.org/) (NSF-funded cloud):

1. **`scraper.py`** runs as a systemd service, polling the NREL ASI-16 real-time
   image every minute with MD5 dedup and blank-frame filtering.
2. **`daily_sync.py`** runs via a systemd timer at 03:00 MST, uploading the
   previous day's images and BMS meteorological data to HuggingFace in atomic
   daily commits.

See [pipeline/README.md](pipeline/README.md) for full deployment documentation.

## Download meteorological data

```bash
cd pipeline/
python download_and_upload_meteo.py \
    --start 2025-01-01 --end 2026-03-04 \
    --out-dir ../Dataset/meteorological
```

## Compile the paper

```bash
cd manuscript/
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Links

- [HuggingFace dataset](https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery)
- [NREL SRRL BMS](https://midcdmz.nrel.gov/srrl_bms/)
- [DMLR (target venue)](https://data.mlr.press/)

## License

MIT
