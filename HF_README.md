---
license: cc-by-4.0
task_categories:
  - image-classification
  - time-series-forecasting
  - image-to-image
tags:
  - solar-energy
  - irradiance-forecasting
  - sky-imagery
  - meteorological-data
  - NREL
  - SRRL
  - deep-learning
  - computer-vision
  - climate
  - renewable-energy
language:
  - en
size_categories:
  - 100K-1M
pretty_name: NREL SRRL Minute-Resolution Sky Imagery
configs:
  - config_name: images
    data_files:
      - split: train
        path: "20*/**/*.jpg"
  - config_name: meteorological
    data_files:
      - split: train
        path: "meteorological/**/*.csv"
---

# NREL SRRL Minute-Resolution Sky Imagery Dataset

## Dataset Description

- **Homepage**: https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery
- **Repository**: https://github.com/LINK_TBD
- **Paper**: Hammond & Korgel (2026), *Journal of Data-centric Machine Learning Research*
- **Point of Contact**: Joshua E. Hammond (jeh5975@utexas.edu)

## Dataset Summary

A continuously growing dataset of **minute-resolution sky images** from the EKO ASI-16 all-sky imager at the National Renewable Energy Laboratory's Solar Radiation Research Laboratory (SRRL) in Golden, Colorado (39.742°N, 105.180°W, 1829 m elevation). While the standard SRRL gallery saves images only every 10 minutes, this dataset captures the camera's native **one-minute output**, providing 10× higher temporal frequency for training deep learning models for intra-hour solar irradiance forecasting.

Each image is temporally aligned with the SRRL Baseline Measurement System (BMS) one-minute meteorological records, which include Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI), Diffuse Horizontal Irradiance (DHI), and over 130 additional atmospheric variables.

The dataset currently contains **120,000+ images** across 170+ collection days and grows daily via an automated pipeline running on Indiana University's Jetstream2 cloud infrastructure.

## Key Features

| Attribute | Value |
|-----------|-------|
| **Image Resolution** | 1920 × 1920 pixels |
| **Temporal Frequency** | 1 minute |
| **Image Format** | JPEG (HDR) |
| **Camera** | EKO ASI-16 All-Sky Imager |
| **Location** | NREL SRRL, Golden, CO, USA (39.742°N, 105.180°W, 1829 m) |
| **Collection Start** | January 2025 |
| **Status** | Continuously growing (updated daily) |
| **Ground Truth** | Paired SRRL BMS 1-minute meteorological data (130+ variables) |
| **Images per Day** | ~720 (daytime hours only) |
| **Daily Growth** | ~200–500 MB |

## Dataset Structure

```
NREL_Sky_Imagery/
├── 2025/
│   ├── 01/
│   │   ├── 15/
│   │   │   ├── 20250115-080033.jpg
│   │   │   ├── 20250115-080133.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── 2026/
│   └── ...
├── meteorological/
│   ├── 2025/
│   │   ├── 01/
│   │   │   ├── 20250115.csv
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── README.md
```

### Image Naming Convention

Images are named as `YYYYMMDD-HHMMSS.jpg` in Mountain Standard Time (MST, UTC−7).
Example: `20260101-143233.jpg` = January 1, 2026 at 14:32:33 MST.

Images are organized hierarchically as `YYYY/MM/DD/` to keep each directory well under HuggingFace's file-count limits.

### Meteorological Data

Each CSV in `meteorological/` contains one day of SRRL Baseline Measurement System (BMS) data at 1-minute resolution. Key variables include:

| Category | Variables |
|----------|-----------|
| **Solar Radiation** | Global Horizontal Irradiance (GHI, CMP22), Direct Normal Irradiance (DNI, CHP1), Diffuse Horizontal Irradiance (DHI) |
| **Solar Geometry** | Zenith angle, Azimuth angle, Air mass |
| **Temperature** | Dry bulb, Dew point |
| **Atmospheric** | Barometric pressure, Relative humidity |
| **Wind** | Speed and direction (at 6 ft and 19 ft) |
| **Cloud** | Total cloud cover (%), Opaque cloud cover (%) |
| **Other** | Precipitation, UV (global/direct), Longwave (up/down), Albedo |

Full BMS documentation and variable list: https://midcdmz.nrel.gov/srrl_bms/

## Data Collection

### Pipeline Architecture

Images are scraped every 60 seconds from the NREL MIDC real-time ASI-16 image feed using an automated Python pipeline running on Indiana University's Jetstream2 cloud computing infrastructure (m3.tiny VM, 1 vCPU, 3 GB RAM).

**Real-time scraping** (`scraper.py`):
- Polls `midcdmz.nrel.gov/data/rt/srrlasi.jpg` every 60 seconds
- Estimates capture timestamp as `(current_time − 1 minute)` with seconds set to `:33` based on observed camera timing
- Filters duplicate frames via MD5 hash comparison with previous image
- Filters nighttime placeholder images via known blank-image MD5 hash (`604c77fd179dd033f129c4397a8095eb`)
- Saves unique daytime images to a 1 TB Cinder persistent volume

**Daily sync** (`daily_sync.py`):
- Runs daily at 03:00 MST via systemd timer
- Finds all image directories not yet uploaded to HuggingFace
- Deduplicates images within each day via SHA256 hash
- Uploads each day as a single atomic HuggingFace commit
- Downloads the previous day's BMS meteorological data from the NREL MIDC API
- Uploads meteorological CSVs to HuggingFace

**Reliability**:
- Scraper runs as a systemd service with `Restart=always` (auto-restarts on crash)
- Daily sync uses `Persistent=true` timer (catches up after any downtime)
- Pipeline is idempotent — safe to re-run without creating duplicates

### Data Provenance

- **Sky images**: Captured by the EKO ASI-16 all-sky imager operated by NREL at the SRRL facility. Images are served publicly via the NREL Measurement and Instrumentation Data Center (MIDC).
- **Meteorological data**: Recorded by the SRRL Baseline Measurement System (BMS), one of the most comprehensive solar radiation measurement facilities in the world, operating since 1981.

## Quick Start

### Download the Dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="knl2366/NREL_Sky_Imagery",
    repo_type="dataset",
    local_dir="./nrel_sky_imagery"
)
```

### Load with Hugging Face Datasets

```python
from datasets import load_dataset
ds = load_dataset("knl2366/NREL_Sky_Imagery")
```

### PyTorch DataLoader

A custom PyTorch `Dataset` class is provided for training forecasting models:

```python
from srrl_dataset import SRRLDataset
from torch.utils.data import DataLoader

dataset = SRRLDataset(
    image_dir="./nrel_sky_imagery",
    meteo_dir="./nrel_sky_imagery/meteorological",
    seq_length=5,          # 5 consecutive images as input
    forecast_horizon=15,   # predict 15 minutes ahead
    target_variable="Global CMP22 (vent/cor) [W/m^2]",
    image_size=(224, 224),
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, meteo_features, target_ghi in loader:
    # images: (batch, 5, 3, 224, 224)
    # meteo_features: (batch, 5, num_meteo_vars)
    # target_ghi: (batch,) - GHI in W/m^2 at t+15min
    pass
```

### Download Meteorological Data Directly

The BMS data can also be downloaded directly from the NREL MIDC API:

```python
import requests

url = "https://midcdmz.nrel.gov/apps/data_api.pl?site=BMS&begin=20250101&end=20250131"
response = requests.get(url)
with open("bms_jan2025.csv", "w") as f:
    f.write(response.text)
```

## Intended Uses

- Training and evaluating deep learning models for **intra-hour solar irradiance forecasting** (1–30 min horizons)
- Studying **cloud dynamics** and their impact on surface irradiance at sub-minute timescales
- **Benchmarking** forecasting methods across different temporal resolutions (1-min vs 10-min)
- **Transfer learning** studies using sky imagery from different geographic sites
- **Nowcasting** applications for grid stability and solar plant operations
- **Computer vision** research on cloud segmentation and motion estimation

## Comparison with Related Datasets

| Dataset | Resolution | Frequency | Irradiance | Meteo Vars | Period | Growing |
|---------|-----------|-----------|------------|------------|--------|---------|
| **This Dataset** | 1920×1920 | 1 min | GHI, DNI, DHI | 130+ | 2025– | Yes |
| SRRL Gallery | 1536×1536 | 10 min | GHI, DNI, DHI | 130+ | 2017– | Yes |
| SKIPP'D | 2048×2048 | 1 min | PV power only | None | 2017–19 | No |
| Folsom | ~1536×1536 | 1 min | GHI, DNI | Limited | 2014–16 | No |
| SIRTA | 768×1024 | 1–2 min | GHI, DHI, DNI | Yes | 2017–19 | No |

**Key differentiator**: This is the only publicly available dataset that combines minute-resolution sky imagery, high image resolution (1920×1920), comprehensive ground-truth meteorological measurements (130+ variables), and continuous daily growth.

## Known Limitations

- **Image gaps are permanent**: The NREL endpoint serves only the current frame. If the scraper is offline, those images cannot be recovered retroactively.
- **Nighttime hours excluded**: Images are filtered between approximately sunset and sunrise (varies seasonally). This is intentional — nighttime frames are blank placeholders with no information content.
- **Dome obstructions**: The ASI-16 camera dome may occasionally be obscured by moisture, frost, or debris, resulting in partially or fully obscured images that are still saved.
- **Meteorological data lag**: BMS data for a given day is only available from NREL after that day completes. The most recent 1–2 days of images may not yet have corresponding meteorological CSVs.
- **Timestamp estimation**: Image capture times are estimated (not embedded in metadata). The actual camera trigger time may differ by a few seconds from the recorded filename.
- **Single site**: All data is from one geographic location (Golden, CO, 1829 m elevation, semi-arid climate). Models trained on this data may not generalize to other climates without transfer learning.

## Ethical Considerations

- **Privacy**: Images are exclusively of the sky hemisphere. No personal data, faces, or identifiable information is captured.
- **Source**: All data originates from NREL, a US Department of Energy national laboratory with open-data policies. Sky images are served publicly via the MIDC without access restrictions.
- **Purpose**: The dataset is intended solely for renewable energy research and education. Accurate solar forecasting supports grid decarbonization and the global transition to clean energy.
- **Environmental impact**: Data collection uses minimal computing resources (1 vCPU cloud VM) and network bandwidth (~300 KB/min).

## Maintenance Plan

- **Daily updates**: New images and meteorological data are uploaded automatically every day at 03:00 MST.
- **Maintained by**: Joshua E. Hammond, McKetta Department of Chemical Engineering, The University of Texas at Austin.
- **Infrastructure**: Pipeline runs on NSF-funded Jetstream2 cloud (Indiana University). Allocation is sufficient for 45+ years of continuous operation.
- **Versioning**: Quarterly snapshot tags (e.g., `snapshot-2026Q1`) will be created so researchers can pin experiments to specific dataset versions.
- **Long-term preservation**: Data is stored on a persistent volume independent of the compute VM. The dataset will remain available on HuggingFace indefinitely.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{hammond2026srrl,
  title={A Minute-Resolution Sky Imagery Dataset from NREL's Solar Radiation
         Research Laboratory for Intra-Hour Solar Irradiance Forecasting},
  author={Hammond, Joshua E. and Korgel, Brian A.},
  journal={Journal of Data-centric Machine Learning Research},
  year={2026}
}
```

## License

This dataset is released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

- **Maintainer**: Joshua E. Hammond (jeh5975@utexas.edu)
- **PI**: Brian A. Korgel (korgel@che.utexas.edu)
- **Affiliation**: McKetta Department of Chemical Engineering, The University of Texas at Austin
- **GitHub**: [Collection and analysis scripts](https://github.com/LINK_TBD)
- **Paper**: [DMLR submission](https://openreview.net/forum?id=XXXX)
