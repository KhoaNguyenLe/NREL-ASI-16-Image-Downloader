# NREL Sky Imagery Dataset Downloader

This project provides a Python script to download all image files from the Hugging Face dataset:

knl2366/NREL_Sky_Imagery

The script preserves the full directory structure and safely skips files that already exist locally.

---

## 📦 Dataset Source

Hosted on Hugging Face:
https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery

The dataset contains time-organized sky imagery stored in the following structure:

YYYY/MM/DD/YYYYMMDD-HHMMSS.jpg

Example:

2026/02/11/20260211-120033.jpg

---

## Features

- Downloads all image files from the dataset
- Preserves folder hierarchy exactly as hosted
- Skips already-downloaded files
- Safe to rerun (idempotent)
- Supports private datasets via Hugging Face token

---

## Installation

### 1. Clone this repository

bash
git clone https://github.com/KhoaNguyenLe/NREL-ASI-16-Image-Downloader.git
cd <repo-directory>

### 2. Create virtual environment
bash
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
bash
pip install -r requirements.txt
