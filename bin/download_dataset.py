#!/usr/bin/env python3

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset"
)
DATASET_NAME = "network-intrusion-dataset"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    zip_path = raw_dir / f"{DATASET_NAME}.zip"
    extract_dir = raw_dir / DATASET_NAME

    raw_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading dataset to {zip_path}...")
        urlretrieve(DATASET_URL, zip_path)
    else:
        print(f"Dataset zip already exists: {zip_path}")

    if not extract_dir.exists():
        print(f"Extracting dataset to {extract_dir}...")
        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)
    else:
        print(f"Dataset already extracted: {extract_dir}")


if __name__ == "__main__":
    main()
