#!/usr/bin/env python3

from urllib.request import urlretrieve
from zipfile import ZipFile
from src.config import settings, logger


def main() -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = settings.raw_dir / f"{settings.dataset_name}.zip"
    extract_dir = settings.raw_dir / settings.dataset_name

    if not zip_path.exists():
        logger.info(f"Downloading dataset to {zip_path}...")
        urlretrieve(settings.dataset_url, zip_path)
    else:
        logger.info(f"Dataset zip already exists: {zip_path}")

    if not extract_dir.exists():
        logger.info(f"Extracting dataset to {extract_dir}...")
        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)
    else:
        logger.info(f"Dataset already extracted: {extract_dir}")


if __name__ == "__main__":
    main()
