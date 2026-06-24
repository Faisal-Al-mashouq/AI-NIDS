"""Application settings and environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

from src.utils import paths
from src.utils.logging import logger

__all__ = ["settings", "logger"]


class Settings(BaseSettings):
    # Directories (see src/utils/paths.py)
    raw_dir: Path = paths.raw_dir
    processed_dir: Path = paths.processed_dir

    model_dir: Path = paths.model_dir
    base_model_dir: Path = paths.base_model_dir
    custom_model_dir: Path = paths.custom_model_dir

    configs_dir: Path = paths.configs_path

    reports_dir: Path = paths.reports_dir
    metrics_dir: Path = paths.metrics_dir
    figures_dir: Path = paths.figures_dir

    # Reproducibility
    random_seed: int = 42

    # Split ratios (train / valid / test). valid + test share the holdout.
    test_size: float = 0.15
    valid_size: float = 0.15

    # Column / label conventions
    raw_label_col: str = "Label"
    label_col: str = "label"  # binary 0/1 column
    benign_value: str = "BENIGN"

    dataset_url: str = (
        "https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset"
    )
    dataset_name: str = "network-intrusion-dataset"


settings = Settings()
