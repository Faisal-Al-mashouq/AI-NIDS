"""This file is used to store the configuration of the application, in addition to the environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings
import logging

repo_root = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Directories
    raw_dir: Path = repo_root / "data" / "raw"
    processed_dir: Path = repo_root / "data" / "processed"
    model_dir: Path = repo_root / "models"
    base_model_dir: Path = repo_root / "models" / "base"
    custom_model_dir: Path = repo_root / "models" / "custom"
    reports_dir: Path = repo_root / "reports"
    metrics_dir: Path = repo_root / "reports" / "metrics"
    figures_dir: Path = repo_root / "reports" / "figures"

    # Reproducibility
    random_seed: int = 42

    # Split ratios (train / valid / test). valid + test share the holdout
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
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
)
logger = logging.getLogger()
