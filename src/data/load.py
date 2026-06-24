"""Loading: raw CSV merge and processed split loading."""

import pandas as pd

from src.config import settings, logger
from src.constants import SPLIT_FILES
from src.schemas import Splits


def load_and_merge() -> pd.DataFrame:
    """Read and concatenate every raw CSV into one DataFrame."""
    raw_glob = settings.raw_dir / settings.dataset_name
    files = sorted(raw_glob.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_glob}")

    frames = []
    for f in files:
        logger.info(f"Reading: {f.name}")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged shape: {df.shape}")

    return df


def load_split(name: str) -> pd.DataFrame:
    """Load one processed split. name: 'train | 'valid' | 'test'."""
    return pd.read_csv(settings.processed_dir / SPLIT_FILES[name])


def load_splits() -> Splits:
    """Load train/valid/test CSVs and split each into X / y."""
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")

    return Splits(
        X_train=train_df.drop(columns=[settings.label_col]),
        y_train=train_df[settings.label_col],
        X_valid=valid_df.drop(columns=[settings.label_col]),
        y_valid=valid_df[settings.label_col],
        X_test=test_df.drop(columns=[settings.label_col]),
        y_test=test_df[settings.label_col],
    )


def feature_names() -> pd.Index:
    """Column names from train.csv (excluding label)."""
    return (
        pd.read_csv(settings.processed_dir / "train.csv", nrows=1)
        .drop(columns=[settings.label_col])
        .columns
    )
