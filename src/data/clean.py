"""Cleaning: normalize columns, deduplicate, drop invalid rows."""

import numpy as np
import pandas as pd

from src.config import logger


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN/inf")

    return df
