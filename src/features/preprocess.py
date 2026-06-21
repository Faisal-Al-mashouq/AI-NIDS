"""Feature engineering: build the binary label column."""

import pandas as pd

from src.config import settings


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Benign -> 0, all attacks -> 1"""

    df[settings.label_col] = (
        df[settings.raw_label_col].astype(str).str.strip() != settings.benign_value
    ).astype(int)
    return df
