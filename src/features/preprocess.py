"""Feature engineering: build model target columns."""

import pandas as pd

from src.config import settings


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Benign -> 0, all attacks -> 1"""

    df[settings.label_col] = (
        df[settings.raw_label_col].astype(str).str.strip() != settings.benign_value
    ).astype(int)
    return df


def make_multiclass_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map each raw label to an integer class. BENIGN is always class 0."""
    labels = df[settings.raw_label_col].astype(str).str.strip()
    attack_labels = sorted(
        label for label in labels.unique() if label != settings.benign_value
    )
    mapping = {
        settings.benign_value: 0,
        **{label: i + 1 for i, label in enumerate(attack_labels)},
    }
    df[settings.label_col] = labels.map(mapping).astype(int)
    return df, mapping
