"""Feature selection: keep numeric features plus the label."""

import numpy as np
import pandas as pd

from src.config import settings


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if settings.label_col not in feature_cols:
        feature_cols.append(settings.label_col)
    return df[feature_cols]
