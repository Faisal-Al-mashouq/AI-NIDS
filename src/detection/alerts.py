"""Detection: turn predictions into alerts."""

import pandas as pd


def to_alerts(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows predicted as attack (1)."""
    return predictions[predictions["predicition"] == 1].copy()
