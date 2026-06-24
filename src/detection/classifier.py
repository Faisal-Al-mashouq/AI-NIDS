"""Detection: classify new flow records with a saved model."""

import pandas as pd

from src.config import settings
from src.models.persist import load_model


def classify(input_csv, model_name: str = "xgboost") -> pd.DataFrame:
    """Load a model, score an input CSV of flows, return predictions."""

    model = load_model(model_name)
    df = pd.read_csv(input_csv)
    features = df.drop(columns=[settings.label_col], errors="ignore")

    result = df.copy()
    result["prediction"] = model.predict(features)

    if hasattr(model, "predict_proba"):
        result["confidence"] = model.predict_proba(features)[:, 1]

    return result
