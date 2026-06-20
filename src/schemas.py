import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class Models:
    name: str
    model: Pipeline


@dataclass
class ModelResults:
    metrics_train: dict
    metrics_valid: dict
    metrics_test: dict

    def metric_rows(self) -> list[dict]:
        return [self.metrics_train, self.metrics_valid, self.metrics_test]
