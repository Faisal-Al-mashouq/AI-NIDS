import pandas as pd
import pytest

from src.models.baseline import build_model, build_models
from src.models.train import train_one
from src.schemas import Splits


def test_build_model_returns_named_pipeline():
    entry = build_model("lr")
    assert entry.name == "logistic_regression"
    assert "clf" in entry.model.named_steps


def test_build_models_all_includes_xgboost():
    models = build_models("all")
    names = {m.name for m in models}
    assert "xgboost" in names
    assert len(models) == 5


def test_build_model_invalid_alias_raises():
    with pytest.raises(ValueError):
        build_model("nope")


def test_train_one_on_synthetic_data():
    X = pd.DataFrame({"f1": list(range(20)), "f2": list(range(20, 40))})
    y = pd.Series([0, 1] * 10)
    splits = Splits(X_train=X, y_train=y, X_valid=X, y_valid=y, X_test=X, y_test=y)
    result = train_one(build_model("dt"), splits)
    rows = result.metric_rows()
    assert len(rows) == 3
    assert all("f1" in r for r in rows)
