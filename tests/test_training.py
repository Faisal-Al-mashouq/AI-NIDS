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
    assert "mlp_neural_network" in names
    assert "mixture_of_experts" in names
    assert len(models) == 7


def test_build_models_accepts_deep_learning_and_ensemble_aliases():
    models = build_models("mlp ens")
    names = [m.name for m in models]
    assert names == ["mlp_neural_network", "mixture_of_experts"]


def test_train_all_accepts_cli_model_list(monkeypatch):
    import src.models.train as train_module

    captured = {}
    monkeypatch.setattr(train_module, "load_splits", lambda: None)
    monkeypatch.setattr(train_module, "append_comparison_rows", lambda rows: None)

    def fake_build_models(selection):
        captured["selection"] = selection
        return []

    monkeypatch.setattr(train_module, "build_models", fake_build_models)
    assert train_module.train_all(["lr", "xgb"]) == []
    assert captured["selection"] == "lr xgb"


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
