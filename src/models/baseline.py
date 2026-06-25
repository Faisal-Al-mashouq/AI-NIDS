"""Model registry: build sklearn / xgboost pipelines by CLI alias.

Default hyperparameters live here. Any value in ``configs/model.yaml`` (keyed by
the model's canonical name) overrides the matching default, so tuning a model is
a config edit rather than a code change.
"""

from functools import lru_cache

import yaml
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import settings
from src.constants import MODEL_ALIASES
from src.models.moe import MixtureOfExpertsClassifier
from src.schemas import Models


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Read configs/model.yaml once. Missing/empty file -> no overrides."""
    path = settings.configs_dir
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _params(name: str, defaults: dict) -> dict:
    """Merge YAML overrides for `name` on top of in-code defaults."""
    return {**defaults, **_load_config().get(name, {})}


def build_model(alias: str) -> Models:
    seed = settings.random_seed

    match alias.lower():
        case "lr":
            name = "logistic_regression"
            params = _params(name, {"max_iter": 1000, "verbose": 1})
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(**params)),
                    ]
                ),
            )
        case "dt":
            name = "decision_tree"
            params = _params(name, {"random_state": seed})
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("clf", DecisionTreeClassifier(**params)),
                    ]
                ),
            )
        case "rf":
            name = "random_forest"
            params = _params(
                name,
                {"n_estimators": 200, "random_state": seed, "verbose": 1, "n_jobs": -1},
            )
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("clf", RandomForestClassifier(**params)),
                    ]
                ),
            )
        case "gb":
            name = "gradient_boosting"
            params = _params(name, {"random_state": seed, "verbose": 1})
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", GradientBoostingClassifier(**params)),
                    ]
                ),
            )
        case "xgb":
            name = "xgboost"
            params = _params(
                name,
                {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "tree_method": "hist",
                    "random_state": seed,
                    "eval_metric": "logloss",
                },
            )
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", XGBClassifier(**params)),
                    ]
                ),
            )
        case "mlp":
            name = "mlp_neural_network"
            params = _params(
                name,
                {
                    "hidden_layer_sizes": (128, 64),
                    "activation": "relu",
                    "early_stopping": True,
                    "max_iter": 50,
                    "random_state": seed,
                    "verbose": 1,
                },
            )
            return Models(
                name=name,
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", MLPClassifier(**params)),
                    ]
                ),
            )
        case "ens":
            name = "mixture_of_experts"
            params = _params(name, {"gate_size": 0.25, "random_state": seed})
            estimators = [
                (
                    "lr",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000)),
                        ]
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        tree_method="hist",
                        random_state=seed,
                        eval_metric="logloss",
                    ),
                ),
            ]
            return Models(
                name=name,
                model=Pipeline(
                    [
                        (
                            "clf",
                            MixtureOfExpertsClassifier(experts=estimators, **params),
                        ),
                    ]
                ),
            )
        case _:
            raise ValueError(
                f"Invalid model: {alias}. Available aliases: {MODEL_ALIASES} or 'all'."
            )


def build_models(selection: str) -> list[Models]:
    """selection: space-separated aliases, or 'all'."""
    if selection.lower() == "all":
        return [build_model(a) for a in MODEL_ALIASES]
    return [build_model(a) for a in selection.split()]
