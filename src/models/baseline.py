"""Model registery: build sklearn / xgboost pipelines by CLI alias."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import settings
from src.constants import MODEL_ALIASES
from src.schemas import Models


def build_model(alias: str) -> Models:
    seed = settings.random_seed

    match alias.lower():
        case "lr":
            return Models(
                name="logistic_regression",
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, verbose=1)),
                    ]
                ),
            )
        case "dt":
            return Models(
                name="decision_tree",
                model=Pipeline(
                    [
                        ("clf", DecisionTreeClassifier(random_state=seed)),
                    ]
                ),
            )
        case "rf":
            return Models(
                name="random_forest",
                model=Pipeline(
                    [
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=200,
                                random_state=seed,
                                verbose=1,
                            ),
                        ),
                    ]
                ),
            )
        case "gb":
            return Models(
                name="gradient_boosting",
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            GradientBoostingClassifier(random_state=seed, verbose=1),
                        ),
                    ]
                ),
            )
        case "xgb":
            return Models(
                name="xgboost",
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
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
                ),
            )
        case _:
            raise ValueError(
                f"Invalid model: {alias}. Available models: {MODEL_ALIASES} or 'all."
            )


def build_models(selection: str) -> list[Models]:
    """selection: space-separated aliases, or 'all'."""
    if selection.lower() == "all":
        return [build_model(a) for a in MODEL_ALIASES]
    return [build_model(a) for a in selection.split()]


if __name__ == "__main__":
    selection = "dt lr xgb"
    # x: list[str] = string.split()

    # print([build_model(_).name for _ in x])
    # x = build_models(selection)
    # print(x)
    # print("-" * 100)
    # [print(_.name) for _ in x]

    # print(_load_model_config("logistic_regression"))
