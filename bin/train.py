"""Train classical & boosted models on the processed splits."""

import time
import joblib
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import settings, logger
from src.metrics import append_comparison_rows, compute_metrics
from src.schemas import Splits, Models, ModelResults
from src.constants import MODEL_ALIASES
from src.data import load_splits


def retrieve_models(models: str) -> Models:
    seed = settings.random_seed

    match models.lower():
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
                        ("scaler", StandardScaler()),
                        ("clf", DecisionTreeClassifier(random_state=seed)),
                    ]
                ),
            )
        case "rf":
            return Models(
                name="random_forest",
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=200,
                                n_jobs=-1,
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
                f"Invalid model: {models}. Available models: lr, dt, rf, gb, all"
            )


def train_validate_test(models: Models, splits: Splits) -> ModelResults:
    logger.info(f"Training {models.name} ...")
    start = time.perf_counter()
    models.model.fit(splits.X_train, splits.y_train)
    train_seconds = time.perf_counter() - start

    def score(split_name: str, X, y) -> dict:
        metrics = compute_metrics(models.model, X, y, split_name)
        metrics["model"] = models.name
        metrics["train_seconds"] = train_seconds
        logger.info(
            f"  {split_name}: f1={metrics['f1']:.4f} recall={metrics['recall']:.4f}"
        )
        return metrics

    return ModelResults(
        metrics_train=score("train", splits.X_train, splits.y_train),
        metrics_valid=score("valid", splits.X_valid, splits.y_valid),
        metrics_test=score("test", splits.X_test, splits.y_test),
    )


def save_models(models: Models) -> None:
    settings.base_model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.base_model_dir / f"{models.name}.joblib"
    joblib.dump(models.model, path)
    logger.info(f"Saved {path}")


def main(models: str) -> None:
    logger.info(f"{'-'*10} Training {models} model(s) started {'-'*10}")

    retrieved_models = []
    if models.lower() == "all":
        retrieved_models.extend([retrieve_models(model) for model in MODEL_ALIASES])
    else:
        retrieved_models.append(retrieve_models(models.lower()))
    splits = load_splits()

    results: list[dict] = []
    for model in tqdm(retrieved_models, desc="Training models"):
        results.extend(train_validate_test(model, splits).metric_rows())
        save_models(model)

    append_comparison_rows(results)

    logger.info(f"{'-'*10} Training {models} model(s) completed {'-'*10}")


if __name__ == "__main__":
    """This is the argument parser for the train.py script."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to train (lr, dt, rf, gb), or 'all' for every baseline.",
    )
    args = parser.parse_args()
    main(args.model)
