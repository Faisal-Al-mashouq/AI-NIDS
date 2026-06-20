"""Train classical & boosted models on the processed splits."""

import time
from argparse import ArgumentParser

from src.config import logger
from src.metrics import compute_metrics
from src.schemas import Splits, Models, ModelResults
from src.constants import MODEL_ALIASES
from src.training import train_all


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


def main(models: str) -> None:
    logger.info(f"{'-'*50} Training {models} model(s) started {'-'*50}")
    train_all(models)
    logger.info(f"{'-'*10} Training {models} model(s) completed {'-'*10}")


if __name__ == "__main__":
    """This is the argument parser for the train.py script."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model name to train {MODEL_ALIASES}, or 'all' for every baseline.",
    )
    args = parser.parse_args()
    main(args.model)
