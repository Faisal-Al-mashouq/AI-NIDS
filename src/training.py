import time

from tqdm import tqdm

from src.data import load_splits
from src.io import save_model
from src.models.registery import build_models
from src.schemas import Models, Splits, ModelResults
from src.metrics import append_comparison_rows, compute_metrics
from src.config import logger


def train_one(entry: Models, splits: Splits) -> ModelResults:
    logger.info(f"Training {entry.name} ...")
    start = time.perf_counter()
    entry.model.fit(splits.X_train, splits.y_train)
    train_seconds = time.perf_counter() - start

    def score(split_name: str, X, y) -> dict:
        metrics = compute_metrics(entry.model, X, y, split_name)
        metrics["model"] = entry.name
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


def train_all(selection: str) -> list[dict]:
    splits = load_splits()

    rows = []
    for entry in tqdm(build_models(selection), desc="Training models"):
        result = train_one(entry, splits)
        rows.extend(result.metric_rows())
        save_model(entry.model, entry.name)

    append_comparison_rows(rows)
    return rows
