"""Training loop: fit pipelines, score, persist, and record metrics."""

import time

from tqdm import tqdm

from src.config import logger
from src.data.load import load_splits
from src.metrics import append_comparison_rows, compute_metrics
from src.models.baseline import build_models
from src.models.persist import save_model
from src.schemas import Models, Splits, ModelResults


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
            f"  {split_name}: f1={metrics['f1']:.5f} recall={metrics['recall']:.5f}"
        )
        return metrics

    return ModelResults(
        metrics_train=score("train", splits.X_train, splits.y_train),
        metrics_valid=score("valid", splits.X_valid, splits.y_valid),
        metrics_test=score("test", splits.X_test, splits.y_test),
    )


def train_all(selection: list[str] | str) -> list[dict]:
    splits = load_splits()
    model_selection = " ".join(selection) if isinstance(selection, list) else selection

    rows: list[dict] = []
    for entry in tqdm(build_models(model_selection), desc="Training models"):
        result = train_one(entry, splits)
        save_model(entry.model, entry.name)
        model_rows = result.metric_rows()
        append_comparison_rows(model_rows)
        rows.extend(model_rows)

    return rows
