"""Evaluation: score already-saved models without retraining."""

from src.config import logger
from src.data.load import load_splits
from src.metrics import append_comparison_rows, compute_metrics
from src.models.persist import load_model
from src.schemas import Splits
from src.utils.target import current_target_tag


def evaluate_one(name: str, splits: Splits) -> list[dict]:
    """Load a saved model by name and score it on all splits."""
    model = load_model(name, target=current_target_tag())
    rows = []
    for split_name, X, y in [
        ("train", splits.X_train, splits.y_train),
        ("valid", splits.X_valid, splits.y_valid),
        ("test", splits.X_test, splits.y_test),
    ]:
        metrics = compute_metrics(model, X, y, split_name)
        metrics["model"] = name
        rows.append(metrics)

    return rows


def evaluate_all(names: list[str]) -> list[dict]:
    splits = load_splits()
    rows: list[dict] = []
    for name in names:
        logger.info(f"Evaluating {name} ...")
        rows.extend(evaluate_one(name, splits))

    append_comparison_rows(rows)
    return rows
