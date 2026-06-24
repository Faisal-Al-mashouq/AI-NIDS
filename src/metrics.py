"""Shared scoring helpers used by training/evaluatation."""

import csv
import time

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .config import settings, logger
from .constants import METRIC_CSV_FIELDS


def append_comparison_rows(rows: list[dict], path: Path = settings.metrics_dir) -> Path:
    """Append rows to model_comparison.csv. Write header only if file is new."""
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "model_comparison.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_CSV_FIELDS)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_CSV_FIELDS})

    logger.info(f"Appended {len(rows)} rows to {path / "model_comparison.csv"}")
    return csv_path


def compute_metrics(model, X, y, split_name: str) -> dict:
    """Computes metrics of a model."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    infer_time = time.perf_counter() - start

    # Probability for ROC-AUC
    roc = None
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(X)[:, 1]
            roc = float(roc_auc_score(y, probability))
        except Exception:
            roc = None

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    precision = float(precision_score(y, y_pred, zero_division=0))
    recall = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    inference_seconds = float(infer_time)

    return {
        "split": split_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "inference_seconds": inference_seconds,
    }


if __name__ == "__main__":
    print(
        append_comparison_rows(
            [
                {
                    "model": "neural_network",
                    "split": "train",
                    "precision": 0.8815713956773216,
                    "recall": 0.860740422590506,
                    "f1": 0.8710313822165175,
                    "false_positive_rate": 0.023497499896013675,
                    "false_negative_rate": 0.13925957740949402,
                    "roc_auc": 0.9880190524718292,
                }
            ]
        )
    )
