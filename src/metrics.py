"""Shared scoring helpers used by training/evaluatation."""

import csv
import time

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .config import settings, logger
from .constants import METRIC_CSV_FIELDS
from .utils.target import current_target_tag


def append_comparison_rows(rows: list[dict], path: Path = settings.metrics_dir) -> Path:
    """Append rows to model_comparison.csv. Write header only if file is new."""
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "model_comparison.csv"
    _ensure_metric_schema(csv_path)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_CSV_FIELDS)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_CSV_FIELDS})

    logger.info(f"Appended {len(rows)} rows to {path / "model_comparison.csv"}")
    return csv_path


def _ensure_metric_schema(csv_path: Path) -> None:
    """Upgrade older metrics CSVs when new columns are added."""
    if not csv_path.exists():
        return

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        old_fields = reader.fieldnames or []

    if old_fields == METRIC_CSV_FIELDS:
        return

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_CSV_FIELDS})


def compute_metrics(model, X, y, split_name: str) -> dict:
    """Computes metrics of a model."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    infer_time = time.perf_counter() - start

    labels = sorted(np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)])))
    is_binary = set(labels).issubset({0, 1})
    average = "binary" if is_binary else "macro"

    roc = None
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(X)[:, 1]
            if is_binary:
                roc = float(roc_auc_score(y, probability))
            else:
                roc = float(roc_auc_score(y, model.predict_proba(X), multi_class="ovr"))
        except Exception:
            roc = None

    matrix = confusion_matrix(y, y_pred, labels=labels)
    if is_binary:
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0
    else:
        total = matrix.sum()
        fprs = []
        fnrs = []
        for i in range(len(labels)):
            tp_i = matrix[i, i]
            fp_i = matrix[:, i].sum() - tp_i
            fn_i = matrix[i, :].sum() - tp_i
            tn_i = total - tp_i - fp_i - fn_i
            fprs.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) else 0)
            fnrs.append(fn_i / (fn_i + tp_i) if (fn_i + tp_i) else 0)
        tn = fp = fn = tp = ""
        fpr = float(np.mean(fprs))
        fnr = float(np.mean(fnrs))

    precision = float(precision_score(y, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y, y_pred, average=average, zero_division=0))
    inference_seconds = float(infer_time)

    return {
        "split": split_name,
        "target": current_target_tag(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc,
        "tn": int(tn) if is_binary else "",
        "fp": int(fp) if is_binary else "",
        "fn": int(fn) if is_binary else "",
        "tp": int(tp) if is_binary else "",
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
