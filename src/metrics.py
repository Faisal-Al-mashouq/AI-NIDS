"""Shared scoring helpers used by training/evaluatation."""

import time

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X, y, split_name: str) -> dict:
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
