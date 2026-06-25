METRIC_CSV_FIELDS = [
    "model",
    "target",
    "split",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "inference_seconds",
    "train_seconds",
]

MODEL_ALIASES: tuple[str, ...] = ("lr", "dt", "rf", "gb", "xgb", "mlp", "ens")

SPLIT_FILES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}
