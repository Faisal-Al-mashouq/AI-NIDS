"""Splitting: stratified train/valid/test split and CSV writing."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings, logger
from src.constants import SPLIT_FILES


def split_and_save(df: pd.DataFrame) -> dict:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    y = df[settings.label_col]
    X = df.drop(columns=[settings.label_col])

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_seed,
    )

    rel_valid = settings.valid_size / (1.0 - settings.test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_valid,
        stratify=y_tmp,
        random_state=settings.random_seed,
    )

    parts = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }
    counts = {}
    for name, (Xp, yp) in parts.items():
        out = Xp.copy()
        out[settings.label_col] = yp.values
        path = settings.processed_dir / SPLIT_FILES[name]
        out.to_csv(path, index=False)
        class_counts = yp.value_counts().sort_index()
        counts[name] = {
            "rows": int(len(out)),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        }
        if set(class_counts.index).issubset({0, 1}):
            counts[name]["attack"] = int(class_counts.get(1, 0))
            counts[name]["benign"] = int(class_counts.get(0, 0))

        logger.info(f"Saved {path} ({len(out)} rows)")

    return counts
