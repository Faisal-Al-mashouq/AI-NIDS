"""Splitting: stratified train/valid/test split and CSV writing"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings, logger


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
        path = settings.processed_dir / f"{name}.csv"
        out.to_csv(path, index=False)
        counts[name] = {
            "rows": int(len(out)),
            "attack": int(yp.sum()),
            "benign": int((yp == 0).sum()),
        }

        logger.info(f"Saved {path} ({len(out)} rows)")

    return counts
