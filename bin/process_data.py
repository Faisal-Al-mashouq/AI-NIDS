#!/usr/bin/env python3
"""Merge, clean, label, and split raw dataset."""

from src.config import settings, logger

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_merge() -> pd.DataFrame:
    raw_glob = settings.raw_dir / settings.dataset_name
    files = sorted(raw_glob.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_glob}")

    frames = []
    for f in files:
        logger.info(f"Reading: {f.name}")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged shape: {df.shape}")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN/inf")

    return df


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    "Benign is assigned 0, All Attacks are set as 1."
    raw = settings.raw_label_col
    df[settings.label_col] = (
        df[raw].astype(str).str.strip() != settings.benign_value
    ).astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    "Keep only numeric features and labels, discard all else."
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if settings.label_col not in feature_cols:
        feature_cols.append(settings.label_col)
    return df[feature_cols]


def split_and_save(df: pd.DataFrame) -> dict:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    y = df[settings.label_col]
    X = df.drop(columns=settings.label_col)

    # Split off the test set.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_seed,
    )

    # Split validation out of the remainder
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
        "validation": (X_valid, y_valid),
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
        logger.info(f"Saved {path} ({len(out)}) rows")

    return counts


def main() -> None:
    logger.info(f"{'-'*10}  Preprocessing  {'-'*10}")
    df = load_and_merge()
    df = clean(df)
    df = make_binary_label(df)
    df = select_features(df)

    counts = split_and_save(df)

    summary = {
        "n_features": int(df.shape[1] - 1),
        "total_rows": int(sum(c["rows"] for c in counts.values())),
        "splits": counts,
        "random_seed": settings.random_seed,
    }
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = settings.metrics_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
