import pandas as pd


from .schemas import Splits
from .constants import SPLIT_FILES
from .config import settings


def load_splits() -> Splits:
    train_df: pd.DataFrame = load_split("train")
    valid_df: pd.DataFrame = load_split("valid")
    test_df: pd.DataFrame = load_split("test")

    # train_df, valid_df, test_df = {load_split(_) for _ in SPLIT_FILES}

    return Splits(
        X_train=train_df.drop(columns=[settings.label_col]),
        y_train=train_df[settings.label_col],
        X_valid=valid_df.drop(columns=[settings.label_col]),
        y_valid=valid_df[settings.label_col],
        X_test=test_df.drop(columns=[settings.label_col]),
        y_test=test_df[settings.label_col],
    )


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """name: 'train' | 'valid' | 'test'"""
    return pd.read_csv(settings.processed_dir / SPLIT_FILES[name])


def feature_names() -> pd.Index:
    """Column names from train.csv (excluding label). Used by plot_importance."""
    return pd.Index(
        pd.read_csv(settings.processed_dir / "train.csv", nrows=1)
        .drop(columns=[settings.label_col])
        .columns
    )


if __name__ == "__main__":
    train, valid, test = load_splits()
    print(train.head(5))
    print(valid.head(5))
    print(test.head(5))

    # print(feature_names())
