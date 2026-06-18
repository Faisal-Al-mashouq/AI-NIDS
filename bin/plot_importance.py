"""Plot XGBoost feature importance."""

import joblib
import matplotlib

matplotlib.use("Agg")  # headless / offline safe
import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings, logger


def main() -> None:
    model = joblib.load(settings.base_model_dir / "xgboost.joblib")
    feature_names = (
        pd.read_csv(settings.processed_dir / "train.csv", nrows=1)
        .drop(columns=[settings.label_col])
        .columns
    )

    clf = model.named_steps["clf"]
    importances = (
        pd.Series(clf.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(20)
    )

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    importances[::-1].plot.barh(ax=ax)
    ax.set_title("XGBoost - Top 20 feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out = settings.figures_dir / "xgboost_feature_importance.png"
    fig.savefig(out, dpi=120)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
