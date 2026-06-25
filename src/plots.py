from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / offline safe

import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings, logger
from src.data.load import feature_names
from src.models.persist import load_model


def plot_xgboost_importance(top_n: int = 20) -> Path:
    model = load_model("xgboost", dir=settings.base_model_dir)
    features = feature_names()

    clf = model.named_steps["clf"]
    importances = (
        pd.Series(clf.feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(top_n)
    )

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances[::-1].plot.barh(ax=ax)

    ax.set_title(f"XGBoost - Top {top_n} feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    out = settings.figures_dir / "xgboost_feature_importance.png"
    fig.savefig(out, dpi=120)
    logger.info(f"Saved {out}")
    return out


def plot_model_comparison() -> Path:
    metrics_path = settings.metrics_dir / "model_comparison.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file found at {metrics_path}")

    df = pd.read_csv(metrics_path)
    test_rows = df[df["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("No test split rows found in model_comparison.csv")

    latest = test_rows.groupby("model", as_index=False).tail(1)
    latest = latest.sort_values("f1", ascending=False)

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = settings.metrics_dir / "latest_test_comparison.csv"
    latest.to_csv(comparison_csv, index=False)
    logger.info(f"Saved {comparison_csv}")

    plot_cols = ["f1", "recall", "precision", "roc_auc"]
    plot_df = latest.set_index("model")[plot_cols]

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df.plot.bar(ax=ax)
    ax.set_title("Latest test-set model comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = settings.figures_dir / "model_comparison.png"
    fig.savefig(out, dpi=120)
    logger.info(f"Saved {out}")
    return out


if __name__ == "__main__":
    plot_xgboost_importance()
