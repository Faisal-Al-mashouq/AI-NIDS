"""Classify new flow records with a saved model and write alerts."""

from argparse import ArgumentParser

from src.config import settings
from src.detection.alerts import to_alerts
from src.detection.classifier import classify


def main(input_csv: str, model_name: str) -> None:
    predictions = classify(input_csv, model_name)
    alerts = to_alerts(predictions)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "detections.csv"
    alerts.to_csv(out, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input flow CSV.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        help="Saved model name.",
    )
    args = parser.parse_args()
    main(args.input, args.model)
