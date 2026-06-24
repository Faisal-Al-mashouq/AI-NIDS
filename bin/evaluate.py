"""Evaluate already-saved models on the processed splits."""

from argparse import ArgumentParser

from src.models.evaluate import evaluate_all


def main(models: list[str]) -> None:
    evaluate_all(models)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names, e.g. logistic_regression xgboost",
    )
    args = parser.parse_args()
    main(args.models)
