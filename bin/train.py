"""Train classical & boosted models on the processed splits."""

from argparse import ArgumentParser

from src.config import logger
from src.constants import MODEL_ALIASES
from src.training import train_all


def main(models: str) -> None:
    logger.info(f"{'-'*50} Training {models} model(s) started {'-'*50}")
    train_all(models)
    logger.info(f"{'-'*10} Training {models} model(s) completed {'-'*10}")


if __name__ == "__main__":
    """This is the argument parser for the train.py script."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model name to train {MODEL_ALIASES}, or 'all' for every baseline.",
    )
    args = parser.parse_args()
    main(args.model)
