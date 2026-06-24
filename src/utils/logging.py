"""Logging configuration and the shared application logger."""

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]")
    return logging.getLogger()


logger = configure_logging()
