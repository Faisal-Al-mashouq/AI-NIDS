"""Model persistence: save & load fitted pipelines."""

from pathlib import Path

import joblib

from src.config import settings, logger


def save_model(pipeline, name: str, *, dir: Path = settings.base_model_dir) -> Path:
    """Saves models with name & directory. Default path: base_model directory."""
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info(f"Saved {path}")
    return path


def load_model(name: str, *, dir: Path = settings.base_model_dir):
    """Loads models by name & directory. Default path: base_model directory"""
    path = dir / f"{name}.joblib"
    return joblib.load(path)
