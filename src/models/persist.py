"""Model persistence: save & load fitted pipelines."""

from pathlib import Path

import joblib

from src.config import settings, logger
from src.utils.target import current_target_tag


def model_filename(name: str, target: str | None = None) -> str:
    tag = target or current_target_tag()
    return f"{name}_{tag}.joblib"


def save_model(
    pipeline,
    name: str,
    *,
    target: str | None = None,
    dir: Path = settings.base_model_dir,
) -> Path:
    """Saves models with name & directory. Default path: base_model directory."""
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / model_filename(name, target)
    joblib.dump(pipeline, path)
    logger.info(f"Saved {path}")
    return path


def load_model(
    name: str,
    *,
    target: str | None = None,
    dir: Path = settings.base_model_dir,
):
    """Loads models by name & directory. Default path: base_model directory"""
    path = dir / model_filename(name, target)
    if not path.exists():
        legacy_path = dir / f"{name}.joblib"
        if legacy_path.exists():
            logger.info(f"Loading legacy untagged model: {legacy_path}")
            return joblib.load(legacy_path)
    return joblib.load(path)
