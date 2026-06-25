"""Helpers for tagging artifacts by target label mode."""

import json

from src.config import settings


def current_target_tag() -> str:
    """Return the active target tag from preprocessing metadata."""
    summary_path = settings.metrics_dir / "preprocessing_summary.json"
    if not summary_path.exists():
        return "unknown"

    summary = json.loads(summary_path.read_text())
    if "target" in summary:
        return str(summary["target"])

    splits = summary.get("splits", {})
    if any("attack" in split and "benign" in split for split in splits.values()):
        return "binary"
    return "unknown"
