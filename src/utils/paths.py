"""Filesystem layout. Single source of truth for project directories."""

from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]

raw_dir = repo_root / "data" / "raw"
processed_dir = repo_root / "data" / "processed"

model_dir = repo_root / "models"
base_model_dir = repo_root / "models" / "base"
custom_model_dir = repo_root / "models" / "custom"

configs_path = repo_root / "configs" / "model.yaml"

reports_dir = repo_root / "reports"
metrics_dir = repo_root / "reports" / "metrics"
figures_dir = repo_root / "reports" / "figures"
