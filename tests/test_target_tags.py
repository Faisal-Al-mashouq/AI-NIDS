import json

from src.models.persist import model_filename
from src.utils.target import current_target_tag


def test_current_target_tag_reads_preprocessing_summary(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "preprocessing_summary.json").write_text(
        json.dumps({"target": "multiclass"})
    )

    monkeypatch.setattr("src.utils.target.settings.metrics_dir", metrics_dir)
    assert current_target_tag() == "multiclass"


def test_model_filename_includes_target_tag():
    assert model_filename("xgboost", "binary") == "xgboost_binary.joblib"
