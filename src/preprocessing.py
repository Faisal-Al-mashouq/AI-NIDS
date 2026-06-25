import json

from src.config import settings, logger
from src.data.load import load_and_merge
from src.data.clean import clean
from src.data.split import split_and_save
from src.features.preprocess import make_binary_label, make_multiclass_label
from src.features.selection import select_features


def run(target: str = "binary") -> None:
    logger.info(f"{'-'*10}  Preprocessing  {'-'*10}")
    df = load_and_merge()
    df = clean(df)
    label_mapping = None
    if target == "binary":
        df = make_binary_label(df)
    elif target == "multiclass":
        df, label_mapping = make_multiclass_label(df)
    else:
        raise ValueError("target must be 'binary' or 'multiclass'")
    df = select_features(df)

    counts = split_and_save(df)

    summary = {
        "target": target,
        "n_features": int(df.shape[1] - 1),
        "total_rows": int(sum(c["rows"] for c in counts.values())),
        "splits": counts,
        "random_seed": settings.random_seed,
    }
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = settings.metrics_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if label_mapping is not None:
        mapping_path = settings.metrics_dir / "label_mapping.json"
        mapping_path.write_text(json.dumps(label_mapping, indent=2))


if __name__ == "__main__":
    run()
