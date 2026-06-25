# AI-NIDS

AI-NIDS is a Python project for building an AI-based Network Intrusion Detection
System. It turns labeled network-flow CSV files into clean, model-ready datasets,
trains and evaluates several classical and boosted-tree models, and can classify
new flow records using a saved model.

The default modeling target is binary classification:

- `BENIGN` -> `0`
- `ATTACK` -> `1`

Multi-class attack-label classification is also supported with
`uv run task process --target multiclass`.

## Current Status

Implemented:

- Python project setup (`pyproject.toml`) with dependency locking (`uv.lock`)
- formatting, linting, pre-commit, and test tasks via `taskipy`
- GitHub Actions CI workflow for pre-commit and tests
- dataset download and extraction (`bin/download_dataset.py`)
- preprocessing pipeline: merge, clean, label, feature-select, and stratified
  train/validation/test split (`bin/process_data.py` -> `src/preprocessing.py`)
- model training for logistic regression, decision tree, random forest,
  gradient boosting, XGBoost, MLP neural network, and mixture-of-experts ensemble
  (`bin/train.py`)
- evaluation of already-saved models (`bin/evaluate.py`)
- XGBoost feature-importance plotting (`bin/plot_importance.py`)
- detection/inference on new flow CSVs (`bin/detect.py`)
- unit tests for preprocessing, training, and detection (`tests/`)

Not implemented yet (see `PLAN.md`):

- final project report

## Repository Layout

```text
AI-NIDS/
├── bin/                         # thin CLI entry points (argparse -> src)
│   ├── download_dataset.py
│   ├── process_data.py          # -> src.preprocessing.run
│   ├── train.py                 # -> src.models.train.train_all
│   ├── evaluate.py              # -> src.models.evaluate.evaluate_all
│   ├── plot_importance.py       # -> src.plots.plot_xgboost_importance
│   └── detect.py                # -> src.detection.*
├── src/
│   ├── config.py                # Settings, settings, logger
│   ├── constants.py             # metric fields, model aliases, split files
│   ├── schemas.py               # Splits, Models, ModelResults
│   ├── metrics.py               # compute_metrics, append_comparison_rows
│   ├── preprocessing.py         # pipeline composition root (run)
│   ├── plots.py                 # plot_xgboost_importance
│   ├── data/                    # load.py, clean.py, split.py
│   ├── features/                # preprocess.py (label), selection.py
│   ├── models/                  # baseline.py, train.py, evaluate.py, persist.py
│   ├── detection/               # classifier.py, alerts.py
│   └── utils/                   # paths.py, logging.py
├── configs/
│   └── model.yaml               # per-model hyperparameter overrides
├── data/
│   ├── raw/network-intrusion-dataset/   # downloaded source CSVs
│   └── processed/               # train.csv, validation.csv, test.csv
├── models/
│   ├── base/                    # trained baseline/boosted models (.joblib)
│   └── custom/
├── notebooks/                   # exploratory notebooks
├── reports/
│   ├── figures/                 # plots (e.g. xgboost_feature_importance.png)
│   └── metrics/                 # model_comparison.csv, preprocessing_summary.json
├── tests/
├── .python-version
├── PLAN.md
├── pyproject.toml
├── README.md
└── uv.lock
```

## Requirements

This project uses Python `3.12` and `uv` for dependency management.

Install dependencies:

```bash
uv sync
```

Install the local Git pre-commit hook once:

```bash
uv run pre-commit install
```

Run formatting, linting, pre-commit checks, and tests:

```bash
uv run task format
uv run task lint
uv run task precommit
uv run task test
```

## Pipeline

The end-to-end workflow is exposed as `taskipy` tasks (each runs the matching
`bin/` entry point as a module). Run them from the repository root.

### 1. Download the dataset

```bash
uv run task dataset
```

This runs `bin/download_dataset.py`, which downloads and extracts the dataset into
`data/raw/network-intrusion-dataset/`.

The current dataset source is:

```text
https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset
```

If Kaggle returns a login page or an invalid ZIP file, the downloader may need to
be updated to use authenticated Kaggle API access.

### 2. Process the data

```bash
uv run task process
```

Merges the raw CSVs, normalizes columns, drops duplicate/invalid rows, builds the
binary label, keeps numeric features, and writes a stratified split to
`data/processed/{train,valid,test}.csv` plus
`reports/metrics/preprocessing_summary.json`.

For multi-class attack-label classification, run:

```bash
uv run task process --target multiclass
```

This writes the same split files, but `label` contains integer attack-class IDs
instead of binary benign/attack values. The mapping is saved to
`reports/metrics/label_mapping.json`.

### 3. Train models

Train one or more models, or all of them. Aliases: `lr`, `dt`, `rf`, `gb`, `xgb`,
`mlp`, `ens`.

```bash
uv run task train --model all
uv run task train --model lr
uv run task train --model "rf xgb"
uv run task train --model mlp
uv run task train --model ens
```

Trained pipelines are saved under `models/base/` with the active target tag in
the filename, for example `xgboost_binary.joblib` or
`xgboost_multiclass.joblib`. Metrics are appended to
`reports/metrics/model_comparison.csv` with a matching `target` column.

Hyperparameters come from in-code defaults in `src/models/baseline.py`. Any value
set in `configs/model.yaml` (keyed by the model's canonical name) overrides the
matching default, so tuning a model is a config edit rather than a code change:

```yaml
random_forest:
  n_estimators: 400
  max_depth: 30
xgboost:
  learning_rate: 0.05
```

### 4. Evaluate saved models

Score already-trained models without retraining (canonical names):

```bash
uv run task evaluate --models logistic_regression xgboost
```

### 5. Plot feature importance

```bash
uv run task plot
```

Writes `reports/figures/xgboost_feature_importance.png`.

### 6. Detect on new flow records

```bash
uv run task detect --input data/processed/test.csv --model xgboost
```

Loads the saved model, classifies the input CSV, and writes attack rows to
`reports/detections.csv`.

## Metrics

Evaluation focuses on security-relevant metrics, not just accuracy. Each
train/validation/test row in `reports/metrics/model_comparison.csv` records:
precision, recall, F1, false positive rate, false negative rate, ROC-AUC, the
confusion-matrix counts (`tn`, `fp`, `fn`, `tp`), and inference/training time.

## To Contribute

Direct pushes to `main` are not allowed. All changes must go through a pull request
from a feature branch.

### Local setup

1. Fork the repository (if you are an external contributor) and clone your fork
   locally.
2. Install dependencies and the local pre-commit hook:

   ```bash
   uv sync
   uv run pre-commit install
   ```

3. Make your changes, then run the same checks used in CI:

   ```bash
   uv run task precommit
   uv run task test
   ```

   To auto-format before committing:

   ```bash
   uv run task format
   ```

4. Commit your changes. Pre-commit runs automatically on each commit; fix any
   reported issues and commit again if needed.

### Opening a pull request

1. Update your local `main` and create a branch for your work:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b your-branch-name
   ```

2. Push the branch to GitHub (never push directly to `main`):

   ```bash
   git push -u origin your-branch-name
   ```

3. Open a pull request targeting `main`:
   - On GitHub: **Compare & pull request** from your branch, or go to
     **Pull requests → New pull request**.
   - With the GitHub CLI:

     ```bash
     gh pr create --base main --head your-branch-name --title "Your PR title" --body "Brief summary of changes"
     ```

4. Wait for CI to finish. The required check is **`CI / lint-and-test`**, which runs
   pre-commit on all tracked files and the test suite (see `.github/workflows/ci.yml`).
5. Address any review feedback by pushing additional commits to the same branch;
   the PR updates automatically.
6. Once CI passes and the PR is approved, merge via GitHub. Do not merge by pushing
   to `main` locally.

For planned features and open work, see `PLAN.md`.

## Development Notes

- One import style: absolute `from src... import ...` everywhere. Run entry points
  as modules (`uv run -m bin.<script>`), which the `taskipy` tasks already do.
- `bin/` stays thin (argparse -> a single call into `src`). All business logic
  lives under `src/`.
- Cross-cutting modules (`config.py`, `constants.py`, `schemas.py`, `metrics.py`)
  stay at the top level of `src/`; domain code lives in `data/`, `features/`,
  `models/`, `detection/`, and `utils/`.
- `pyproject.toml` defines dependencies and developer tasks. Black targets Python
  `3.12`. CI runs `uv run pre-commit run --all-files` and `uv run pytest` on pushes
  and pull requests targeting `main`.
- Generated artifacts (raw data, processed datasets, trained models, metrics,
  figures) should generally stay out of Git unless they are small and intentionally
  included for the course submission.
```
