# AI-NIDS Project Plan

This document tracks the work for building AI-NIDS into a reproducible offline
machine-learning pipeline for network intrusion detection. It compares several
modeling approaches — classical ML, boosted trees, and (planned) deep learning and
ensembles — on labeled network-flow CSV data.

For how to run the pipeline, see `README.md`.

## Primary Objective

Build a reproducible system that classifies labeled network-flow records as benign
or malicious, then compares multiple model families to identify which approach
works best for intrusion detection.

**Current modeling target (implemented):**

- Binary classification: `BENIGN` → `0`, `ATTACK` → `1`

**Planned extension (not implemented):**

- Multi-class classification across specific attack labels (DDoS, PortScan, Web
  Attack, Infiltration, Bot, and other dataset labels)

## Success Criteria

| Deliverable | Status |
|-------------|--------|
| Repeatable dataset preprocessing pipeline | Done |
| Saved train/validation/test datasets | Done |
| Trained models under `models/base/` | Done (5 model families) |
| Evaluation metrics under `reports/metrics/` | Done |
| Plots under `reports/figures/` | Done (class distribution, XGBoost importance) |
| Consistent model comparison metrics | Done (`model_comparison.csv`) |
| Detection script for new flow records | Done (`bin/detect.py`) |
| Unit tests and CI | Done |
| Deep learning models | Not started |
| Ensemble methods | Not started |
| Final project report | Not started |

Evaluation focuses on security-relevant metrics, not only accuracy:

- precision, recall, F1-score
- false positive rate, false negative rate
- confusion matrix counts (`tn`, `fp`, `fn`, `tp`)
- ROC-AUC
- training and inference time

All metrics are recorded in `reports/metrics/model_comparison.csv` via
`src/metrics.py`.

---

## Current Status (June 2026)

### Implemented

**Project infrastructure**

- Python `3.12` project with `uv` dependency locking (`pyproject.toml`, `uv.lock`)
- Formatting, linting, pre-commit, and test tasks via `taskipy`
- GitHub Actions CI (`.github/workflows/ci.yml`) — `CI / lint-and-test` on pushes
  and PRs to `main`
- Hyperparameter overrides in `configs/model.yaml` (merged into in-code defaults
  in `src/models/baseline.py`)

**Repository layout**

- Thin CLI entry points in `bin/` (argparse → single call into `src/`)
- Domain-oriented library code in `src/`:
  - `data/` — load, clean, split
  - `features/` — binary labeling, feature selection
  - `models/` — baseline definitions, train, evaluate, persist
  - `detection/` — classify new flows, write alerts
  - `utils/` — paths, logging
- Cross-cutting modules at `src/` root: `config.py`, `constants.py`, `schemas.py`,
  `metrics.py`, `preprocessing.py`, `plots.py`

**Pipeline scripts** (each exposed as a `taskipy` task; see `README.md`)

| Task | Script | Purpose |
|------|--------|---------|
| `dataset` | `bin/download_dataset.py` | Download and extract raw CSVs |
| `process` | `bin/process_data.py` | Merge, clean, label, split → `data/processed/` |
| `train` | `bin/train.py` | Train one or more models → `models/base/` |
| `evaluate` | `bin/evaluate.py` | Score saved models without retraining |
| `plot` | `bin/plot_importance.py` | XGBoost feature-importance figure |
| `detect` | `bin/detect.py` | Classify new flow CSVs → `reports/detections.csv` |

**Trained model families** (aliases: `lr`, `dt`, `rf`, `gb`, `xgb`)

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (scikit-learn)
- XGBoost

Artifacts are saved as `models/base/<canonical_name>.joblib`.

**Tests**

- `tests/test_preprocess.py` — cleaning, labeling, feature selection
- `tests/test_training.py` — model training and artifact persistence
- `tests/test_detection.py` — classification and alert output

**Exploratory work**

- `notebooks/01_data_exploration.ipynb` — Phase 1 data understanding; produces
  `reports/metrics/dataset_summary.json` and
  `reports/figures/class_distribution.png`

### Not implemented yet

- Multi-class attack classification
- Deep learning models (MLP, autoencoder, etc.)
- Ensemble methods (voting, stacking, weighted blends)
- Optional boosted-tree variants (LightGBM, CatBoost)
- Final project report / course write-up

---

## Phase Tracker

### Phase 0: Project Setup — Done

- `uv` project, locked dependencies, `taskipy` tasks
- Black, Ruff, pre-commit, pytest
- GitHub Actions CI
- Dataset downloader (`bin/download_dataset.py`)

Raw data source:

```text
https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset
```

### Phase 1: Data Understanding — Done

**Goal:** Inspect raw CSVs and document the dataset before modeling.

**Completed via:** `notebooks/01_data_exploration.ipynb`

**Outputs:**

- `reports/metrics/dataset_summary.json`
- `reports/figures/class_distribution.png`

### Phase 2: Data Preprocessing — Done

**Goal:** Turn raw CSV files into clean, model-ready datasets.

**Implementation:** `src/preprocessing.py` (orchestration) with `src/data/*` and
`src/features/*`; invoked by `uv run task process`.

**Outputs:**

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `reports/metrics/preprocessing_summary.json`

### Phase 3: Baseline Models — Done

**Goal:** Establish simple baseline results before advanced methods.

**Models:** Logistic Regression, Decision Tree, Random Forest, scikit-learn
Gradient Boosting.

**Implementation:** `src/models/baseline.py`, `src/models/train.py`;
`uv run task train --model lr|dt|rf|gb`.

**Outputs:**

- `models/base/logistic_regression.joblib`
- `models/base/decision_tree.joblib`
- `models/base/random_forest.joblib`
- `models/base/gradient_boosting.joblib`
- Rows appended to `reports/metrics/model_comparison.csv`

**Evaluation:** `uv run task evaluate --models <canonical names>` via
`src/models/evaluate.py`.

### Phase 4: Boosted Tree Models — Done (core scope)

**Goal:** Test stronger tree-boosting methods on tabular flow features.

**Implemented:** XGBoost (`xgb` alias); feature-importance plot via
`uv run task plot` → `reports/figures/xgboost_feature_importance.png`.

**Output:** `models/base/xgboost.joblib`

**Deferred (optional):** LightGBM, CatBoost — add only if time allows and gains
justify the extra dependencies.

### Phase 5: Deep Learning Models — Not started

**Goal:** Test whether neural networks improve detection on processed flow
features.

**Planned approaches:**

- Multi-Layer Perceptron for tabular features
- Optional autoencoder-based anomaly detection
- Optional sequence/CNN model only if the representation supports it

**Planned actions:**

- Choose PyTorch or TensorFlow before implementation
- Scale numeric features consistently; use early stopping
- Save model under `models/custom/` (or a dedicated subdirectory)
- Add training-curve plot under `reports/figures/`
- Append metrics to `model_comparison.csv`

**Dependency note:** Add the DL framework only when this phase begins to avoid
lockfile conflicts with other branches.

### Phase 6: Ensemble Methods — Not started

**Goal:** Combine strong models and test whether an ensemble improves
robustness.

**Planned approaches:**

- Soft/hard voting ensembles
- Stacking classifier
- Weighted ensemble using validation-set performance

**Prerequisite:** Best classical, boosted, and (if built) deep learning models
from earlier phases.

**Expected outputs:**

- `models/custom/ensemble.joblib` (or similar)
- Final comparison in `reports/metrics/model_comparison.csv`
- Confusion-matrix plot for the selected final model

### Phase 7: Detection Script — Done

**Goal:** Load a saved model and classify new network-flow records.

**Implementation:** `bin/detect.py` → `src/detection/classifier.py`,
`src/detection/alerts.py`.

**Example:**

```bash
uv run task detect --input data/processed/test.csv --model xgboost
```

**Output:** `reports/detections.csv` (attack rows with predictions)

### Phase 8: Testing and Quality — Done (core scope)

**Goal:** Keep the project reproducible and safe to change.

**Completed:**

- Unit tests for preprocessing, training, and detection (`tests/`)
- `uv run task lint`, `uv run task precommit`, `uv run task test`
- CI runs pre-commit and pytest on every PR to `main`

**Ongoing:** Expand test coverage as new model families and ensembles land.

### Phase 9: Final Report — Not started

**Goal:** Summarize the project for course submission.

**Planned contents:**

- Dataset source and preprocessing decisions
- Per-model-family comparison using the same metrics
- Discussion of false positives and false negatives
- Best-model selection with justification
- Confusion matrices and key plots from `reports/figures/`
- Limitations and future work (multi-class labels, ensembles, etc.)

---

## Repository Layout

```text
AI-NIDS/
├── bin/                         # thin CLI entry points
│   ├── download_dataset.py
│   ├── process_data.py          # → src.preprocessing.run
│   ├── train.py                 # → src.models.train.train_all
│   ├── evaluate.py              # → src.models.evaluate.evaluate_all
│   ├── plot_importance.py       # → src.plots.plot_xgboost_importance
│   └── detect.py                # → src.detection.*
├── src/
│   ├── config.py                # Settings, settings, logger
│   ├── constants.py             # metric fields, model aliases, split files
│   ├── schemas.py               # Splits, Models, ModelResults
│   ├── metrics.py               # compute_metrics, append_comparison_rows
│   ├── preprocessing.py         # pipeline composition root (run)
│   ├── plots.py                 # plot_xgboost_importance
│   ├── data/                    # load.py, clean.py, split.py
│   ├── features/                # preprocess.py, selection.py
│   ├── models/                  # baseline.py, train.py, evaluate.py, persist.py
│   ├── detection/               # classifier.py, alerts.py
│   └── utils/                   # paths.py, logging.py
├── configs/
│   └── model.yaml               # per-model hyperparameter overrides
├── data/
│   ├── raw/network-intrusion-dataset/
│   └── processed/               # train.csv, validation.csv, test.csv
├── models/
│   ├── base/                    # trained baseline/boosted models (.joblib)
│   └── custom/                  # future: DL, ensembles
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
├── .github/workflows/ci.yml
├── PLAN.md
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Suggested Next Steps

1. **Deep learning baseline** (Phase 5) — MLP on the existing processed splits
2. **Ensemble** (Phase 6) — combine best classical, boosted, and DL models
3. **Final report** (Phase 9) — write-up using `model_comparison.csv` and figures
4. **Optional:** multi-class labeling, LightGBM/CatBoost, additional notebooks

---

## Notes

- Keep raw data unchanged under `data/raw/`.
- Save generated datasets under `data/processed/`.
- Save trained artifacts under `models/base/` (classical/boosted) or
  `models/custom/` (future DL/ensembles).
- Save metrics and plots under `reports/`.
- Use the same train/validation/test split for all model families.
- Run entry points as modules (`uv run -m bin.<script>`); `taskipy` tasks wrap
  this already.
- Generated artifacts (raw data, processed CSVs, models, metrics) should stay
  out of Git unless small and intentionally included for submission.

---

## Task Split Between Contributors

Contributors:

- **Faisal:** classical and boosted models plus pipeline plumbing
- **Khalid:** evaluation, deep learning, detection, and testing

Phases 1–4, 7, and 8 (core) are complete. Remaining work is concentrated in
Phases 5, 6, and 9.

### Completed ownership (summary)

| Area | Owner | Status |
|------|-------|--------|
| Preprocessing pipeline (`bin/process_data.py`, `src/preprocessing.py`) | Faisal | Done |
| Baseline + boosted training (`bin/train.py`, `src/models/*`) | Faisal | Done |
| XGBoost feature-importance plot | Faisal | Done |
| Data exploration notebook + summary artifacts | Khalid | Done |
| Evaluation harness (`bin/evaluate.py`, `src/models/evaluate.py`) | Khalid | Done |
| Detection script (`bin/detect.py`, `src/detection/*`) | Khalid | Done |
| Unit tests + CI | Khalid | Done |

### Remaining work

| Phase | Owner | Notes |
|-------|-------|-------|
| Phase 5 — Deep learning | Khalid | Add DL dependency in a dedicated PR |
| Phase 6 — Ensemble | Both | Needs best models from Phases 3–5 |
| Phase 9 — Final report | Both | Split by section; one owner for comparison table |

### Workflow

- All changes go through a pull request from a feature branch; direct pushes to
  `main` are blocked.
- Required check: `CI / lint-and-test` (pre-commit + pytest).
- Add new dependencies per phase (e.g., PyTorch/TensorFlow only in Phase 5) to
  reduce `uv.lock` conflicts.
