# AI-NIDS Project Plan

This document tracks the planned work for building AI-NIDS into an offline machine learning pipeline for network intrusion detection. The project will compare several modeling approaches, including classical machine learning, boosted trees, deep learning, and ensemble methods.

## Primary Objective

Build a reproducible system that can classify labeled network-flow records as either benign or malicious, then compare multiple model families to identify which approach performs best for intrusion detection.

Initial task:

- Binary classification: `BENIGN` vs `ATTACK`

Later extension:

- Multi-class classification across specific attack labels such as DDoS, PortScan, Web Attack, Infiltration, Bot, and other available dataset labels.

## Success Criteria

The project should produce:

- a repeatable dataset preprocessing pipeline
- saved train/test datasets
- trained models saved under `models/`
- evaluation metrics saved under `reports/metrics/`
- plots saved under `reports/figures/`
- a final model comparison using consistent metrics
- a detection script that can classify new flow records using a saved model

Evaluation should focus on security-relevant metrics, not only accuracy:

- precision
- recall
- F1-score
- false positive rate
- false negative rate
- confusion matrix
- ROC-AUC or PR-AUC where useful
- training and inference time

## Phase 1: Data Understanding

Goal: inspect the downloaded CSV files and understand the dataset before modeling.

Planned actions:

- list all raw CSV files under `data/raw/network-intrusion-dataset/`
- inspect column names, row counts, data types, and label values
- identify the label column
- check for missing values, infinite values, duplicate rows, and invalid numeric values
- inspect class imbalance between benign and attack traffic
- document the dataset shape and label distribution

Expected outputs:

- `reports/metrics/dataset_summary.json`
- `reports/figures/class_distribution.png`

## Phase 2: Data Preprocessing

Goal: turn the raw CSV files into clean model-ready datasets.

Planned actions:

- merge the raw CSV files into one dataframe
- normalize column names
- remove duplicate rows
- replace infinite values with missing values
- handle missing values consistently
- remove columns that leak labels or are not useful for modeling
- convert labels into binary classes: `BENIGN` and `ATTACK`
- split data into train, validation, and test sets
- preserve class distribution with stratified splitting
- save processed datasets under `data/processed/`

Expected scripts:

- `bin/process_data.py`

Expected outputs:

- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `reports/metrics/preprocessing_summary.json`

## Phase 3: Baseline Models

Goal: create simple baseline results before testing more advanced methods.

Planned models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting with scikit-learn

Planned actions:

- build a reusable training pipeline with preprocessing and model fitting
- standardize numeric features where needed
- save each trained model with a clear name and timestamp or version
- evaluate every model on the same validation and test splits
- record metrics in a comparable format

Expected scripts:

- `bin/train.py`
- `bin/evaluate.py`

Expected outputs:

- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/gradient_boosting.joblib`
- `reports/metrics/model_comparison.csv`

## Phase 4: Boosted Tree Models

Goal: test stronger tree boosting methods that often perform well on tabular network-flow data.

Planned models:

- XGBoost classifier
- optional LightGBM classifier if time allows
- optional CatBoost classifier if categorical handling becomes useful

Planned actions:

- add the selected boosting dependency only when this phase begins
- train boosted tree models on the same processed train/validation/test split
- tune a small set of high-impact hyperparameters
- compare boosted tree results against the baseline models
- measure whether performance gains justify added dependency and complexity

Expected outputs:

- `models/xgboost.joblib`
- updated `reports/metrics/model_comparison.csv`
- feature importance plot under `reports/figures/`

## Phase 5: Deep Learning Models

Goal: test whether neural networks improve detection performance on the processed flow features.

Planned approaches:

- Multi-Layer Perceptron classifier for tabular features
- optional autoencoder-based anomaly detection
- optional 1D CNN or sequence-style model only if the data representation supports it

Planned actions:

- decide between PyTorch and TensorFlow before implementation
- scale numeric features consistently
- create a small neural network baseline before adding complexity
- track training loss and validation metrics
- use early stopping to avoid overfitting
- compare deep learning results against tree-based models

Expected outputs:

- saved neural network model under `models/`
- training curve plot under `reports/figures/`
- updated `reports/metrics/model_comparison.csv`

## Phase 6: Ensemble Methods

Goal: combine strong models and test whether an ensemble improves robustness.

Planned approaches:

- soft voting ensemble
- hard voting ensemble
- stacking classifier
- weighted ensemble using validation-set performance

Planned actions:

- choose the best-performing classical, boosted, and deep learning models
- combine model predictions on the validation set
- tune ensemble weights if using weighted voting
- evaluate the final ensemble on the held-out test set only after selection
- compare ensemble performance against the best single model

Expected outputs:

- `models/ensemble.joblib`
- final comparison metrics in `reports/metrics/model_comparison.csv`
- confusion matrix plot for the selected final model

## Phase 7: Detection Script

Goal: provide a command that loads a saved model and classifies new network-flow records.

Planned actions:

- create `bin/detect.py`
- accept an input CSV path
- load the selected saved model
- apply the same preprocessing used during training
- output predictions and confidence scores where available
- save detection results under `reports/`

Expected command:

```bash
uv run python bin/detect.py --input data/processed/test.csv --model models/ensemble.joblib
```

Expected outputs:

- `reports/detections.csv`

## Phase 8: Testing And Quality

Goal: keep the project reproducible and safe to change.

Planned actions:

- add unit tests for preprocessing helpers
- add tests for label conversion
- add tests that verify train/validation/test split outputs exist
- add tests that verify model training saves an artifact
- add tests that verify evaluation writes metrics
- keep formatting and linting passing with `uv run task lint`

Expected scripts:

- `uv run pytest`

## Phase 9: Final Report

Goal: summarize the project clearly for course submission.

Planned actions:

- explain dataset source and preprocessing decisions
- compare each model family using the same metrics
- discuss false positives and false negatives
- identify the best model and explain why it was selected
- include confusion matrices and important plots
- document limitations and possible future work

Expected outputs:

- final metrics table
- final plots under `reports/figures/`
- short project write-up or report section

## Suggested Implementation Order

1. Data inspection and summary
2. Preprocessing pipeline
3. Baseline scikit-learn models
4. Random Forest tuning
5. XGBoost or another boosted tree model
6. Deep learning baseline
7. Ensemble model
8. Detection script
9. Tests and final report

## Notes

- Keep raw data unchanged under `data/raw/`.
- Save generated datasets under `data/processed/`.
- Save trained model artifacts under `models/`.
- Save metrics and plots under `reports/`.
- Use the same train/validation/test split for all model families.
- Avoid comparing models that were trained or evaluated on different data splits.
- Prefer simple, reproducible experiments before adding complex model architectures.

## Task Split Between Contributors

Contributors:

- **Faisal** : classical and boosted models plus pipeline plumbing
- **Khalid** : evaluation, deep learning, detection, and testing

The entire plan depends on one hard constraint from the Notes: every model family
must use the same train/validation/test split. Phases 1 and 2 are therefore a shared
foundation that must land before the modeling work forks.

### Step 0: Shared Foundation

Build the shared backbone first, since everything downstream depends on it:

- the preprocessing pipeline (`bin/process_data.py`) producing
  `data/processed/{train,valid,test}.csv`
- a shared interface contract: the metrics schema for
  `reports/metrics/model_comparison.csv`, model artifact naming, and how
  `bin/train.py` / `bin/evaluate.py` load data

Pragmatic parallel start: Faisal owns Phases 1-2, while Khalid builds the evaluation
harness (`bin/evaluate.py` plus metrics/plotting utilities) against a small mock split.
Both pieces are prerequisites and are naturally separable.

### Faisal : Classical & Boosted Models + Pipeline Plumbing

- Phase 2: Preprocessing pipeline (`bin/process_data.py`)
- Phase 3: Baseline models (Logistic Regression, Decision Tree, Random Forest,
  scikit-learn Gradient Boosting) plus `bin/train.py`
- Phase 4: Boosted trees (XGBoost, optional LightGBM/CatBoost) plus the feature
  importance plot

### Khalid : Evaluation, Deep Learning, Detection & Testing

- Phase 1: Data understanding plus `dataset_summary.json` and `class_distribution.png`
- Shared `bin/evaluate.py` plus metrics/plotting utilities (the reusable scoring code
  both tracks consume)
- Phase 5: Deep learning (MLP, optional autoencoder/CNN)
- Phase 7: Detection script (`bin/detect.py`)
- Phase 8: Testing and quality (pytest, lint)

### Converge

- Phase 6: Ensemble - done together, since it consumes the best classical, boosted, and
  deep models from both tracks. Whoever has lighter load drives it.
- Phase 9: Final report - split by section. Each contributor writes up the models they
  built; one person owns the final comparison table and cross-model discussion.

### Why This Division Works

- **Balanced load.** Faisal owns more model families (Phases 3-4) plus preprocessing;
  Khalid owns deep learning plus the surrounding infrastructure (evaluation harness,
  detection, tests). Roughly even.
- **Clean ownership of the shared contract.** Putting `bin/evaluate.py`, the metrics
  schema, and tests on one person (Khalid) prevents two people from defining
  `model_comparison.csv` differently, which would break the whole comparison.
- **Low merge conflict surface.** Faisal touches `bin/train.py` plus classical/boosted
  training; Khalid touches `bin/evaluate.py`, `bin/detect.py`, and `tests/`. They mostly
  meet at the metrics CSV (append-only) and `src/config.py`.
- **Respects the dependency chain.** The suggested implementation order is preserved;
  only the independent branches run in parallel.

### Workflow Notes

All changes go through a pull request from a feature branch; direct pushes to `main` are
blocked and `CI / lint-and-test` must pass. With two contributors:

- Each person works on their own feature branch and reviews the other's PRs.
- Land the shared preprocessing and evaluation-contract PRs first, then rebase the model
  PRs on top to avoid fighting over `src/config.py`, `pyproject.toml`, and the metrics
  schema.
- Add new dependencies per phase (e.g., XGBoost only in Phase 4, PyTorch/TensorFlow only
  in Phase 5) so the two branches do not both edit `pyproject.toml` / `uv.lock` at the
  same time and create lockfile conflicts.
