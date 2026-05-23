# AI-NIDS

AI-NIDS is a Python project for building an AI-based Network Intrusion Detection System. The project is currently in its dataset setup stage: it can download and extract labeled network-flow CSV files that will later be used for training and evaluating intrusion detection models.

The first modeling target is binary classification:

- `BENIGN`
- `ATTACK`

Future work, including data processing, training, evaluation, and detection scripts, should be tracked in `PLAN.md`.

## Current Status

Implemented:

- Python project setup in `pyproject.toml`
- dependency locking with `uv.lock`
- formatting, linting, pre-commit, and test tasks via `taskipy`
- GitHub Actions CI workflow for pre-commit and tests
- dataset download and extraction script at `bin/download_dataset.py`
- downloaded raw dataset files under `data/raw/network-intrusion-dataset/`

Not implemented yet:

- data cleaning and preprocessing pipeline
- model training
- model evaluation
- detection/inference command
- automated tests

## Repository Layout

```text
AI-NIDS/
├── bin/
│   └── download_dataset.py
├── data/
│   └── raw/
│       └── network-intrusion-dataset/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   └── config.py
├── .gitignore
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

## Download The Dataset

Run:

```bash
uv run task dataset
```

This runs `bin/download_dataset.py`, which downloads and extracts the dataset into:

```text
data/raw/network-intrusion-dataset/
```

The current dataset source is:

```text
https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset
```

If Kaggle returns a login page or an invalid ZIP file, the downloader may need to be updated to use authenticated Kaggle API access.

## Dataset Files

The downloaded dataset currently contains CSV files for multiple traffic capture periods, including:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

These raw files should be treated as input data. Cleaned, merged, or split datasets should be written separately in a future processing step.

## Development Notes

`pyproject.toml` defines the project dependencies and developer tasks. Black is configured to target Python `3.12`, matching the local runtime. CI runs `uv run pre-commit run --all-files` and `uv run pytest` on pushes and pull requests targeting `main`.

Generated artifacts such as raw data, processed datasets, trained models, metrics, and figures should generally stay out of Git unless they are small and intentionally included for the course submission.
