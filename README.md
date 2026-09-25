# Explainable Fraud Risk Engine

A portfolio project demonstrating production-style Python, fraud-risk modelling, explainable machine learning, and a future agentic AI extension.

## Phase 0: Repository foundation
This phase creates the initial project scaffold, packaging, documentation, and CI to support future fraud modelling work. No data ingestion or model training is included yet.

## Getting started
Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install .[dev]
```

Run lint and tests:

```bash
ruff check src tests
pytest
```

## Phase 1: Data ingestion and validation
This phase adds safe PaySim data loading and schema validation, including checks for required columns, numeric values, fraud labels restricted to 0 or 1, and warnings for zero-amount transactions. It also documents the dataset expectations and separates fatal errors from non-fatal data-quality warnings.

## Phase 2: Exploratory data analysis
This phase adds reusable exploratory-analysis functions for dataset structure, fraud prevalence, fraud by transaction type, transaction amounts, balances, identifiers, and the existing `isFlaggedFraud` benchmark. Figures are saved under `reports/eda/figures/`.

Run EDA against an explicitly supplied PaySim CSV:

```bash
python scripts/run_eda.py data/raw/paysim_transactions.csv
```

No modelling, feature engineering, resampling, or train/test splitting occurs during Phase 2. The raw PaySim dataset is not committed to this repository.

## Phase 3: Leakage-aware baseline feature engineering
Phase 3 creates a deliberately conservative feature matrix with exactly 11 inspectable features: transaction amount, a log-transformed amount, origin balance relationships, and fixed transaction-type indicators. The target `isFraud`, benchmark `isFlaggedFraud`, raw identifiers, both post-transaction balances, destination balance data, and `step` are excluded from the baseline matrix. `step` is retained in raw data for future chronological splitting after its time semantics are verified.
