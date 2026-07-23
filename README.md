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
