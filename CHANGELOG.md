# Changelog

All notable changes to this project are documented at a high level here. Detailed
modelling decisions live in `docs/decisions.md`.

## [0.1.0] - 2026-07-22
### Added
- Initial project scaffold for Explainable Fraud Risk Engine.
- `pyproject.toml` with developer dependencies.
- `.gitignore`, `LICENSE`, and GitHub Actions CI.
- `data/README.md` describing dataset expectations and raw data exclusion.

## Phase 1 - Data validation
### Added
- PaySim CSV loading and raw transaction validation.
- Fatal validation errors and non-fatal data-quality warnings.

## Phase 2 - Exploratory data analysis
### Added
- Reusable EDA summaries for class balance, transaction types, amounts, balances,
  identifiers, and the `isFlaggedFraud` benchmark.
- EDA figures under `reports/eda/figures/`.

## Phase 3 - Leakage-aware feature engineering
### Added
- Conservative 11-feature model matrix.
- Feature-availability documentation excluding target, benchmark, identifiers,
  post-transaction balances, destination balance information, and `step` from `X`.

## Phase 4 - Chronological split
### Added
- Whole-step TRAIN, VALIDATION, and TEST partitions.
- Split documentation preserving forward-in-time evaluation.

## Phase 5 - Validation modelling
### Added
- Prior dummy, `isFlaggedFraud`, unweighted logistic, and balanced logistic
  validation baselines.
- Validation metrics and validation-only PR/ROC figures.

## Phase 6A - Threshold policy
### Added
- Validation-only recall operating-point analysis.
- Frozen demonstration threshold policy: `minimum_recall_0.70`.

## Phase 6B - Final out-of-time TEST evaluation
### Added
- Frozen final TEST evaluation implementation.
- Final TEST artifact under `reports/modeling/test_results.json`.
- Documentation confirming no post-TEST tuning.

## Portfolio polish
### Changed
- Restructured documentation for portfolio readability and reproducibility.
- Standardised PaySim path examples around `data/PS_20174392719_1491204439457_log.csv`.
- Removed obsolete Phase 0 scaffolding that had no current portfolio,
  packaging, CI, or reproducibility role.
