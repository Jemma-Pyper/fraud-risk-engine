# Decisions

## 2026-07-22
- Chose a `src/` package layout with `pyproject.toml` and setuptools.
- Kept dependencies minimal for Phase 0: only `pytest` and `ruff`.
- Added data guidance to `data/README.md` and excluded raw data from Git.
- Created `AGENTS.md` to guide future agents on project purpose and workflow.

## 2026-07-23
- Added PaySim ingestion and validation modules.
- Decided to treat `isFraud` as the target and `isFlaggedFraud` as a benchmark only.
- Decided to reject negative amounts and balances, while warning on zero amounts.
- Decided to detect duplicate rows without dropping or rejecting them in Phase 1.

## 2026-09-25
- Kept reusable EDA calculations in `src/fraud_engine/eda.py` and kept the runner dependent on an explicit input path.
- Will not use accuracy as the sole future metric because fraud is an imbalanced classification problem; precision, recall, and review consequences are more informative.
- Kept `isFlaggedFraud` as a rule-based benchmark and excluded it, along with `isFraud`, from any future model feature matrix.
- Kept raw transaction identifiers descriptive only because identifier frequency can encode identity or dataset-specific history rather than generalisable transaction behaviour.
- Kept exploratory balance quantities descriptive only; balance discrepancies in simulated data are not proof of fraud and are not automatically promoted to features.
- Did not perform modelling, feature engineering, resampling, or splitting during EDA.
- For continuous-distribution plots, retain all fraud rows and deterministically sample only non-fraud rows with `random_state=42`; keep all summaries and benchmark metrics on the full dataset.
