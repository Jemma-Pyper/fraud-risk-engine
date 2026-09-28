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

## 2026-09-25
- Made the Phase 3 baseline deliberately conservative: it excludes `step`, destination balances, raw identifiers, target/benchmark columns, and all post-transaction balances.
- Fixed the baseline feature order to amount, `log1p_amount`, origin balance features, and five transaction-type indicators.
- Treat `amount / oldbalanceOrg` as `0.0` when the origin balance is zero, with a separate `origin_zero_balance` indicator preserving that state.
- Deferred historical identifier aggregates until a temporally safe pipeline exists that uses only transactions strictly before the scored transaction.

## 2026-09-25: Phase 4 chronological split
- Use whole-step partitions: train `1-446`, validation `447-594`, and test `595-743`.
- Reject random splitting because the model is intended to simulate scoring future transactions and the observed fraud prevalence changes over time.
- Prefer chronological step proportions over row-balanced 70/15/15 because transaction volume varies strongly by step; the selected split is approximately 60/20/20 by chronological step sequence.
- Keep every shared-step group together because the schema does not provide reliable ordering within a step.
- Keep `step` out of the predictor matrix; use it only for chronological partitioning until its hourly meaning is verified.
- Keep the test partition untouched during Phase 5 model selection.
- Permit future historical features to use only transactions from strictly earlier steps, never other transactions from the same step.

## 2026-09-26: Phase 5 validation baselines
- Added scikit-learn for a prior dummy classifier and explicit logistic-regression pipelines.
- Fit preprocessing and models on train only; used validation only for comparison and reference-threshold metrics; did not build or evaluate test features.
- Chose Average Precision as the primary ranking metric because fraud is rare; ROC-AUC remains secondary and accuracy is contextual only.
- Kept `isFlaggedFraud` separate as a binary validation benchmark rather than passing it to continuous-score metrics.
- Validation Average Precision was `0.729030` for unweighted logistic regression and `0.692946` for balanced logistic regression; the unweighted model is the stronger provisional ranking baseline under the primary metric.
- Balanced logistic regression achieved recall `1.000000` at threshold `0.5` but precision `0.133333` and 10,010 false positives, demonstrating the operational trade-off from class weighting.
- Threshold `0.5` remains a reference/default threshold, not an operational decision; calibration and final test evaluation are deferred.
