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

## Phase 4: Chronological data splitting
Phase 4 prepares chronological modelling datasets using whole PaySim `step` values. The fixed partitions are train steps `1-446`, validation steps `447-594`, and test steps `595-743`. This preserves temporal ordering and avoids splitting transactions from the same step across partitions. `step` is used for chronology only and remains excluded from the Phase 3 feature matrix.

The observed fraud rate rises across later periods, so this temporal distribution shift is documented rather than hidden with a random split. The test partition must remain untouched during Phase 5 model selection. Historical account features, if added later, must use only steps strictly earlier than the transaction being scored.

## Phase 5: Baseline modelling and validation
Phase 5 evaluates a prior-probability dummy baseline, the validation-only `isFlaggedFraud` rule benchmark, and two logistic-regression pipelines using train steps `1-446` and validation steps `447-594`. The test partition is sealed: no test features, labels, predictions, or metrics are used.

Average Precision is the primary ranking metric, ROC-AUC is secondary, and threshold `0.5` is reported only as a reference/default threshold. The validation results and validation-only curves are saved under `reports/modeling/`. Balanced logistic regression improves recall at the reference threshold but creates substantially more false positives; its outputs are not assumed to be calibrated probabilities.

## Phase 6A: Validation threshold analysis
Phase 6A advances the unweighted logistic-regression pipeline because it had the strongest validation Average Precision under the pre-defined model-selection policy. The balanced model's higher ROC-AUC and perfect recall at threshold `0.5` do not override the AP-primary policy.

The phase examines validation-only recall-target operating points for the TRAIN-fitted unweighted model. It reports precision, recall, F1, confusion counts, alert counts, and alert rates without inventing business costs, review capacity, or an automatic max-F1 threshold. The threshold is selected only after owner review of the Phase 6A evidence, and the test partition remains sealed for a later one-time evaluation.

After reviewing the validation trade-off table, the project owner selected the `minimum_recall_0.70` operating point as the frozen demonstration threshold for the later out-of-time test evaluation. This is a project policy choice, not an economically optimal or production-ready bank threshold. The selected threshold preserves its full machine-readable precision in `reports/modeling/threshold_analysis.json`; documentation may show it rounded.

At validation recall `0.70`, the model captured 1,078 fraud cases with 734 false positives and an alert rate below 1%. Moving from 50% to 70% recall added 308 fraud detections and 668 false positives; later recall increases created steeper false-positive growth. The unweighted logistic model and preprocessing remain fitted on TRAIN only, no train+validation refit is planned, and the selected threshold will transfer unchanged to TEST when Phase 6B opens the sealed test period.

## Phase 6B: Final out-of-time test evaluation
Phase 6B performs the one-time final TEST evaluation on steps `595-743` using the frozen Phase 6A policy: unweighted logistic regression, preprocessing fitted on TRAIN only, model fitted on TRAIN only, no train+validation refit, and the validation-selected threshold `0.02895689437774259` transferred unchanged to TEST.

The final TEST ranking metrics were Average Precision `0.780443` and ROC-AUC `0.982586`. At the frozen threshold, the model reached precision `0.725879`, recall `0.686820`, F1 `0.705809`, with 1,136 true positives, 429 false positives, 518 false negatives, 1,565 alerts, and alert rate `1.2664%`.

Compared with validation, TEST Average Precision was higher while ROC-AUC remained similar. Average Precision is prevalence-sensitive, and the TEST period has a different fraud prevalence, so the AP increase should not be read as an intrinsic model improvement. The threshold transfer was reasonable: the validation-selected 70% recall policy produced TEST recall about `68.7%` and higher precision than validation.

The `isFlaggedFraud` TEST benchmark had precision `1.000000` but recall only `0.004837`, detecting 8 of 1,654 fraud cases. It remains a narrow binary benchmark rather than a model feature or replacement for scored review prioritisation.

The correct test-seal claim is that TEST was not used for model fitting, preprocessing fitting, model selection, threshold selection, or model-performance evaluation before Phase 6B. Split-level TEST row counts and prevalence were documented earlier, so the project does not claim test labels were literally never observed anywhere. TEST results are final out-of-time evaluation evidence only and are not used to change the model, threshold, features, preprocessing, solver, class weight, resampling, or hyperparameters.
