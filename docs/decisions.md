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

## 2026-09-28: Phase 6A validation threshold analysis
- Advanced the unweighted logistic-regression pipeline because Average Precision was the pre-defined primary model-selection metric and it outperformed the balanced model on validation AP.
- Kept the balanced model as useful contrast only; its recall `1.000000` at threshold `0.5` reflects one threshold's classification behaviour, not superior AP-primary ranking performance.
- Retained the TRAIN-fitted preprocessing and unweighted logistic model for the future final test evaluation instead of refitting on train plus validation, so the validation-selected threshold remains tied to the same score distribution.
- Examined validation-only recall-target operating points using the most conservative threshold that still reaches each requested recall level.
- Reported alert counts and alert rates so threshold trade-offs are visible without inventing business costs, review capacity, or false-positive/false-negative cost assumptions.
- Kept F1 descriptive only because maximizing F1 would assume precision and recall should be weighted equally.
- The initial threshold-analysis run did not automatically select a threshold, evaluate test metrics, build test features, resample data, or begin Phase 6B.
- Documented that temporal prevalence shift may affect threshold transfer; this will be measured once on the sealed test period later, not tuned on test.
- After reviewing the validation operating points, selected `minimum_recall_0.70` as the frozen demonstration threshold for the future out-of-time test evaluation.
- Stored the exact selected threshold as `0.02895689437774259`; rounded displays are documentation-only.
- Treated the threshold as a project/portfolio policy choice, not an optimal, profit-maximising, production-ready, or economically optimal bank threshold.
- The selected point captured validation recall `0.700000` with precision `0.594923`, 1,078 true positives, 734 false positives, 462 false negatives, 1,812 alerts, and alert rate about `0.7944%`.
- Chose this point because marginal alert burden increased beyond 70% recall: 70% to 80% added 154 fraud detections and 1,558 false positives; 80% to 90% added 154 fraud detections and 4,527 false positives; 90% to 95% added 77 fraud detections and 5,497 false positives.
- The model and preprocessing remain fitted on TRAIN only; no train+validation refit will occur before the one-time TEST evaluation.
- TEST remains sealed, and the selected threshold will be transferred unchanged to TEST in Phase 6B.

## 2026-09-28: Phase 6B final test preparation
- Prepared the final out-of-time TEST evaluation code without running it on the real TEST period.
- The final evaluation policy remains frozen: unweighted logistic regression, preprocessing fitted on TRAIN only, model fitted on TRAIN only, no train+validation refit, and selected threshold `0.02895689437774259`.
- The final-test runner reads and verifies the Phase 6A threshold artifact rather than recalculating or reselecting a threshold.
- The eventual model evaluation will report TEST Average Precision, ROC-AUC, frozen-threshold classification metrics, alert count, and alert rate.
- The `isFlaggedFraud` benchmark will be evaluated on TEST as binary predictions only, not as continuous model scores.
- TEST results will be used as final out-of-time evaluation only; they must not drive model, threshold, feature, preprocessing, solver, class-weight, resampling, or hyperparameter changes.
- The correct test-seal wording is that TEST was not used for model fitting, preprocessing fitting, model selection, threshold selection, or model-performance evaluation before Phase 6B.
