# Phase 5 validation baselines

This directory contains validation-only baseline results. The test partition is sealed during Phase 5: no test features, predictions, labels, or metrics are written here.

Run the validation workflow with an explicit PaySim CSV:

```bash
python scripts/run_validation_baselines.py data/PS_20174392719_1491204439457_log.csv
```

The workflow evaluates a prior dummy classifier, the `isFlaggedFraud` rule benchmark, unweighted logistic regression, and class-weighted logistic regression using training steps `1-446` and validation steps `447-594` only. Average Precision is the primary comparison metric; ROC-AUC is secondary. Threshold `0.5` is reported as a reference/default threshold, not an operational choice.

The real validation run produced these ranking results:

| Baseline | Average Precision | ROC-AUC |
|---|---:|---:|
| Prior dummy | 0.006752 | 0.500000 |
| Logistic, unweighted | 0.729030 | 0.984607 |
| Logistic, balanced | 0.692946 | 0.994853 |

At the reference threshold of `0.5`, unweighted logistic regression had precision `1.000000`, recall `0.203247`, and 313 true positives. Balanced logistic regression had precision `0.133333`, recall `1.000000`, 1,540 true positives, and 10,010 false positives. These are validation observations only, not a final threshold or test result.

The `isFlaggedFraud` validation benchmark had precision `1.000000`, recall `0.001299`, F1 `0.002594`, 2 true positives, and 0 false positives. It was evaluated as binary predictions only; no model-style ranking metrics were calculated for it.

The test partition was sealed and no test features, labels, predictions, or metrics were accessed. Balanced-model scores should not automatically be interpreted as calibrated probabilities because class weighting changes the effective fitting prior and validation prevalence differs from training prevalence.

## Phase 6A threshold analysis

Phase 6A advances the unweighted logistic-regression pipeline because Average Precision was the pre-defined primary model-selection metric. The balanced model's higher ROC-AUC and perfect recall at threshold `0.5` are useful context, but they do not override the AP-primary model-selection policy.

The threshold analysis keeps the TRAIN-fitted preprocessing and unweighted logistic model. It uses validation scores only to report recall-target operating points and the reference threshold `0.5`, which remains a reference/default threshold rather than an operationally selected threshold.

The threshold table reports precision, recall, F1, confusion counts, alert counts, and alert rates. These quantities show the operating trade-off without inventing fraud-review capacity, false-positive costs, false-negative costs, or an automatic max-F1 decision rule. The demonstration threshold was selected only after owner review of the Phase 6A evidence.

The selected demonstration policy is `minimum_recall_0.70`, with exact stored threshold `0.02895689437774259`. This threshold is a project/portfolio policy choice chosen after reviewing validation operating points; it is not optimal, profit-maximising, production-ready, or economically optimal for a real bank.

At the selected point, validation precision was `0.594923`, recall was `0.700000`, F1 was `0.643198`, and the model produced 1,812 alerts, or about `0.7944%` of validation transactions. The marginal trade-off supported this choice: moving from 50% to 70% recall added 308 fraud detections and 668 false positives, while 70% to 80% added 154 fraud detections and 1,558 false positives; later recall increases raised the false-positive burden more sharply.

The selected threshold was transferred unchanged to the Phase 6B TEST evaluation. The model remained unweighted logistic regression, preprocessing remained fitted on TRAIN only, the model remained fitted on TRAIN only, and no train+validation refit occurred.

## Phase 6B final out-of-time test evaluation

Phase 6B executed the one-time final out-of-time TEST evaluation on steps `595-743`. The frozen policy remained unchanged: unweighted logistic regression, preprocessing fitted on TRAIN only, model fitted on TRAIN only, no train+validation refit, and selected threshold `0.02895689437774259` from the Phase 6A validation artifact.

| TEST metric | Value |
|---|---:|
| Average Precision | 0.780443 |
| ROC-AUC | 0.982586 |
| Precision at frozen threshold | 0.725879 |
| Recall at frozen threshold | 0.686820 |
| F1 at frozen threshold | 0.705809 |
| Accuracy at frozen threshold | 0.992337 |
| True positives | 1,136 |
| False positives | 429 |
| True negatives | 121,497 |
| False negatives | 518 |
| Alerts | 1,565 |
| Alert rate | 1.2664% |

Validation-to-TEST transfer:

| Metric | Validation | TEST |
|---|---:|---:|
| Average Precision | 0.729030 | 0.780443 |
| ROC-AUC | 0.984607 | 0.982586 |
| Precision at selected threshold | 0.594923 | 0.725879 |
| Recall at selected threshold | 0.700000 | 0.686820 |
| F1 at selected threshold | 0.643198 | 0.705809 |
| Alert rate | 0.7944% | 1.2664% |

The selected validation threshold transferred reasonably: the 70% validation recall policy produced TEST recall about `68.7%` and higher TEST precision. The higher TEST Average Precision should be interpreted carefully because AP is prevalence-sensitive and the TEST period has different fraud prevalence; it should not be described as intrinsic model improvement. ROC-AUC remained similar across validation and TEST.

The portfolio comparison figure `figures/validation_vs_test_performance.png` visualizes Average Precision, ROC-AUC, precision, recall, and F1 using only the committed validation, threshold-analysis, and final TEST JSON artifacts.

`isFlaggedFraud` TEST benchmark:

| Metric | Value |
|---|---:|
| Precision | 1.000000 |
| Recall | 0.004837 |
| F1 | 0.009627 |
| Accuracy | 0.986681 |
| True positives | 8 |
| False positives | 0 |
| True negatives | 121,926 |
| False negatives | 1,646 |
| Alerts | 8 |
| Alert rate | 0.00647% |

The project distinguishes documented split-level TEST information from model evaluation. TEST was not used for preprocessing fitting, model fitting, model selection, threshold selection, or model-performance evaluation before Phase 6B. TEST results are final out-of-time evaluation evidence only and are not used to change the model, threshold, features, preprocessing, class weights, solver, resampling, or hyperparameters.

## Descriptive coefficient explainability

`model_explainability.json` records descriptive logistic-regression coefficient information for the frozen unweighted model configuration. Because no model binary is committed, the artifact is produced by reconstructing the exact TRAIN-fitted unweighted logistic pipeline and checking that validation Average Precision and ROC-AUC exactly reproduce `validation_results.json`.

This reconstruction is for explanation only. It uses TRAIN for fitting and VALIDATION for equivalence checking; it does not build TEST features, generate TEST predictions, rerun final TEST evaluation, change model settings, select features, tune thresholds, or alter the frozen result artifacts.

The coefficient figure `figures/logistic_coefficients.png` shows coefficient direction for the TRAIN-fitted logistic model. Positive coefficients indicate higher fitted model log-odds of fraud, holding other features fixed; negative coefficients indicate lower fitted log-odds. Continuous-feature coefficients correspond to one-standard-deviation changes after TRAIN-fitted scaling, while non-type binary coefficients correspond to changing an indicator from 0 to 1.

Transaction type requires a separate caveat because all five mutually exclusive type indicators are retained with the intercept. There is no omitted transaction-type reference category, so type coefficients should be interpreted through relative category contrasts rather than as standalone effects relative to a baseline category. Regularisation and the chosen coding affect how fitted contribution is distributed across the intercept and the full set of type indicators.

Coefficients are not causal effects, and correlated features such as `amount` and `log1p_amount` complicate isolated interpretation.
