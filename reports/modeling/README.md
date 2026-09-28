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

The threshold table reports precision, recall, F1, confusion counts, alert counts, and alert rates. These quantities show the operating trade-off without inventing fraud-review capacity, false-positive costs, false-negative costs, or an automatic max-F1 decision rule. The demonstration threshold is selected only after owner review of the Phase 6A evidence, and the test partition remains sealed for the later one-time final evaluation.

The selected demonstration policy is `minimum_recall_0.70`, with exact stored threshold `0.02895689437774259`. This threshold is a project/portfolio policy choice chosen after reviewing validation operating points; it is not optimal, profit-maximising, production-ready, or economically optimal for a real bank.

At the selected point, validation precision was `0.594923`, recall was `0.700000`, F1 was `0.643198`, and the model produced 1,812 alerts, or about `0.7944%` of validation transactions. The marginal trade-off supported this choice: moving from 50% to 70% recall added 308 fraud detections and 668 false positives, while 70% to 80% added 154 fraud detections and 1,558 false positives; later recall increases raised the false-positive burden more sharply.

The selected threshold will be transferred unchanged to the future TEST evaluation. The model remains unweighted logistic regression, preprocessing remains fitted on TRAIN only, the model remains fitted on TRAIN only, and no train+validation refit will occur before Phase 6B. TEST remains sealed.

## Phase 6B final test preparation

Phase 6B prepares, but does not yet execute, the one-time final out-of-time TEST evaluation. The prepared runner verifies the frozen Phase 6A threshold artifact, fits the approved unweighted logistic pipeline on TRAIN only, and will apply the validation-selected threshold unchanged when TEST is explicitly opened.

The eventual `test_results.json` artifact will record TEST Average Precision, ROC-AUC, frozen-threshold metrics, alert count, alert rate, and the `isFlaggedFraud` binary benchmark. TEST results are final evaluation evidence only: they must not be used to change the model, threshold, features, preprocessing, class weights, solver, or hyperparameters.

The project distinguishes documented split-level TEST information from model evaluation. TEST was not used for preprocessing fitting, model fitting, model selection, threshold selection, or model-performance evaluation before Phase 6B.
