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
