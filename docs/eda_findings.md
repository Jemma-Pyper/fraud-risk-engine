# Exploratory Data Analysis Findings

## Dataset and validation

The analysis used `data/PS_20174392719_1491204439457_log.csv` through the Phase 1 loader and validator. The raw file is ignored by Git and was not modified.

- Rows: 6,362,620
- Columns: 11
- `step` range: 1 to 743
- Exact duplicate rows: 0
- Zero-amount transactions: 16
- Fatal validation errors: none
- Non-fatal validation warnings: 16 zero-amount transactions

## Observed target facts

- Fraud transactions: 8,213
- Non-fraud transactions: 6,354,407
- Fraud rate: 0.129082%
- Non-fraud-to-fraud ratio: 773.70:1

This is a highly imbalanced classification problem. Accuracy alone would be misleading because a classifier could achieve high accuracy by mostly predicting the majority non-fraud class while missing fraud cases.

## Observed transaction-type facts

| Type | Transactions | Fraud count | Fraud rate |
|---|---:|---:|---:|
| `CASH_IN` | 1,399,284 | 0 | 0.000000% |
| `CASH_OUT` | 2,237,500 | 4,116 | 0.183955% |
| `DEBIT` | 41,432 | 0 | 0.000000% |
| `PAYMENT` | 2,151,495 | 0 | 0.000000% |
| `TRANSFER` | 532,909 | 4,097 | 0.768799% |

`CASH_OUT` contains the largest number of fraud cases because it has the highest transaction volume. `TRANSFER` has the highest conditional fraud rate. These are descriptive comparisons, not causal findings.

## Observed amount facts

| Class | Count | Mean | Median | Std. dev. | Minimum | Maximum | P01 | P25 | P75 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-fraud | 6,354,407 | 178,197.04 | 74,684.72 | 596,236.98 | 0.01 | 92,445,516.64 | 449.24 | 13,368.40 | 208,364.76 | 1,586,064.17 |
| Fraud | 8,213 | 1,467,967.30 | 441,423.44 | 2,404,252.95 | 0.00 | 10,000,000.00 | 1,842.00 | 127,091.33 | 1,517,771.48 | 10,000,000.00 |

Fraudulent transactions have higher mean, median, upper quartiles, and dispersion in this dataset. Non-fraud has a larger maximum because of extreme non-fraud observations. Amount distributions are strongly skewed, so the saved visualization also includes `log1p(amount)`; the raw `amount` column is not changed.

## Observed balance facts

| Balance column | Mean | Std. dev. | Minimum | P25 | Median | P75 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oldbalanceOrg` | 833,883.10 | 2,888,242.67 | 0.00 | 0.00 | 14,208.00 | 107,315.18 | 59,585,040.37 |
| `newbalanceOrig` | 855,113.67 | 2,924,048.50 | 0.00 | 0.00 | 0.00 | 144,258.41 | 49,585,040.37 |
| `oldbalanceDest` | 1,100,701.67 | 3,399,180.11 | 0.00 | 0.00 | 132,705.67 | 943,036.71 | 356,015,889.35 |
| `newbalanceDest` | 1,224,996.40 | 3,674,128.94 | 0.00 | 0.00 | 214,661.44 | 1,111,909.25 | 356,179,278.92 |

All four balance columns have zero minimum and first quartile values. `newbalanceOrig` has a zero median. These are descriptive properties of simulated data; balance discrepancies are not proof of fraud and no derived balance quantity is automatically promoted to a future feature.

## Existing `isFlaggedFraud` benchmark

| Measure | Value |
|---|---:|
| True positives | 16 |
| True negatives | 6,354,407 |
| False positives | 0 |
| False negatives | 8,197 |
| Precision | 1.000000 |
| Recall | 0.001948 |
| Missed fraud cases | 8,197 |
| False alerts | 0 |

The existing rule is very precise on this dataset but catches only a small fraction of fraud. It therefore illustrates the trade-off between avoiding false alerts and missing suspicious transactions. `isFlaggedFraud` remains a benchmark only and is not a future model feature.

## Identifier facts

| Identifier | Cardinality | Repeated identifiers | Maximum frequency | Top identifier share |
|---|---:|---:|---:|---:|
| `nameOrig` | 6,353,307 | 9,298 | 3 | 0.00004715% |
| `nameDest` | 2,722,362 | 459,658 | 113 | 0.00177600% |

Destination identifiers repeat much more often than origin identifiers in this dataset. Identifiers remain descriptive only; they are not one-hot encoded, hashed, frequency encoded, or otherwise used as modelling features during Phase 2.

## Interpretation and next-phase decisions

- The data is extremely imbalanced, so future evaluation must include precision, recall, average precision, threshold analysis, and business review consequences.
- Transaction volume and conditional fraud rate provide different views of transaction-type risk and should not be conflated.
- Amount and balance distributions motivate careful preprocessing and leakage review, but do not establish causality.
- The PaySim rule benchmark is useful for comparison but has poor recall in this dataset.
- No feature engineering, splitting, resampling, or modelling was performed in Phase 2.

## Reproducibility

```bash
python scripts/run_eda.py data/PS_20174392719_1491204439457_log.csv
```

All numerical summaries and benchmark metrics use the full dataset. Only the amount and balance distribution plots use a visualization sample of approximately 250,000 rows: all 8,213 fraud rows are retained, only non-fraud rows are sampled, and `random_state=42` is used for reproducibility.