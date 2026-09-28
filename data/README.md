# Data guidance

This project uses the PaySim synthetic transaction dataset as a demonstration of fraud-risk modelling.

## Expected dataset location
Place the raw dataset CSV in:

```
data/PS_20174392719_1491204439457_log.csv
```

## Expected columns
The loader expects columns similar to PaySim, including:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

## Expected data types and validation rules
- `step`: non-negative integer
- `type`: recognised PaySim category such as `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, or `CASH_IN`
- `amount`: numeric and non-negative; zero amounts are reported as warnings
- `nameOrig` / `nameDest`: non-empty identifiers
- `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`: numeric and non-negative
- `isFraud`: binary label in `{0, 1}` used only as the training target
- `isFlaggedFraud`: binary benchmark label in `{0, 1}` used only for rule-based comparison

## Data-quality notifications
Phase 1 distinguishes fatal validation errors from non-fatal warnings. For example, negative amounts are invalid, while zero amounts are reported for later inspection.

## Why raw data is excluded from Git
Raw PaySim data is excluded because it may be large and is not required for repository structure, documentation, or tests. Users should obtain their own copy and place it in the expected path.

## Test fixture
A small synthetic fixture lives under `tests/fixtures/` to support automated tests without committing the full dataset.
