# Data guidance

This project uses the PaySim synthetic transaction dataset as a demonstration of fraud-risk modelling.

## Expected dataset location
Place the raw dataset CSV in:

```
data/raw/paysim_transactions.csv
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

## Why raw data is excluded from Git
Raw PaySim data is excluded because it may be large and is not required for repository structure, documentation, or tests. Users should obtain their own copy and place it in the expected path.

## Test fixture
A small synthetic fixture lives under `tests/fixtures/` to support automated tests without committing the full dataset.
