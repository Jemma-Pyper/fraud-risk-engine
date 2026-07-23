from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_transactions(path: str | Path) -> pd.DataFrame:
    """Load raw PaySim-style transaction data from a CSV file.

    Parameters
    ----------
    path:
        Path or string location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw transaction records.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Raw transaction data file not found: {path_obj}")

    return pd.read_csv(path_obj)


def summarize_raw_transactions(
    df: pd.DataFrame,
) -> dict[str, int | float | dict[str, int]]:
    """Return a lightweight summary of raw transaction data."""
    required_columns = {"isFraud", "type", "step", "amount"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot summarize raw transactions because required columns are missing: "
            f"{missing_columns}"
        )

    total_rows = int(len(df))
    fraud_labels = (
        pd.to_numeric(df["isFraud"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    fraud_count = int(fraud_labels.sum())
    fraud_rate = float(fraud_count / total_rows) if total_rows else 0.0
    transaction_type_counts = (
        df["type"].astype(str).str.strip().value_counts(dropna=False).to_dict()
    )

    return {
        "total_rows": total_rows,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "transaction_type_counts": transaction_type_counts,
    }
