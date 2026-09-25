"""Leakage-aware baseline feature construction for PaySim transactions."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

TARGET_COLUMN: Final = "isFraud"
TRANSACTION_TYPES: Final = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)
FEATURE_COLUMNS: Final = (
    "amount",
    "log1p_amount",
    "oldbalanceOrg",
    "origin_zero_balance",
    "amount_exceeds_origin_balance",
    "amount_to_origin_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic leakage-aware baseline feature matrix."""
    amount = _non_negative_numeric_column(df, "amount")
    origin_balance = _non_negative_numeric_column(df, "oldbalanceOrg")
    transaction_type = _transaction_type_column(df)

    features = pd.DataFrame(index=df.index)
    features["amount"] = amount
    features["log1p_amount"] = np.log1p(amount)
    features["oldbalanceOrg"] = origin_balance
    features["origin_zero_balance"] = (origin_balance == 0).astype(np.int8)
    features["amount_exceeds_origin_balance"] = (
        amount > origin_balance
    ).astype(np.int8)
    features["amount_to_origin_balance"] = _safe_ratio(amount, origin_balance)

    for category in TRANSACTION_TYPES:
        features[f"type_{category}"] = (
            transaction_type == category
        ).astype(np.int8)

    return features.loc[:, FEATURE_COLUMNS].copy()


def build_target(df: pd.DataFrame) -> pd.Series:
    """Return a copy of the ``isFraud`` target with the source index preserved."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Input dataframe must contain 'isFraud'")
    return df[TARGET_COLUMN].copy(deep=True)


def _non_negative_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Input dataframe must contain '{column}'")

    values = pd.to_numeric(df[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"'{column}' must not contain missing values")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"'{column}' must contain only finite values")
    if (values < 0).any():
        raise ValueError(f"'{column}' must be non-negative")
    return values


def _transaction_type_column(df: pd.DataFrame) -> pd.Series:
    if "type" not in df.columns:
        raise ValueError("Input dataframe must contain 'type'")

    transaction_type = df["type"].astype("string").str.strip()
    if transaction_type.isna().any():
        raise ValueError("'type' must not contain missing values")

    unsupported = sorted(set(transaction_type) - set(TRANSACTION_TYPES))
    if unsupported:
        raise ValueError(f"Unsupported transaction type(s): {unsupported}")
    return transaction_type


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = pd.Series(0.0, index=numerator.index)
    non_zero = denominator > 0
    ratio.loc[non_zero] = numerator.loc[non_zero] / denominator.loc[non_zero]
    return ratio
