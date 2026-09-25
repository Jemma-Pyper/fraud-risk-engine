"""Chronological whole-step splitting for PaySim transactions."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

TRAIN_END_STEP: Final = 446
VALIDATION_END_STEP: Final = 594
STEP_COLUMN: Final = "step"
TARGET_COLUMN: Final = "isFraud"


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw transactions into fixed whole-step train/validation/test periods.

    The function does not enforce a particular minimum or maximum step. Values
    below the validation boundary enter training; values at or above the test
    boundary enter testing. This avoids silently discarding valid input rows.
    """
    steps = _validated_step_column(df)
    train_mask = steps <= TRAIN_END_STEP
    validation_mask = (steps > TRAIN_END_STEP) & (
        steps <= VALIDATION_END_STEP
    )
    test_mask = steps > VALIDATION_END_STEP

    train = df.loc[train_mask].copy(deep=True)
    validation = df.loc[validation_mask].copy(deep=True)
    test = df.loc[test_mask].copy(deep=True)
    return train, validation, test


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize rows, chronology, fraud count, and class prevalence."""
    if df.empty:
        return {
            "rows": 0,
            "fraud": 0,
            "non_fraud": 0,
            "fraud_rate": 0.0,
            "imbalance_ratio": float("inf"),
            "min_step": None,
            "max_step": None,
        }

    steps = _validated_step_column(df)
    fraud = _validated_target_column(df)
    row_count = int(len(df))
    fraud_count = int(fraud.sum())
    non_fraud_count = row_count - fraud_count
    return {
        "rows": row_count,
        "fraud": fraud_count,
        "non_fraud": non_fraud_count,
        "fraud_rate": fraud_count / row_count,
        "imbalance_ratio": (
            non_fraud_count / fraud_count
            if fraud_count
            else float("inf")
        ),
        "min_step": int(steps.min()),
        "max_step": int(steps.max()),
    }


def _validated_step_column(df: pd.DataFrame) -> pd.Series:
    if STEP_COLUMN not in df.columns:
        raise ValueError("Input dataframe must contain 'step'")

    steps = pd.to_numeric(df[STEP_COLUMN], errors="raise")
    if steps.isna().any():
        raise ValueError("'step' must not contain missing values")
    numeric_steps = steps.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_steps).all() or (steps % 1 != 0).any():
        raise ValueError("'step' must contain finite integer values")
    return steps


def _validated_target_column(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Input dataframe must contain 'isFraud'")

    target = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("'isFraud' must contain only 0 or 1 values")
    return target.astype(int)
