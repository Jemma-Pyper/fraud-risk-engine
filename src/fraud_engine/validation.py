from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
KNOWN_TRANSACTION_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}
NUMERIC_NON_NEGATIVE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
LABEL_COLUMNS = ["isFraud", "isFlaggedFraud"]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    zero_amount_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DataValidationError(ValueError):
    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        message = self._format_message(result)
        super().__init__(message)

    @staticmethod
    def _format_message(result: ValidationResult) -> str:
        text = ["Data validation failed:"]
        text.extend(f"- {error}" for error in result.errors)
        if result.warnings:
            text.append("Warnings:")
            text.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(text)


def assert_valid_raw_transactions(df: DataFrame) -> ValidationResult:
    result = validate_raw_transactions(df)
    if result.errors:
        raise DataValidationError(result)
    return result


def _check_presence_and_nonempty(df: DataFrame, result: ValidationResult) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        result.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return

    if df.empty:
        result.errors.append("Dataset is empty")
        return

    missing_value_columns = [col for col in REQUIRED_COLUMNS if df[col].isna().any()]
    if missing_value_columns:
        result.errors.append(
            "Missing values found in required columns: "
            f"{', '.join(missing_value_columns)}"
        )

    for col in IDENTIFIER_COLUMNS:
        invalid_values = df[col].astype(str).str.strip() == ""
        if invalid_values.any():
            result.errors.append(
                f"Identifier column '{col}' contains empty values in "
                f"{int(invalid_values.sum())} row(s)"
            )


def _check_step(df: DataFrame, result: ValidationResult) -> None:
    step_values = pd.to_numeric(df["step"], errors="coerce")
    invalid_step = step_values.isna() | (step_values < 0) | (step_values % 1 != 0)
    if invalid_step.any():
        result.errors.append("step must be non-negative integers; invalid values found")


def _check_transaction_types(df: DataFrame, result: ValidationResult) -> None:
    transaction_types = df["type"].astype(str).str.strip()
    invalid_types = sorted(
        set(transaction_types[~transaction_types.isin(KNOWN_TRANSACTION_TYPES)])
    )
    if invalid_types:
        result.errors.append(f"Unrecognised transaction types: {invalid_types}")


def _check_amounts(df: DataFrame, result: ValidationResult) -> None:
    amount_values = pd.to_numeric(df["amount"], errors="coerce")
    if amount_values.isna().any():
        result.errors.append("amount must be numeric")
        return

    if (amount_values < 0).any():
        result.errors.append("amount must be non-negative")

    zero_amounts = int((amount_values == 0).sum())
    if zero_amounts:
        result.warnings.append(
            f"Found {zero_amounts} zero-amount transaction(s)"
        )
        result.zero_amount_count = zero_amounts


def _check_balances(df: DataFrame, result: ValidationResult) -> None:
    for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            result.errors.append(f"{col} must be numeric")
        elif (values < 0).any():
            result.errors.append(f"{col} must be non-negative")


def _check_labels(df: DataFrame, result: ValidationResult) -> None:
    for col in LABEL_COLUMNS:
        label_values = pd.to_numeric(df[col], errors="coerce")
        invalid_labels = label_values.isna() | ~label_values.isin([0, 1])
        if invalid_labels.any():
            result.errors.append(f"{col} must contain only 0 or 1 values")


def _check_duplicates(df: DataFrame, result: ValidationResult) -> None:
    duplicate_rows = int(df.duplicated(keep=False).sum())
    if duplicate_rows:
        result.warnings.append(f"Found {duplicate_rows} exact duplicate row(s)")
        result.duplicate_rows = duplicate_rows


def validate_raw_transactions(df: DataFrame) -> ValidationResult:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    result = ValidationResult()
    _check_presence_and_nonempty(df, result)
    if result.errors:
        return result

    _check_step(df, result)
    _check_transaction_types(df, result)
    _check_amounts(df, result)
    _check_balances(df, result)
    _check_labels(df, result)
    _check_duplicates(df, result)

    return result
