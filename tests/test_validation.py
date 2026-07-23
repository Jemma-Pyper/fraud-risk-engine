import pandas as pd
import pytest

from fraud_engine.validation import (
    DataValidationError,
    assert_valid_raw_transactions,
    validate_raw_transactions,
)


def make_valid_transaction() -> dict[str, object]:
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C54321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def test_validate_raw_transactions_valid_data() -> None:
    df = pd.DataFrame([make_valid_transaction()])
    result = validate_raw_transactions(df)

    assert result.is_valid
    assert result.warnings == []


def test_validate_raw_transactions_empty_data() -> None:
    df = pd.DataFrame(columns=[
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
    ])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "Dataset is empty" in result.errors


def test_validate_raw_transactions_missing_columns() -> None:
    df = pd.DataFrame([make_valid_transaction()])
    df = df.drop(columns=["amount"])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "Missing required columns" in result.errors[0]


def test_validate_raw_transactions_invalid_labels() -> None:
    transaction = make_valid_transaction()
    transaction["isFraud"] = 2
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "isFraud must contain only 0 or 1 values" in result.errors[0]


def test_validate_raw_transactions_negative_amount() -> None:
    transaction = make_valid_transaction()
    transaction["amount"] = -1.0
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "amount must be non-negative" in result.errors


def test_validate_raw_transactions_zero_amount_warning() -> None:
    transaction = make_valid_transaction()
    transaction["amount"] = 0.0
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert result.is_valid
    assert "zero-amount" in result.warnings[0]


def test_validate_raw_transactions_negative_balance() -> None:
    transaction = make_valid_transaction()
    transaction["oldbalanceOrg"] = -10.0
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "oldbalanceOrg must be non-negative" in result.errors


def test_validate_raw_transactions_invalid_transaction_type() -> None:
    transaction = make_valid_transaction()
    transaction["type"] = "UNKNOWN"
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "Unrecognised transaction types" in result.errors[0]


def test_validate_raw_transactions_missing_values() -> None:
    transaction = make_valid_transaction()
    transaction["nameDest"] = ""
    df = pd.DataFrame([transaction])
    result = validate_raw_transactions(df)

    assert not result.is_valid
    assert "Identifier column 'nameDest' contains empty values" in result.errors[0]


def test_validate_raw_transactions_duplicate_rows() -> None:
    row = make_valid_transaction()
    df = pd.DataFrame([row, row])
    result = validate_raw_transactions(df)

    assert result.is_valid
    assert result.duplicate_rows == 2
    assert "duplicate row" in result.warnings[0]


def test_validate_raw_transactions_non_destructive() -> None:
    df = pd.DataFrame([make_valid_transaction()])
    original = df.copy(deep=True)
    validate_raw_transactions(df)

    pd.testing.assert_frame_equal(df, original)


def test_assert_valid_raw_transactions_raises_for_invalid_data() -> None:
    transaction = make_valid_transaction()
    transaction["amount"] = -1.0
    df = pd.DataFrame([transaction])

    with pytest.raises(DataValidationError) as exc_info:
        assert_valid_raw_transactions(df)

    assert "amount must be non-negative" in str(exc_info.value)
