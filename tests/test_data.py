from pathlib import Path

import pytest

from fraud_engine.data import load_raw_transactions, summarize_raw_transactions
from fraud_engine.validation import assert_valid_raw_transactions


def test_load_raw_transactions_pathlib() -> None:
    path = Path("tests/fixtures/sample_transactions.csv")
    df = load_raw_transactions(path)

    assert list(df.columns) == [
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
    assert len(df) == 1


def test_load_raw_transactions_missing_file() -> None:
    path = Path("tests/fixtures/missing_file.csv")

    with pytest.raises(FileNotFoundError, match="Raw transaction data file not found"):
        load_raw_transactions(path)


def test_summarize_raw_transactions() -> None:
    df = load_raw_transactions(Path("tests/fixtures/sample_transactions.csv"))
    summary = summarize_raw_transactions(df)

    assert summary["total_rows"] == 1
    assert summary["fraud_count"] == 0
    assert summary["fraud_rate"] == 0.0
    assert summary["transaction_type_counts"]["TRANSFER"] == 1


def test_assert_valid_raw_transactions_passes_for_fixture() -> None:
    df = load_raw_transactions(Path("tests/fixtures/sample_transactions.csv"))
    result = assert_valid_raw_transactions(df)

    assert result.is_valid
    assert result.warnings == []
