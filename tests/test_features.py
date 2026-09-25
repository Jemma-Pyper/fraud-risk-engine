import numpy as np
import pandas as pd
import pytest

from fraud_engine.features import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    build_target,
)

EXPECTED_FEATURE_COLUMNS = [
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
]


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5],
            "type": ["TRANSFER", "PAYMENT", "CASH_IN", "DEBIT", "CASH_OUT"],
            "amount": [100.0, 10.0, 25.0, 50.0, 75.0],
            "nameOrig": ["O1", "O2", "O3", "O4", "O5"],
            "oldbalanceOrg": [200.0, 0.0, 25.0, 40.0, 100.0],
            "newbalanceOrig": [100.0, 0.0, 0.0, 0.0, 25.0],
            "nameDest": ["D1", "D2", "D3", "D4", "D5"],
            "oldbalanceDest": [0.0, 10.0, 0.0, 40.0, 20.0],
            "newbalanceDest": [100.0, 20.0, 25.0, 90.0, 95.0],
            "isFraud": [1, 0, 0, 0, 1],
            "isFlaggedFraud": [0, 0, 0, 0, 1],
        },
        index=[10, 20, 30, 40, 50],
    )


def test_feature_columns_are_exact_and_deterministic() -> None:
    features = build_feature_matrix(make_feature_frame())

    assert list(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert list(FEATURE_COLUMNS) == EXPECTED_FEATURE_COLUMNS


def test_excluded_columns_never_enter_features() -> None:
    features = build_feature_matrix(make_feature_frame())

    excluded = {
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
        "newbalanceOrig",
        "newbalanceDest",
        "oldbalanceDest",
        "step",
    }
    assert excluded.isdisjoint(features.columns)


def test_all_transaction_type_columns_exist_and_encode_correctly() -> None:
    features = build_feature_matrix(make_feature_frame())

    assert features.loc[10, "type_TRANSFER"] == 1
    assert features.loc[20, "type_PAYMENT"] == 1
    assert features.loc[30, "type_CASH_IN"] == 1
    assert features.loc[40, "type_DEBIT"] == 1
    assert features.loc[50, "type_CASH_OUT"] == 1
    assert features.filter(like="type_").sum(axis=1).eq(1).all()


def test_unsupported_transaction_type_fails_clearly() -> None:
    df = make_feature_frame()
    df.loc[10, "type"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unsupported transaction type"):
        build_feature_matrix(df)


def test_engineered_amount_and_balance_features() -> None:
    features = build_feature_matrix(make_feature_frame())

    np.testing.assert_allclose(
        features["log1p_amount"], np.log1p([100, 10, 25, 50, 75])
    )
    assert features["origin_zero_balance"].tolist() == [0, 1, 0, 0, 0]
    assert features["amount_exceeds_origin_balance"].tolist() == [0, 1, 0, 1, 0]
    np.testing.assert_allclose(
        features["amount_to_origin_balance"], [0.5, 0.0, 1.0, 1.25, 0.75]
    )


def test_features_contain_no_nan_or_infinite_values() -> None:
    features = build_feature_matrix(make_feature_frame())

    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy(dtype=np.float64)).all()


def test_feature_matrix_does_not_mutate_input() -> None:
    df = make_feature_frame()
    original = df.copy(deep=True)

    build_feature_matrix(df)

    pd.testing.assert_frame_equal(df, original)


def test_build_target_returns_correct_values_and_alignment() -> None:
    df = make_feature_frame()

    features = build_feature_matrix(df)
    target = build_target(df)

    assert target.tolist() == [1, 0, 0, 0, 1]
    assert target.index.equals(features.index)
    assert target.name == "isFraud"


def test_repeated_calls_return_identical_features() -> None:
    df = make_feature_frame()

    first = build_feature_matrix(df)
    second = build_feature_matrix(df)

    pd.testing.assert_frame_equal(first, second)


def test_small_numeric_strings_are_coercible() -> None:
    df = make_feature_frame()
    df["amount"] = df["amount"].astype(str)
    df["oldbalanceOrg"] = df["oldbalanceOrg"].astype(str)

    features = build_feature_matrix(df)

    assert features.loc[10, "amount"] == 100.0
