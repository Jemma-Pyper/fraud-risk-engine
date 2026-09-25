import numpy as np
import pandas as pd
import pytest

from fraud_engine.features import FEATURE_COLUMNS, build_feature_matrix, build_target
from fraud_engine.split import (
    TRAIN_END_STEP,
    VALIDATION_END_STEP,
    chronological_split,
    summarize_split,
)


def make_split_frame() -> pd.DataFrame:
    rows = []
    for index, step in enumerate(
        [0, 1, 446, 447, 594, 595, 743, 744],
        start=100,
    ):
        rows.append(
            {
                "step": step,
                "type": "TRANSFER",
                "amount": float(index + 1),
                "nameOrig": f"O{index}",
                "oldbalanceOrg": float(index + 2),
                "newbalanceOrig": 0.0,
                "nameDest": f"D{index}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": float(index + 1),
                "isFraud": int(index % 2 == 0),
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(rows, index=range(100, 108))


def test_fixed_boundaries_and_outside_range_behavior() -> None:
    train, validation, test = chronological_split(make_split_frame())

    assert train["step"].tolist() == [0, 1, 446]
    assert validation["step"].tolist() == [447, 594]
    assert test["step"].tolist() == [595, 743, 744]
    assert TRAIN_END_STEP == 446
    assert VALIDATION_END_STEP == 594


def test_no_step_overlap_and_temporal_order() -> None:
    train, validation, test = chronological_split(make_split_frame())
    train_steps = set(train["step"])
    validation_steps = set(validation["step"])
    test_steps = set(test["step"])

    assert train_steps.isdisjoint(validation_steps)
    assert train_steps.isdisjoint(test_steps)
    assert validation_steps.isdisjoint(test_steps)
    assert train["step"].max() < validation["step"].min()
    assert validation["step"].max() < test["step"].min()


def test_every_row_is_preserved_exactly_once() -> None:
    df = make_split_frame()
    partitions = chronological_split(df)
    combined = pd.concat(partitions).sort_index()

    assert len(combined) == len(df)
    pd.testing.assert_frame_equal(combined, df.sort_index())
    assert not combined.index.duplicated().any()


def test_same_step_rows_stay_together() -> None:
    df = pd.concat([make_split_frame(), make_split_frame().iloc[[2]]])
    train, validation, test = chronological_split(df)

    assert (train["step"] == 446).sum() == 2
    assert not (validation["step"] == 446).any()
    assert not (test["step"] == 446).any()


def test_split_is_deterministic_and_input_is_unchanged() -> None:
    df = make_split_frame()
    original = df.copy(deep=True)

    first = chronological_split(df)
    second = chronological_split(df)

    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_part, second_part)
    pd.testing.assert_frame_equal(df, original)


def test_returned_partitions_are_independent_copies() -> None:
    df = make_split_frame()
    train, validation, test = chronological_split(df)

    train.loc[100, "amount"] = 999.0
    validation.loc[103, "amount"] = 998.0
    test.loc[105, "amount"] = 997.0

    assert df.loc[100, "amount"] != 999.0
    assert df.loc[103, "amount"] != 998.0
    assert df.loc[105, "amount"] != 997.0


def test_summary_reports_partition_statistics() -> None:
    train, _, _ = chronological_split(make_split_frame())
    summary = summarize_split(train)

    assert summary["rows"] == 3
    assert summary["fraud"] == 2
    assert summary["non_fraud"] == 1
    assert summary["min_step"] == 0
    assert summary["max_step"] == 446


def test_empty_input_returns_three_empty_partitions() -> None:
    df = make_split_frame().iloc[0:0]
    train, validation, test = chronological_split(df)

    assert train.empty and validation.empty and test.empty
    assert list(train.columns) == list(df.columns)
    assert summarize_split(train)["rows"] == 0


@pytest.mark.parametrize("steps", [[100], [500], [700]])
def test_single_period_inputs_are_not_discarded(steps: list[int]) -> None:
    df = make_split_frame().iloc[[0]].copy()
    df["step"] = steps

    partitions = chronological_split(df)

    assert sum(len(partition) for partition in partitions) == 1


def test_missing_or_invalid_step_fails_clearly() -> None:
    df = make_split_frame().drop(columns=["step"])
    with pytest.raises(ValueError, match="must contain 'step'"):
        chronological_split(df)

    invalid = make_split_frame()
    invalid["step"] = invalid["step"].astype(float)
    invalid.loc[100, "step"] = 1.5
    with pytest.raises(ValueError, match="finite integer"):
        chronological_split(invalid)


def test_each_partition_preserves_phase_3_feature_contract() -> None:
    for partition in chronological_split(make_split_frame()):
        features = build_feature_matrix(partition)
        target = build_target(partition)

        assert list(features.columns) == list(FEATURE_COLUMNS)
        assert "step" not in features.columns
        assert target.name == "isFraud"
        assert target.index.equals(features.index)
        assert np.isfinite(features.to_numpy(dtype=np.float64)).all()


def test_forbidden_fields_do_not_enter_features() -> None:
    forbidden = {
        "step",
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceOrig",
        "newbalanceDest",
    }
    for partition in chronological_split(make_split_frame()):
        features = build_feature_matrix(partition)
        assert forbidden.isdisjoint(features.columns)
