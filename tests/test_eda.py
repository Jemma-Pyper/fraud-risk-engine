from pathlib import Path

import pandas as pd

from fraud_engine.data import load_raw_transactions
from fraud_engine.eda import (
    amount_summary,
    benchmark_summary,
    build_visualization_sample,
    dataset_overview,
    fraud_by_transaction_type,
    identifier_summary,
    save_eda_figures,
    target_summary,
)
from fraud_engine.validation import assert_valid_raw_transactions

FIXTURE_PATH = Path("tests/fixtures/sample_transactions.csv")


def make_eda_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0,
                "isFraud": 1,
                "isFlaggedFraud": 1,
            },
            {
                "step": 2,
                "type": "PAYMENT",
                "amount": 10.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 90.0,
                "nameDest": "C3",
                "oldbalanceDest": 20.0,
                "newbalanceDest": 30.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 3,
                "type": "PAYMENT",
                "amount": 0.0,
                "nameOrig": "C4",
                "oldbalanceOrg": 50.0,
                "newbalanceOrig": 50.0,
                "nameDest": "C3",
                "oldbalanceDest": 30.0,
                "newbalanceDest": 30.0,
                "isFraud": 0,
                "isFlaggedFraud": 1,
            },
        ]
    )


def test_target_summary_calculates_prevalence_and_imbalance() -> None:
    summary = target_summary(make_eda_frame())

    assert summary["fraud_count"] == 1
    assert summary["non_fraud_count"] == 2
    assert summary["fraud_rate"] == 1 / 3
    assert summary["imbalance_ratio_non_fraud_to_fraud"] == 2.0


def test_fraud_by_type_reports_volume_and_conditional_rate() -> None:
    summary = fraud_by_transaction_type(make_eda_frame())

    assert summary.loc["TRANSFER", "transaction_count"] == 1
    assert summary.loc["TRANSFER", "fraud_rate"] == 1.0
    assert summary.loc["PAYMENT", "transaction_count"] == 2
    assert summary.loc["PAYMENT", "fraud_count"] == 0


def test_benchmark_summary_calculates_confusion_metrics() -> None:
    summary = benchmark_summary(make_eda_frame())

    assert summary["true_positives"] == 1
    assert summary["true_negatives"] == 1
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 1.0
    assert summary["true_fraud_cases_missed"] == 0
    assert summary["false_alerts"] == 1


def test_benchmark_summary_handles_zero_denominators() -> None:
    df = make_eda_frame()
    df["isFlaggedFraud"] = 0
    df["isFraud"] = 0

    summary = benchmark_summary(df)

    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0


def test_eda_calculations_do_not_modify_input() -> None:
    df = make_eda_frame()
    original = df.copy(deep=True)

    dataset_overview(df)
    target_summary(df)
    fraud_by_transaction_type(df)
    amount_summary(df)
    identifier_summary(df)
    benchmark_summary(df)

    pd.testing.assert_frame_equal(df, original)


def test_fixture_runs_through_phase_1_and_eda() -> None:
    df = load_raw_transactions(FIXTURE_PATH)
    validation_result = assert_valid_raw_transactions(df)
    overview = dataset_overview(df, validation_result)

    assert overview["row_count"] == 1
    assert overview["validation_warnings"] == []


def test_save_eda_figures_creates_expected_files(tmp_path: Path) -> None:
    files = save_eda_figures(make_eda_frame(), tmp_path)

    assert len(files) == 6
    assert all(path.exists() for path in files)


def test_visualization_sample_retains_all_fraud_and_requested_size() -> None:
    df = make_eda_frame()

    sample = build_visualization_sample(df, target_size=2)

    assert len(sample) == 2
    assert int(sample["isFraud"].sum()) == int(df["isFraud"].sum())
    assert int((sample["isFraud"] == 0).sum()) == 1


def test_visualization_sample_is_deterministic_for_non_fraud_rows() -> None:
    df = make_eda_frame()

    first = build_visualization_sample(df, target_size=2, random_state=42)
    second = build_visualization_sample(df, target_size=2, random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_visualization_sample_handles_small_datasets_without_mutation() -> None:
    df = make_eda_frame()
    original = df.copy(deep=True)

    sample = build_visualization_sample(df, target_size=250_000)

    assert len(sample) == len(df)
    pd.testing.assert_frame_equal(df, original)
