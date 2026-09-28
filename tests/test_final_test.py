import json
from pathlib import Path

import pytest

from fraud_engine.features import FEATURE_COLUMNS
from fraud_engine.final_test import (
    EXPECTED_SELECTED_THRESHOLD,
    MODEL_NAME,
    build_final_test_result,
    evaluate_binary_rule_benchmark,
    evaluate_frozen_threshold,
    load_frozen_threshold_policy,
)
from fraud_engine.modeling import build_logistic_pipeline


def write_threshold_artifact(path: Path, **overrides: object) -> Path:
    artifact: dict[str, object] = {
        "final_threshold_selected": True,
        "selected_policy": "minimum_recall_0.70",
        "selected_threshold": EXPECTED_SELECTED_THRESHOLD,
        "test_evaluated": False,
        "test_features_built": False,
    }
    artifact.update(overrides)
    path.write_text(json.dumps(artifact))
    return path


def test_load_frozen_threshold_policy_accepts_exact_phase_6a_artifact(
    tmp_path: Path,
) -> None:
    path = write_threshold_artifact(tmp_path / "threshold.json")

    artifact = load_frozen_threshold_policy(path)

    assert artifact["selected_policy"] == "minimum_recall_0.70"
    assert artifact["selected_threshold"] == EXPECTED_SELECTED_THRESHOLD


def test_load_frozen_threshold_policy_rejects_changed_threshold(
    tmp_path: Path,
) -> None:
    path = write_threshold_artifact(
        tmp_path / "threshold.json",
        selected_threshold=0.028957,
    )

    with pytest.raises(ValueError, match="selected_threshold"):
        load_frozen_threshold_policy(path)


def test_load_frozen_threshold_policy_requires_selected_policy(
    tmp_path: Path,
) -> None:
    path = write_threshold_artifact(
        tmp_path / "threshold.json",
        selected_policy="minimum_recall_0.80",
    )

    with pytest.raises(ValueError, match="selected_policy"):
        load_frozen_threshold_policy(path)


def test_load_frozen_threshold_policy_requires_final_selection(
    tmp_path: Path,
) -> None:
    path = write_threshold_artifact(
        tmp_path / "threshold.json",
        final_threshold_selected=False,
    )

    with pytest.raises(ValueError, match="final_threshold_selected"):
        load_frozen_threshold_policy(path)


def test_predictions_use_scores_greater_than_or_equal_to_frozen_threshold() -> None:
    metrics = evaluate_frozen_threshold(
        [1, 0, 1],
        [
            EXPECTED_SELECTED_THRESHOLD,
            EXPECTED_SELECTED_THRESHOLD,
            EXPECTED_SELECTED_THRESHOLD - 0.001,
        ],
        EXPECTED_SELECTED_THRESHOLD,
    )["threshold_metrics"]

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_final_test_metrics_keep_ranking_and_threshold_metrics_separate() -> None:
    result = evaluate_frozen_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        0.5,
    )

    assert result["ranking_metrics"]["average_precision"] == 1.0
    assert result["ranking_metrics"]["roc_auc"] == 1.0
    assert result["threshold_metrics"]["precision"] == 1.0
    assert result["threshold_metrics"]["recall"] == 1.0


def test_alert_count_and_rate_are_calculated_for_frozen_threshold() -> None:
    metrics = evaluate_frozen_threshold(
        [0, 0, 1, 1],
        [0.7, 0.1, 0.8, 0.2],
        0.5,
    )["threshold_metrics"]

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["alerts"] == 2
    assert metrics["alert_rate"] == pytest.approx(0.5)


def test_isflaggedfraud_rule_benchmark_is_binary_only() -> None:
    metrics = evaluate_binary_rule_benchmark([0, 1, 1], [0, 1, 0])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["alerts"] == 1
    assert metrics["alert_rate"] == pytest.approx(1 / 3)
    assert "average_precision" not in metrics
    assert "roc_auc" not in metrics


def test_exact_feature_contract_is_preserved() -> None:
    assert list(FEATURE_COLUMNS) == [
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


def test_final_model_contract_is_unweighted_logistic_regression() -> None:
    model = build_logistic_pipeline(class_weight=None)

    assert MODEL_NAME == "unweighted logistic regression"
    assert model.named_steps["model"].class_weight is None
    assert model.named_steps["model"].solver == "lbfgs"
    assert model.named_steps["model"].max_iter == 1000


def test_final_result_schema_records_validation_threshold_and_no_post_test_tuning(
    tmp_path: Path,
) -> None:
    threshold_policy = load_frozen_threshold_policy(
        write_threshold_artifact(tmp_path / "threshold.json")
    )
    model_metrics = evaluate_frozen_threshold([0, 1], [0.1, 0.9], 0.5)
    rule_metrics = evaluate_binary_rule_benchmark([0, 1], [0, 1])

    result = build_final_test_result(
        feature_names=FEATURE_COLUMNS,
        threshold_policy=threshold_policy,
        model_metrics=model_metrics,
        rule_benchmark_metrics=rule_metrics,
    )

    assert result["evaluation_type"] == "final_out_of_time_test"
    assert result["threshold_selected_on"] == "VALIDATION"
    assert result["post_test_tuning"] is False
    assert result["model_fitting_policy"] == "TRAIN only"
    assert result["preprocessing_fitting_policy"] == "TRAIN only"
    assert result["selected_threshold"] == EXPECTED_SELECTED_THRESHOLD


def test_phase_6b_code_contains_no_selection_or_search_logic() -> None:
    script = Path("scripts/run_final_test_evaluation.py").read_text()
    module = Path("src/fraud_engine/final_test.py").read_text()
    combined = script + module

    forbidden_terms = [
        "select_threshold",
        "build_recall_operating_points",
        "GridSearchCV",
        "RandomizedSearchCV",
        'class_weight="balanced"',
        "class_weight='balanced'",
    ]
    assert all(term not in combined for term in forbidden_terms)


def test_no_train_validation_combined_fit_path_exists() -> None:
    script = Path("scripts/run_final_test_evaluation.py").read_text()

    assert "validation_raw" in script
    assert "model.fit(train_features, train_target)" in script
    assert "concat" not in script
    assert "train_validation" not in script
