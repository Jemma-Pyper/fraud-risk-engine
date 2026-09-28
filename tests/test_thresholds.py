import pytest

from fraud_engine.thresholds import (
    build_recall_operating_points,
    build_threshold_analysis_report,
    evaluate_threshold_operating_point,
    select_threshold_for_minimum_recall,
)


def make_threshold_fixture() -> tuple[list[int], list[float]]:
    return (
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0.95, 0.80, 0.60, 0.40, 0.90, 0.70, 0.50, 0.10],
    )


def test_threshold_operating_point_uses_greater_than_or_equal() -> None:
    point = evaluate_threshold_operating_point(
        [1, 0, 1],
        [0.5, 0.5, 0.49],
        0.5,
    )

    assert point["true_positives"] == 1
    assert point["false_positives"] == 1
    assert point["false_negatives"] == 1


def test_minimum_recall_selector_chooses_highest_feasible_threshold() -> None:
    y_true, scores = make_threshold_fixture()

    threshold = select_threshold_for_minimum_recall(y_true, scores, 0.50)

    assert threshold == 0.80
    assert threshold is not None
    point = evaluate_threshold_operating_point(y_true, scores, threshold)
    assert point["recall"] >= 0.50


def test_duplicate_scores_are_deterministic() -> None:
    threshold = select_threshold_for_minimum_recall(
        [1, 1, 0, 0],
        [0.80, 0.80, 0.80, 0.20],
        1.0,
    )

    assert threshold == 0.80


def test_recall_target_25_percent() -> None:
    y_true, scores = make_threshold_fixture()

    threshold = select_threshold_for_minimum_recall(y_true, scores, 0.25)

    assert threshold == 0.95


def test_recall_target_50_percent() -> None:
    y_true, scores = make_threshold_fixture()

    threshold = select_threshold_for_minimum_recall(y_true, scores, 0.50)

    assert threshold == 0.80


def test_high_recall_target() -> None:
    y_true, scores = make_threshold_fixture()

    threshold = select_threshold_for_minimum_recall(y_true, scores, 0.75)

    assert threshold == 0.60


def test_unattainable_recall_target_reports_none() -> None:
    threshold = select_threshold_for_minimum_recall([0, 0], [0.9, 0.1], 0.5)

    assert threshold is None


def test_threshold_metrics_and_alerts_are_correct() -> None:
    y_true, scores = make_threshold_fixture()

    point = evaluate_threshold_operating_point(y_true, scores, 0.80)

    assert point["precision"] == pytest.approx(2 / 3)
    assert point["recall"] == pytest.approx(0.5)
    assert point["f1"] == pytest.approx(4 / 7)
    assert point["accuracy"] == pytest.approx(5 / 8)
    assert point["true_positives"] == 2
    assert point["false_positives"] == 1
    assert point["true_negatives"] == 3
    assert point["false_negatives"] == 2
    assert point["alerts"] == 3
    assert point["alert_rate"] == pytest.approx(3 / 8)


def test_recall_operating_points_include_reference_row_in_deterministic_order() -> None:
    y_true, scores = make_threshold_fixture()
    points = build_recall_operating_points(y_true, scores, [0.25, 0.50, 0.75])
    reference = evaluate_threshold_operating_point(
        y_true,
        scores,
        0.5,
        policy="reference_threshold_0.5",
    )
    points.append(reference)

    assert [point["policy"] for point in points] == [
        "minimum_recall_0.25",
        "minimum_recall_0.50",
        "minimum_recall_0.75",
        "reference_threshold_0.5",
    ]
    assert points[-1]["threshold"] == 0.5


def test_threshold_analysis_report_keeps_test_sealed_and_no_optimal_policy() -> None:
    y_true, scores = make_threshold_fixture()
    points = build_recall_operating_points(y_true, scores, [0.25, 0.50])
    report = build_threshold_analysis_report(
        model_name="unweighted logistic regression",
        validation_average_precision=0.72,
        feature_names=["amount"],
        recall_targets=[0.25, 0.50],
        operating_points=points,
    )

    assert report["final_threshold_selected"] is False
    assert report["test_evaluated"] is False
    assert report["test_features_built"] is False
    assert "test" not in build_recall_operating_points.__annotations__
    assert all("optimal" not in point["policy"] for point in points)


def test_negative_target_recall_fails_clearly() -> None:
    with pytest.raises(ValueError, match="target_recall"):
        select_threshold_for_minimum_recall([0, 1], [0.1, 0.9], -0.1)
