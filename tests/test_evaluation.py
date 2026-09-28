import numpy as np

from fraud_engine.evaluation import (
    evaluate_predictions,
    evaluate_scores,
    predictions_from_threshold,
)


def test_evaluate_scores_calculates_ranking_metrics() -> None:
    metrics = evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_predictions_from_threshold_uses_greater_than_or_equal() -> None:
    predictions = predictions_from_threshold([0.49, 0.5, 0.51], threshold=0.5)

    assert predictions.tolist() == [0, 1, 1]


def test_evaluate_predictions_returns_explicit_confusion_counts() -> None:
    metrics = evaluate_predictions([0, 0, 1, 1], [0, 1, 0, 1])

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_zero_positive_predictions_are_safe() -> None:
    metrics = evaluate_predictions([0, 1], [0, 0])

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_negatives"] == 1


def test_constant_target_scores_use_safe_roc_auc() -> None:
    metrics = evaluate_scores([0, 0], [0.1, 0.1])

    assert metrics["average_precision"] == 0.0
    assert metrics["roc_auc"] == 0.0
    assert np.isfinite(list(metrics.values())).all()
