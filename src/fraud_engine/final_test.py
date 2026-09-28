"""Helpers for the frozen final out-of-time test evaluation policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fraud_engine.evaluation import (
    evaluate_predictions,
    evaluate_scores,
    predictions_from_threshold,
)

EXPECTED_SELECTED_POLICY = "minimum_recall_0.70"
EXPECTED_SELECTED_THRESHOLD = 0.02895689437774259
MODEL_NAME = "unweighted logistic regression"
MODEL_SELECTION_METRIC = "Average Precision"


def load_frozen_threshold_policy(path: Path) -> dict[str, Any]:
    """Load and validate the Phase 6A frozen threshold artifact."""
    artifact = json.loads(path.read_text())
    if artifact.get("final_threshold_selected") is not True:
        raise ValueError("final_threshold_selected must be true")
    if artifact.get("selected_policy") != EXPECTED_SELECTED_POLICY:
        raise ValueError("selected_policy must be minimum_recall_0.70")
    if artifact.get("selected_threshold") != EXPECTED_SELECTED_THRESHOLD:
        raise ValueError("selected_threshold does not match the frozen value")
    if artifact.get("test_evaluated") is not False:
        raise ValueError("Phase 6A artifact must have test_evaluated=false")
    if artifact.get("test_features_built") is not False:
        raise ValueError("Phase 6A artifact must have test_features_built=false")
    return artifact


def evaluate_frozen_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate continuous scores and frozen-threshold predictions."""
    ranking_metrics = evaluate_scores(y_true, scores)
    predictions = predictions_from_threshold(scores, threshold)
    threshold_metrics = evaluate_predictions(y_true, predictions)
    alerts = (
        threshold_metrics["true_positives"]
        + threshold_metrics["false_positives"]
    )
    row_count = len(np.asarray(y_true))
    threshold_metrics["alerts"] = int(alerts)
    threshold_metrics["alert_rate"] = alerts / row_count if row_count else 0.0
    return {
        "ranking_metrics": ranking_metrics,
        "threshold_metrics": threshold_metrics,
    }


def evaluate_binary_rule_benchmark(
    y_true: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fixed binary rule benchmark without ranking metrics."""
    metrics = evaluate_predictions(y_true, predictions)
    metrics["alerts"] = int(
        metrics["true_positives"] + metrics["false_positives"]
    )
    row_count = len(np.asarray(y_true))
    metrics["alert_rate"] = (
        metrics["alerts"] / row_count if row_count else 0.0
    )
    return metrics


def build_final_test_result(
    *,
    feature_names: Sequence[str],
    threshold_policy: dict[str, Any],
    model_metrics: dict[str, Any],
    rule_benchmark_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the final test result schema after the approved execution."""
    return {
        "evaluation_type": "final_out_of_time_test",
        "model_name": MODEL_NAME,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "train_step_range": [1, 446],
        "test_step_range": [595, 743],
        "feature_names": list(feature_names),
        "model_fitting_policy": "TRAIN only",
        "preprocessing_fitting_policy": "TRAIN only",
        "selected_threshold_policy": threshold_policy["selected_policy"],
        "selected_threshold": threshold_policy["selected_threshold"],
        "threshold_selected_on": "VALIDATION",
        "test_ranking_metrics": model_metrics["ranking_metrics"],
        "test_threshold_metrics": model_metrics["threshold_metrics"],
        "isFlaggedFraud_test_benchmark": rule_benchmark_metrics,
        "test_evaluated": True,
        "frozen_policy_statement": (
            "Model and threshold were frozen before final out-of-time test "
            "evaluation."
        ),
        "post_test_tuning": False,
        "post_test_tuning_statement": (
            "No model, feature, preprocessing, class-weight, hyperparameter, "
            "or threshold changes are made after seeing test results."
        ),
    }
