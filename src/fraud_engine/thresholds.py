"""Validation-only threshold operating-point helpers."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from fraud_engine.evaluation import evaluate_predictions, predictions_from_threshold

REFERENCE_THRESHOLD_DESCRIPTION = (
    "Reference/default threshold, not an operationally selected threshold"
)
MINIMUM_RECALL_POLICY_DESCRIPTION = (
    "For each target recall, choose the highest available validation score "
    "threshold with recall greater than or equal to the target."
)


def select_threshold_for_minimum_recall(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_recall: float,
) -> float | None:
    """Return the highest score threshold that reaches ``target_recall``."""
    if target_recall < 0:
        raise ValueError("target_recall must be non-negative")
    if target_recall > 1:
        return None

    true_values = np.asarray(y_true)
    score_values = np.asarray(scores, dtype=float)
    _validate_equal_length(true_values, score_values)
    if int((true_values == 1).sum()) == 0 and target_recall > 0:
        return None

    order = np.argsort(-score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_positive = (true_values[order] == 1).astype(int)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_positives = np.cumsum(sorted_positive)[group_ends]
    recalls = cumulative_positives / int((true_values == 1).sum())
    feasible_positions = np.flatnonzero(recalls >= target_recall)
    if len(feasible_positions):
        return float(sorted_scores[group_ends[feasible_positions[0]]])
    return None


def evaluate_threshold_operating_point(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    policy: str = "",
    target_recall: float | None = None,
) -> dict[str, Any]:
    """Evaluate one threshold using validation scores only."""
    true_values = np.asarray(y_true)
    score_values = np.asarray(scores, dtype=float)
    _validate_equal_length(true_values, score_values)

    predictions = predictions_from_threshold(score_values, threshold)
    metrics = evaluate_predictions(true_values, predictions)
    alerts = metrics["true_positives"] + metrics["false_positives"]
    validation_rows = int(len(true_values))
    alert_rate = alerts / validation_rows if validation_rows else 0.0

    return {
        "policy": policy,
        "target_recall": target_recall,
        "threshold": float(threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_negatives": metrics["false_negatives"],
        "alerts": int(alerts),
        "alert_rate": float(alert_rate),
    }


def build_recall_operating_points(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    recall_targets: Iterable[float],
) -> list[dict[str, Any]]:
    """Build operating points in the supplied recall-target order."""
    operating_points: list[dict[str, Any]] = []
    for target_recall in recall_targets:
        threshold = select_threshold_for_minimum_recall(
            y_true,
            scores,
            target_recall,
        )
        policy = f"minimum_recall_{target_recall:.2f}"
        if threshold is None:
            operating_points.append(
                {
                    "policy": policy,
                    "target_recall": float(target_recall),
                    "threshold": None,
                    "feasible": False,
                    "note": "No validation score threshold achieves this recall.",
                }
            )
            continue

        point = evaluate_threshold_operating_point(
            y_true,
            scores,
            threshold,
            policy=policy,
            target_recall=float(target_recall),
        )
        point["feasible"] = True
        operating_points.append(point)
    return operating_points


def build_threshold_analysis_report(
    *,
    model_name: str,
    validation_average_precision: float,
    feature_names: Sequence[str],
    recall_targets: Sequence[float],
    operating_points: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build deterministic metadata for the Phase 6A analysis artifact."""
    return {
        "model_name": model_name,
        "model_selection_metric": "Average Precision",
        "validation_average_precision": float(validation_average_precision),
        "train_step_range": [1, 446],
        "validation_step_range": [447, 594],
        "feature_names": list(feature_names),
        "recall_targets": list(recall_targets),
        "operating_points": list(operating_points),
        "threshold_policy_description": MINIMUM_RECALL_POLICY_DESCRIPTION,
        "reference_threshold_description": REFERENCE_THRESHOLD_DESCRIPTION,
        "final_threshold_selected": False,
        "test_evaluated": False,
        "test_features_built": False,
        "final_model_policy": (
            "Retain the TRAIN-fitted preprocessing and unweighted logistic "
            "pipeline for future final test evaluation; do not refit on "
            "TRAIN + VALIDATION in Phase 6A."
        ),
    }


def _validate_equal_length(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> None:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
