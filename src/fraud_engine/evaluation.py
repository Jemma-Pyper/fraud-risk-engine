"""Validation-only ranking and threshold metric helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_scores(y_true: Any, scores: Any) -> dict[str, float]:
    """Evaluate continuous positive-class scores for ranking performance."""
    true_values = np.asarray(y_true)
    score_values = np.asarray(scores)
    positive_count = int((true_values == 1).sum())
    if positive_count == 0:
        average_precision = 0.0
    else:
        average_precision = float(
            average_precision_score(true_values, score_values)
        )

    if np.unique(true_values).size < 2:
        roc_auc = 0.0
    else:
        roc_auc = float(roc_auc_score(true_values, score_values))

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def predictions_from_threshold(
    scores: Any,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert scores to binary predictions using ``scores >= threshold``."""
    return (np.asarray(scores) >= threshold).astype(int)


def evaluate_predictions(y_true: Any, predictions: Any) -> dict[str, Any]:
    """Evaluate binary predictions at one explicit threshold."""
    true_values = np.asarray(y_true)
    prediction_values = np.asarray(predictions)
    matrix = confusion_matrix(true_values, prediction_values, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "precision": float(
            precision_score(
                true_values,
                prediction_values,
                zero_division=cast(Any, 0),
            )
        ),
        "recall": float(
            recall_score(
                true_values,
                prediction_values,
                zero_division=cast(Any, 0),
            )
        ),
        "f1": float(
            f1_score(
                true_values,
                prediction_values,
                zero_division=cast(Any, 0),
            )
        ),
        "accuracy": float(accuracy_score(true_values, prediction_values)),
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "true_negatives": int(true_negative),
        "false_negatives": int(false_negative),
        "confusion_matrix": matrix.tolist(),
    }
