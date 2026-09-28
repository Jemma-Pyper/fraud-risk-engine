"""Run Phase 5 baselines using train and validation data only."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve, roc_curve

from fraud_engine.data import load_raw_transactions
from fraud_engine.evaluation import (
    evaluate_predictions,
    evaluate_scores,
    predictions_from_threshold,
)
from fraud_engine.features import build_feature_matrix, build_target
from fraud_engine.modeling import (
    build_dummy_prior,
    build_logistic_pipeline,
    positive_class_scores,
)
from fraud_engine.split import chronological_split
from fraud_engine.validation import assert_valid_raw_transactions

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 0.5
TRAIN_COLUMNS = ["type", "amount", "oldbalanceOrg", "isFraud"]
VALIDATION_COLUMNS = TRAIN_COLUMNS + ["isFlaggedFraud"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to the PaySim CSV file")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("reports/modeling/validation_results.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/modeling/figures"),
    )
    args = parser.parse_args()

    raw = load_raw_transactions(args.data_path)
    validation_result = assert_valid_raw_transactions(raw)
    if validation_result.warnings:
        print(f"Validation warnings: {validation_result.warnings}")

    train_raw, validation_raw, test_raw = chronological_split(raw)
    del raw, test_raw

    train_raw = train_raw.loc[:, TRAIN_COLUMNS].copy()
    validation_benchmark = validation_raw["isFlaggedFraud"].copy()
    validation_raw = validation_raw.loc[:, VALIDATION_COLUMNS].copy()

    X_train = build_feature_matrix(train_raw)
    y_train = build_target(train_raw)
    X_validation = build_feature_matrix(validation_raw)
    y_validation = build_target(validation_raw)
    del train_raw, validation_raw

    results: dict[str, Any] = {
        "train_step_range": [1, 446],
        "validation_step_range": [447, 594],
        "feature_names": list(X_train.columns),
        "primary_metric": "Average Precision",
        "threshold": THRESHOLD,
        "threshold_label": (
            "Reference/default threshold, not an operationally selected threshold"
        ),
        "test_evaluated": False,
        "baselines": {},
    }
    score_sets: dict[str, np.ndarray] = {}

    dummy = build_dummy_prior()
    dummy.fit(X_train, y_train)
    dummy_scores = positive_class_scores(dummy, X_validation).to_numpy()
    score_sets["dummy_prior"] = dummy_scores
    results["baselines"]["dummy_prior"] = _probabilistic_result(
        "dummy_prior",
        dummy_scores,
        y_validation,
        class_weight=None,
        fit_seconds=None,
        convergence_warnings=[],
    )
    del dummy, dummy_scores

    benchmark_predictions = validation_benchmark.to_numpy(dtype=int)
    results["baselines"]["isFlaggedFraud"] = {
        "model_name": "isFlaggedFraud",
        "binary_metrics": evaluate_predictions(y_validation, benchmark_predictions),
        "note": "Fixed binary rule benchmark; no ranking metrics calculated.",
    }
    del benchmark_predictions, validation_benchmark

    for model_name, class_weight in (
        ("logistic_unweighted", None),
        ("logistic_balanced", "balanced"),
    ):
        model = build_logistic_pipeline(class_weight=class_weight)
        start_time = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start_time
        convergence_warnings = [str(item.message) for item in caught_warnings]
        scores = positive_class_scores(model, X_validation).to_numpy()
        score_sets[model_name] = scores
        results["baselines"][model_name] = _probabilistic_result(
            model_name,
            scores,
            y_validation,
            class_weight=class_weight,
            fit_seconds=fit_seconds,
            convergence_warnings=convergence_warnings,
        )
        if convergence_warnings:
            print(f"{model_name} convergence warnings: {convergence_warnings}")
        del model, scores

    _save_validation_figures(
        y_validation.to_numpy(), score_sets, args.figures_dir
    )
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Validation results written to {args.results_path}")
    print(f"Validation figures written to {args.figures_dir}")


def _probabilistic_result(
    model_name: str,
    scores: np.ndarray,
    y_true: pd.Series,
    *,
    class_weight: str | None,
    fit_seconds: float | None,
    convergence_warnings: list[str],
) -> dict[str, Any]:
    predictions = predictions_from_threshold(scores, THRESHOLD)
    return {
        "model_name": model_name,
        "class_weight": class_weight,
        "ranking_metrics": evaluate_scores(y_true, scores),
        "threshold": THRESHOLD,
        "threshold_label": (
            "Reference/default threshold, not an operationally selected threshold"
        ),
        "threshold_metrics": evaluate_predictions(y_true, predictions),
        "fit_seconds": fit_seconds,
        "convergence_warnings": convergence_warnings,
    }


def _save_validation_figures(
    y_true: np.ndarray,
    score_sets: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_recall_figure, precision_recall_axis = plt.subplots(figsize=(7, 5))
    for name, scores in score_sets.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = evaluate_scores(y_true, scores)["average_precision"]
        precision_recall_axis.plot(
            recall,
            precision,
            label=f"{name} (AP={average_precision:.4f})",
        )
    precision_recall_axis.set_title("Validation precision-recall curves")
    precision_recall_axis.set_xlabel("Recall")
    precision_recall_axis.set_ylabel("Precision")
    precision_recall_axis.legend()
    precision_recall_figure.tight_layout()
    precision_recall_figure.savefig(
        output_dir / "validation_precision_recall.png",
        dpi=150,
    )
    plt.close(precision_recall_figure)

    roc_figure, roc_axis = plt.subplots(figsize=(7, 5))
    for name, scores in score_sets.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        roc_auc = evaluate_scores(y_true, scores)["roc_auc"]
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (ROC-AUC={roc_auc:.4f})",
        )
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    roc_axis.set_title("Validation ROC curves")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.legend()
    roc_figure.tight_layout()
    roc_figure.savefig(output_dir / "validation_roc.png", dpi=150)
    plt.close(roc_figure)


if __name__ == "__main__":
    main()
