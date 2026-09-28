"""Run Phase 6A validation threshold analysis for the selected model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve

from fraud_engine.data import load_raw_transactions
from fraud_engine.evaluation import evaluate_scores
from fraud_engine.features import build_feature_matrix, build_target
from fraud_engine.modeling import build_logistic_pipeline, positive_class_scores
from fraud_engine.split import chronological_split
from fraud_engine.thresholds import (
    REFERENCE_THRESHOLD_DESCRIPTION,
    build_recall_operating_points,
    build_threshold_analysis_report,
    evaluate_threshold_operating_point,
)
from fraud_engine.validation import assert_valid_raw_transactions

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECALL_TARGETS = (0.25, 0.50, 0.70, 0.80, 0.90, 0.95)
REFERENCE_THRESHOLD = 0.5
EXPECTED_VALIDATION_AVERAGE_PRECISION = 0.7290299850363265
AVERAGE_PRECISION_TOLERANCE = 1e-6
TRAIN_COLUMNS = ["type", "amount", "oldbalanceOrg", "isFraud"]
CSV_FIELDS = [
    "policy",
    "target_recall",
    "threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "alerts",
    "alert_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to the PaySim CSV file")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("reports/modeling/threshold_analysis.json"),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("reports/modeling/threshold_analysis.csv"),
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("reports/modeling/figures/validation_threshold_tradeoff.png"),
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    raw = load_raw_transactions(args.data_path)
    _print_elapsed("Loaded raw transactions", start_time)
    validation_result = assert_valid_raw_transactions(raw)
    if validation_result.warnings:
        print(f"Validation warnings: {validation_result.warnings}")
    _print_elapsed("Validated raw transactions", start_time)

    train_raw, validation_raw, test_raw = chronological_split(raw)
    del raw, test_raw
    _print_elapsed("Created chronological train/validation split", start_time)

    train_raw = train_raw.loc[:, TRAIN_COLUMNS].copy()
    validation_raw = validation_raw.loc[:, TRAIN_COLUMNS].copy()
    _print_elapsed("Narrowed train/validation columns", start_time)

    X_train = build_feature_matrix(train_raw)
    y_train = build_target(train_raw)
    X_validation = build_feature_matrix(validation_raw)
    y_validation = build_target(validation_raw)
    del train_raw, validation_raw
    _print_elapsed("Built train/validation features and targets", start_time)

    model = build_logistic_pipeline(class_weight=None)
    model.fit(X_train, y_train)
    _print_elapsed("Fitted unweighted logistic pipeline", start_time)
    validation_scores = positive_class_scores(model, X_validation).to_numpy()
    del model, X_train, y_train
    _print_elapsed("Generated validation scores", start_time)

    ranking_metrics = evaluate_scores(y_validation, validation_scores)
    validation_average_precision = ranking_metrics["average_precision"]
    if (
        abs(
            validation_average_precision
            - EXPECTED_VALIDATION_AVERAGE_PRECISION
        )
        > AVERAGE_PRECISION_TOLERANCE
    ):
        raise RuntimeError(
            "Reproduced validation Average Precision differs from Phase 5: "
            f"{validation_average_precision:.12f}"
        )

    operating_points = build_recall_operating_points(
        y_validation.to_numpy(),
        validation_scores,
        RECALL_TARGETS,
    )
    reference_point = evaluate_threshold_operating_point(
        y_validation.to_numpy(),
        validation_scores,
        REFERENCE_THRESHOLD,
        policy="reference_threshold_0.5",
    )
    reference_point["description"] = REFERENCE_THRESHOLD_DESCRIPTION
    operating_points.append(reference_point)

    report = build_threshold_analysis_report(
        model_name="unweighted logistic regression",
        validation_average_precision=validation_average_precision,
        feature_names=list(X_validation.columns),
        recall_targets=RECALL_TARGETS,
        operating_points=operating_points,
    )
    del X_validation

    _write_json(args.json_path, report)
    _write_csv(args.csv_path, operating_points)
    _save_threshold_figure(
        y_validation.to_numpy(),
        validation_scores,
        operating_points,
        args.figure_path,
    )

    print(json.dumps(report, indent=2))
    print(f"Threshold analysis JSON written to {args.json_path}")
    print(f"Threshold analysis CSV written to {args.csv_path}")
    print(f"Validation threshold figure written to {args.figure_path}")


def _print_elapsed(message: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"{message} ({elapsed:.1f}s)", flush=True)


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _write_csv(path: Path, operating_points: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(operating_points)


def _save_threshold_figure(
    y_true: np.ndarray,
    scores: np.ndarray,
    operating_points: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="Unweighted logistic")
    for point in operating_points:
        if point.get("threshold") is None:
            continue
        recall_value = cast(float, point["recall"])
        precision_value = cast(float, point["precision"])
        axis.scatter(recall_value, precision_value, s=36)
    axis.set_title("VALIDATION precision-recall threshold trade-off")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
