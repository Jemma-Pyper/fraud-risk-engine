"""Create portfolio result and explainability artifacts from frozen outputs.

The performance figure reads committed JSON artifacts only. The coefficient
explainability artifact reconstructs the frozen TRAIN-fitted unweighted logistic
pipeline for descriptive interpretation; it does not build TEST features or
generate TEST scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from fraud_engine.data import load_raw_transactions
from fraud_engine.evaluation import evaluate_scores
from fraud_engine.features import (
    FEATURE_COLUMNS,
    TRANSACTION_TYPES,
    build_feature_matrix,
    build_target,
)
from fraud_engine.modeling import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    MODEL_FEATURES,
    build_logistic_pipeline,
    positive_class_scores,
)
from fraud_engine.split import TRAIN_END_STEP, VALIDATION_END_STEP
from fraud_engine.validation import assert_valid_raw_transactions

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
VALIDATION_RESULTS_PATH = Path("reports/modeling/validation_results.json")
THRESHOLD_ANALYSIS_PATH = Path("reports/modeling/threshold_analysis.json")
TEST_RESULTS_PATH = Path("reports/modeling/test_results.json")
FIGURES_DIR = Path("reports/modeling/figures")
EXPLAINABILITY_PATH = Path("reports/modeling/model_explainability.json")

TRAIN_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "isFraud"]
VALIDATION_AP_TOLERANCE = 1e-6
VALIDATION_ROC_TOLERANCE = 1e-6
TRANSACTION_TYPE_FEATURES = tuple(
    f"type_{transaction_type}" for transaction_type in TRANSACTION_TYPES
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_performance_comparison(
    validation_results: dict[str, Any],
    threshold_analysis: dict[str, Any],
    test_results: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Build validation-vs-TEST metrics from committed result artifacts."""
    validation_ranking = validation_results["baselines"]["logistic_unweighted"][
        "ranking_metrics"
    ]
    validation_threshold = threshold_analysis["selected_validation_metrics"]
    test_ranking = test_results["test_ranking_metrics"]
    test_threshold = test_results["test_threshold_metrics"]

    return [
        {
            "metric": "Average Precision",
            "validation": float(validation_ranking["average_precision"]),
            "test": float(test_ranking["average_precision"]),
        },
        {
            "metric": "ROC-AUC",
            "validation": float(validation_ranking["roc_auc"]),
            "test": float(test_ranking["roc_auc"]),
        },
        {
            "metric": "Precision",
            "validation": float(validation_threshold["precision"]),
            "test": float(test_threshold["precision"]),
        },
        {
            "metric": "Recall",
            "validation": float(validation_threshold["recall"]),
            "test": float(test_threshold["recall"]),
        },
        {
            "metric": "F1",
            "validation": float(validation_threshold["f1"]),
            "test": float(test_threshold["f1"]),
        },
    ]


def save_validation_vs_test_figure(
    comparison: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Save a compact validation-vs-TEST metric comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = [str(row["metric"]) for row in comparison]
    validation_values = [float(row["validation"]) for row in comparison]
    test_values = [float(row["test"]) for row in comparison]
    x_positions = np.arange(len(metric_labels))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(
        x_positions - bar_width / 2,
        validation_values,
        bar_width,
        label="Validation",
        color="#4C78A8",
    )
    axis.bar(
        x_positions + bar_width / 2,
        test_values,
        bar_width,
        label="TEST",
        color="#F58518",
    )
    axis.set_title("Frozen policy performance: validation vs TEST")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(metric_labels, rotation=20, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.0,
        -0.28,
        (
            "TEST Average Precision is prevalence-sensitive; higher TEST AP "
            "does not imply intrinsic model improvement."
        ),
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_coefficient_record(
    feature_name: str,
    coefficient: float,
) -> dict[str, Any]:
    """Build one descriptive coefficient record with interpretation metadata."""
    if feature_name in CONTINUOUS_FEATURES:
        transformed_feature_type = "scaled_continuous"
        interpretation_note = (
            "Coefficient corresponds to a one-standard-deviation increase "
            "after TRAIN-fitted scaling, holding other features fixed."
        )
    elif feature_name in TRANSACTION_TYPE_FEATURES:
        transformed_feature_type = "binary_indicator"
        interpretation_note = (
            "Transaction type is encoded with all five mutually exclusive "
            "indicators and no omitted reference category; interpret this "
            "coefficient through relative category contrasts, holding other "
            "features fixed."
        )
    elif feature_name in BINARY_FEATURES:
        transformed_feature_type = "binary_indicator"
        interpretation_note = (
            "Coefficient corresponds to changing the indicator from 0 to 1, "
            "holding other features fixed."
        )
    else:
        raise ValueError(f"Unexpected feature name: {feature_name}")

    if coefficient > 0:
        direction = "positive"
    elif coefficient < 0:
        direction = "negative"
    else:
        direction = "zero"

    return {
        "feature_name": feature_name,
        "coefficient": float(coefficient),
        "coefficient_direction": direction,
        "transformed_feature_type": transformed_feature_type,
        "interpretation_note": interpretation_note,
    }


def reconstruct_train_fitted_explainability(
    data_path: Path,
    validation_results: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the frozen TRAIN-fitted pipeline for coefficient reporting."""
    raw = load_raw_transactions(data_path)
    validation_result = assert_valid_raw_transactions(raw)
    if validation_result.errors:
        raise RuntimeError("Raw data failed validation")

    steps = raw["step"]
    train_raw = raw.loc[steps <= TRAIN_END_STEP, TRAIN_COLUMNS].copy()
    validation_raw = raw.loc[
        (steps > TRAIN_END_STEP) & (steps <= VALIDATION_END_STEP),
        TRAIN_COLUMNS,
    ].copy()
    del raw

    train_features = build_feature_matrix(train_raw)
    train_target = build_target(train_raw)
    validation_features = build_feature_matrix(validation_raw)
    validation_target = build_target(validation_raw)
    del train_raw, validation_raw

    model = build_logistic_pipeline(class_weight=None)
    model.fit(train_features, train_target)
    validation_scores = positive_class_scores(
        model,
        validation_features,
    ).to_numpy()
    reproduced_metrics = evaluate_scores(validation_target, validation_scores)
    expected_metrics = validation_results["baselines"]["logistic_unweighted"][
        "ranking_metrics"
    ]

    average_precision_difference = abs(
        reproduced_metrics["average_precision"]
        - expected_metrics["average_precision"]
    )
    roc_auc_difference = abs(
        reproduced_metrics["roc_auc"] - expected_metrics["roc_auc"]
    )
    equivalence_passed = (
        average_precision_difference <= VALIDATION_AP_TOLERANCE
        and roc_auc_difference <= VALIDATION_ROC_TOLERANCE
    )
    if not equivalence_passed:
        raise RuntimeError(
            "Reconstructed validation metrics do not match committed "
            "validation artifact."
        )

    logistic_model = model.named_steps["model"]
    coefficients = logistic_model.coef_.ravel()
    feature_names = list(MODEL_FEATURES)
    if len(coefficients) != len(feature_names):
        raise RuntimeError("Coefficient count does not match feature contract")

    records = [
        build_coefficient_record(feature_name, coefficient)
        for feature_name, coefficient in zip(
            feature_names,
            coefficients,
            strict=True,
        )
    ]

    return {
        "model_name": "unweighted logistic regression",
        "class_weight": None,
        "solver": logistic_model.solver,
        "max_iter": int(logistic_model.max_iter),
        "fitting_policy": "TRAIN only",
        "preprocessing_fitting_policy": "TRAIN only",
        "model_input_feature_contract": list(FEATURE_COLUMNS),
        "coefficient_feature_order": feature_names,
        "intercept": float(logistic_model.intercept_[0]),
        "coefficients": records,
        "validation_equivalence_check": {
            "metric_source": "reports/modeling/validation_results.json",
            "average_precision_expected": float(
                expected_metrics["average_precision"]
            ),
            "average_precision_reproduced": float(
                reproduced_metrics["average_precision"]
            ),
            "average_precision_tolerance": VALIDATION_AP_TOLERANCE,
            "roc_auc_expected": float(expected_metrics["roc_auc"]),
            "roc_auc_reproduced": float(reproduced_metrics["roc_auc"]),
            "roc_auc_tolerance": VALIDATION_ROC_TOLERANCE,
            "passed": True,
        },
        "coefficient_interpretation": {
            "positive": (
                "Higher fitted model log-odds of fraud, holding other "
                "features fixed."
            ),
            "negative": (
                "Lower fitted model log-odds of fraud, holding other "
                "features fixed."
            ),
            "scaled_continuous": (
                "Continuous coefficients are for one-standard-deviation "
                "changes after TRAIN-fitted scaling."
            ),
            "binary_indicator": (
                "Non-type binary coefficients are for changing the indicator "
                "from 0 to 1, holding other features fixed."
            ),
            "transaction_type_indicators": (
                "All five mutually exclusive transaction-type indicators are "
                "included, so there is no omitted reference category; "
                "interpret transaction-type coefficients through relative "
                "category contrasts."
            ),
        },
        "caveats": [
            "Coefficients describe associations within the fitted model, not "
            "causal effects.",
            "Continuous and binary coefficient magnitudes have different "
            "units and should not be compared naively.",
            "Correlated features, including amount and log1p_amount, "
            "complicate isolated interpretation.",
            "Transaction-type coefficients should be interpreted through "
            "relative category contrasts because all five type indicators are "
            "retained with the intercept; regularisation and this coding "
            "affect how fitted contribution is distributed across the "
            "intercept and type indicators.",
            "PaySim is simulated data, so explanations are portfolio evidence "
            "rather than production bank claims.",
        ],
        "descriptive_only": True,
        "causal_interpretation": False,
        "post_test_tuning": False,
        "test_predictions_generated": False,
        "test_evaluation_rerun": False,
    }


def save_coefficient_figure(
    explainability: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a horizontal coefficient plot with feature-type colouring."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = explainability["coefficients"]
    ordered = sorted(coefficients, key=lambda item: item["coefficient"])
    feature_names = [item["feature_name"] for item in ordered]
    values = [float(item["coefficient"]) for item in ordered]
    colors = [
        "#4C78A8"
        if item["transformed_feature_type"] == "scaled_continuous"
        else "#72B7B2"
        for item in ordered
    ]

    figure_height = max(5.5, 0.42 * len(feature_names))
    figure, axis = plt.subplots(figsize=(9, figure_height))
    y_positions = np.arange(len(feature_names))
    axis.barh(y_positions, values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(feature_names)
    axis.set_xlabel("Logistic-regression coefficient")
    axis.set_title("Frozen TRAIN-fitted logistic coefficients")
    axis.grid(axis="x", alpha=0.25)

    continuous_proxy = plt.Rectangle((0, 0), 1, 1, color="#4C78A8")
    binary_proxy = plt.Rectangle((0, 0), 1, 1, color="#72B7B2")
    axis.legend(
        [continuous_proxy, binary_proxy],
        ["Scaled continuous", "Binary indicator"],
        loc="lower right",
    )
    axis.text(
        0.0,
        -0.16,
        (
            "Signs indicate fitted log-odds direction, not causality. "
            "Type indicators have no omitted reference category."
        ),
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--validation-results-path",
        type=Path,
        default=VALIDATION_RESULTS_PATH,
    )
    parser.add_argument(
        "--threshold-analysis-path",
        type=Path,
        default=THRESHOLD_ANALYSIS_PATH,
    )
    parser.add_argument(
        "--test-results-path",
        type=Path,
        default=TEST_RESULTS_PATH,
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--explainability-path",
        type=Path,
        default=EXPLAINABILITY_PATH,
    )
    args = parser.parse_args()

    validation_results = read_json(args.validation_results_path)
    threshold_analysis = read_json(args.threshold_analysis_path)
    test_results = read_json(args.test_results_path)

    comparison = build_performance_comparison(
        validation_results,
        threshold_analysis,
        test_results,
    )
    validation_test_path = (
        args.figures_dir / "validation_vs_test_performance.png"
    )
    save_validation_vs_test_figure(comparison, validation_test_path)

    explainability = reconstruct_train_fitted_explainability(
        args.data_path,
        validation_results,
    )
    write_json(args.explainability_path, explainability)
    coefficient_path = args.figures_dir / "logistic_coefficients.png"
    save_coefficient_figure(explainability, coefficient_path)

    print(f"Validation-vs-TEST figure written to {validation_test_path}")
    print(f"Explainability artifact written to {args.explainability_path}")
    print(f"Coefficient figure written to {coefficient_path}")


if __name__ == "__main__":
    main()
