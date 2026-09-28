"""Run the one-time final out-of-time test evaluation.

This script is prepared in Phase 6B but must be run only after explicit approval
to open the TEST period for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_engine.data import load_raw_transactions
from fraud_engine.features import build_feature_matrix, build_target
from fraud_engine.final_test import (
    build_final_test_result,
    evaluate_binary_rule_benchmark,
    evaluate_frozen_threshold,
    load_frozen_threshold_policy,
)
from fraud_engine.modeling import build_logistic_pipeline, positive_class_scores
from fraud_engine.split import chronological_split
from fraud_engine.validation import assert_valid_raw_transactions

TRAIN_COLUMNS = ["type", "amount", "oldbalanceOrg", "isFraud"]
TEST_COLUMNS = TRAIN_COLUMNS + ["isFlaggedFraud"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to the PaySim CSV file")
    parser.add_argument(
        "--threshold-policy-path",
        type=Path,
        default=Path("reports/modeling/threshold_analysis.json"),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("reports/modeling/test_results.json"),
    )
    args = parser.parse_args()

    threshold_policy = load_frozen_threshold_policy(args.threshold_policy_path)

    raw = load_raw_transactions(args.data_path)
    validation_result = assert_valid_raw_transactions(raw)
    if validation_result.warnings:
        print(f"Validation warnings: {validation_result.warnings}")

    train_raw, validation_raw, test_raw = chronological_split(raw)
    del raw, validation_raw

    train_raw = train_raw.loc[:, TRAIN_COLUMNS].copy()
    test_benchmark = test_raw["isFlaggedFraud"].copy()
    test_raw = test_raw.loc[:, TEST_COLUMNS].copy()

    train_features = build_feature_matrix(train_raw)
    train_target = build_target(train_raw)
    test_features = build_feature_matrix(test_raw)
    test_target = build_target(test_raw)
    del train_raw, test_raw

    model = build_logistic_pipeline(class_weight=None)
    model.fit(train_features, train_target)
    test_scores = positive_class_scores(model, test_features).to_numpy()
    del model, train_features, train_target

    model_metrics = evaluate_frozen_threshold(
        test_target.to_numpy(),
        test_scores,
        threshold_policy["selected_threshold"],
    )
    rule_benchmark_metrics = evaluate_binary_rule_benchmark(
        test_target.to_numpy(),
        test_benchmark.to_numpy(dtype=int),
    )
    result = build_final_test_result(
        feature_names=list(test_features.columns),
        threshold_policy=threshold_policy,
        model_metrics=model_metrics,
        rule_benchmark_metrics=rule_benchmark_metrics,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Final test results written to {args.results_path}")


if __name__ == "__main__":
    main()
