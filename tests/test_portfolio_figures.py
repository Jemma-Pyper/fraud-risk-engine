from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "create_portfolio_figures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_portfolio_figures",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load create_portfolio_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_performance_comparison_uses_frozen_artifact_fields() -> None:
    module = load_script_module()
    validation_results = {
        "baselines": {
            "logistic_unweighted": {
                "ranking_metrics": {
                    "average_precision": 0.7,
                    "roc_auc": 0.9,
                }
            }
        }
    }
    threshold_analysis = {
        "selected_validation_metrics": {
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
        }
    }
    test_results = {
        "test_ranking_metrics": {
            "average_precision": 0.8,
            "roc_auc": 0.88,
        },
        "test_threshold_metrics": {
            "precision": 0.72,
            "recall": 0.68,
            "f1": 0.70,
        },
    }

    rows = module.build_performance_comparison(
        validation_results,
        threshold_analysis,
        test_results,
    )

    assert rows == [
        {"metric": "Average Precision", "validation": 0.7, "test": 0.8},
        {"metric": "ROC-AUC", "validation": 0.9, "test": 0.88},
        {"metric": "Precision", "validation": 0.6, "test": 0.72},
        {"metric": "Recall", "validation": 0.7, "test": 0.68},
        {"metric": "F1", "validation": 0.65, "test": 0.70},
    ]


def test_build_coefficient_record_documents_scaled_continuous_features() -> None:
    module = load_script_module()

    record = module.build_coefficient_record("amount", 1.25)

    assert record["coefficient_direction"] == "positive"
    assert record["transformed_feature_type"] == "scaled_continuous"
    assert "one-standard-deviation" in record["interpretation_note"]


def test_build_coefficient_record_documents_binary_indicators() -> None:
    module = load_script_module()

    record = module.build_coefficient_record("origin_zero_balance", -0.5)
    type_record = module.build_coefficient_record("type_TRANSFER", 0.5)

    assert record["coefficient_direction"] == "negative"
    assert record["transformed_feature_type"] == "binary_indicator"
    assert "0 to 1" in record["interpretation_note"]
    assert type_record["coefficient_direction"] == "positive"
    assert type_record["transformed_feature_type"] == "binary_indicator"
    assert "no omitted reference category" in type_record["interpretation_note"]


def test_build_coefficient_record_rejects_unexpected_feature() -> None:
    module = load_script_module()

    with pytest.raises(ValueError, match="Unexpected feature"):
        module.build_coefficient_record("not_a_feature", 0.1)
