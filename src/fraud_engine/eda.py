"""Reusable exploratory analysis calculations for PaySim transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .validation import KNOWN_TRANSACTION_TYPES, ValidationResult

TARGET_COLUMN = "isFraud"
BENCHMARK_COLUMN = "isFlaggedFraud"
BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
AMOUNT_QUANTILES = np.array([0.01, 0.25, 0.5, 0.75, 0.99], dtype=np.float64)
PLOT_SAMPLE_SIZE = 250_000


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="raise")


def dataset_overview(
    df: pd.DataFrame,
    validation_result: ValidationResult | None = None,
) -> dict[str, Any]:
    """Return structural counts and Phase 1 quality findings."""
    step_values = _numeric_series(df, "step")
    amount_values = _numeric_series(df, "amount")
    type_counts = df["type"].value_counts().sort_index().to_dict()
    warnings = list(validation_result.warnings) if validation_result else []

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "step_min": int(step_values.min()),
        "step_max": int(step_values.max()),
        "transaction_type_counts": type_counts,
        "unique_nameOrig": int(df["nameOrig"].nunique()),
        "unique_nameDest": int(df["nameDest"].nunique()),
        "exact_duplicate_row_count": int(df.duplicated(keep=False).sum()),
        "zero_amount_count": int((amount_values == 0).sum()),
        "validation_warnings": warnings,
    }


def target_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize fraud prevalence and the class imbalance ratio."""
    labels = _numeric_series(df, TARGET_COLUMN).astype(int)
    fraud_count = int((labels == 1).sum())
    non_fraud_count = int((labels == 0).sum())
    total_count = fraud_count + non_fraud_count
    fraud_rate = fraud_count / total_count if total_count else 0.0
    imbalance_ratio = (
        non_fraud_count / fraud_count if fraud_count else float("inf")
    )

    return {
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_rate": fraud_rate,
        "imbalance_ratio_non_fraud_to_fraud": imbalance_ratio,
    }


def fraud_by_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Return volume, fraud count, and conditional fraud rate by type."""
    working = pd.DataFrame(
        {
            "type": df["type"].astype(str).str.strip(),
            TARGET_COLUMN: _numeric_series(df, TARGET_COLUMN).astype(int),
        }
    )
    grouped = (
        working.groupby("type", sort=True)[TARGET_COLUMN]
        .agg(
            transaction_count="size",
            fraud_count="sum",
        )
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    return (
        grouped.set_index("type")
        .reindex(sorted(KNOWN_TRANSACTION_TYPES), fill_value=0)
        .fillna(0)
    )


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Describe raw transaction amounts separately for each target class."""
    working = pd.DataFrame(
        {
            TARGET_COLUMN: _numeric_series(df, TARGET_COLUMN).astype(int),
            "amount": _numeric_series(df, "amount"),
        }
    )
    summary = working.groupby(TARGET_COLUMN)["amount"].agg(
        count="count",
        mean="mean",
        median="median",
        standard_deviation="std",
        minimum="min",
        maximum="max",
    )
    quantiles = working.groupby(TARGET_COLUMN)["amount"].quantile(AMOUNT_QUANTILES)
    quantiles = quantiles.unstack().rename(
        columns={
            0.01: "quantile_01",
            0.25: "quantile_25",
            0.5: "quantile_50",
            0.75: "quantile_75",
            0.99: "quantile_99",
        }
    )
    return summary.join(quantiles)


def balance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the four PaySim balance columns."""
    return df[BALANCE_COLUMNS].apply(pd.to_numeric, errors="raise").describe().T


def identifier_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize identifier cardinality and transaction-frequency concentration."""
    summaries: dict[str, dict[str, Any]] = {}
    for column in IDENTIFIER_COLUMNS:
        frequencies = df[column].value_counts()
        repeated = frequencies[frequencies > 1]
        summaries[column] = {
            "cardinality": int(frequencies.size),
            "repeated_identifier_count": int(repeated.size),
            "frequency_distribution": frequencies.value_counts().sort_index().to_dict(),
            "maximum_transaction_frequency": int(frequencies.max()),
            "top_identifier_share": float(frequencies.iloc[0] / len(df)),
        }
    return summaries


def benchmark_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Compare the existing PaySim flag with the fraud target."""
    actual = _numeric_series(df, TARGET_COLUMN).astype(int)
    predicted = _numeric_series(df, BENCHMARK_COLUMN).astype(int)
    true_positive = int(((predicted == 1) & (actual == 1)).sum())
    true_negative = int(((predicted == 0) & (actual == 0)).sum())
    false_positive = int(((predicted == 1) & (actual == 0)).sum())
    false_negative = int(((predicted == 0) & (actual == 1)).sum())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)

    return {
        "true_positives": true_positive,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "true_fraud_cases_missed": false_negative,
        "false_alerts": false_positive,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def save_eda_figures(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create reproducible EDA figures without changing the input dataframe.

    Count and benchmark plots use all observations. Distribution plots use a
    deterministic sample for large datasets to keep plotting memory bounded.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    distribution_df = build_visualization_sample(df)
    figures = [
        _plot_class_distribution(df, output_path),
        _plot_transaction_type_distribution(df, output_path),
        _plot_fraud_rate_by_type(df, output_path),
        _plot_amount_distributions(distribution_df, output_path),
        _plot_balance_distributions(distribution_df, output_path),
        _plot_benchmark_confusion_matrix(df, output_path),
    ]
    return figures


def build_visualization_sample(
    df: pd.DataFrame,
    target_size: int = PLOT_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Retain all fraud rows and sample only non-fraud rows for plots."""
    if target_size < 1:
        raise ValueError("target_size must be at least 1")

    labels = _numeric_series(df, TARGET_COLUMN)
    fraud_rows = df.loc[labels == 1]
    non_fraud_rows = df.loc[labels == 0]
    non_fraud_target = max(target_size - len(fraud_rows), 0)
    non_fraud_sample_size = min(non_fraud_target, len(non_fraud_rows))
    sampled_non_fraud = non_fraud_rows.sample(
        n=non_fraud_sample_size,
        random_state=random_state,
    )
    return pd.concat([fraud_rows, sampled_non_fraud], axis=0).copy()


def _plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        _numeric_series(df, TARGET_COLUMN)
        .astype(int)
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        ["Non-fraud", "Fraud"],
        np.asarray(counts.to_numpy(dtype=np.int64)),
        color=["#4472c4", "#c00000"],
    )
    axis.set_title("Transaction class distribution")
    axis.set_ylabel("Transactions")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "class_distribution.png")


def _plot_transaction_type_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["type"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(
        counts.index.astype(str).tolist(),
        np.asarray(counts.to_numpy(dtype=np.int64)),
        color="#4472c4",
    )
    axis.set_title("Transaction volume by type")
    axis.set_xlabel("Transactions")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "transaction_type_distribution.png")


def _plot_fraud_rate_by_type(df: pd.DataFrame, output_dir: Path) -> Path:
    rates = fraud_by_transaction_type(df)["fraud_rate"].sort_values()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(
        rates.index.astype(str).tolist(),
        np.asarray(rates.to_numpy(dtype=np.float64)),
        color="#c00000",
    )
    axis.set_title("Conditional fraud rate by transaction type")
    axis.set_xlabel("Fraud rate")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "fraud_rate_by_type.png")


def _plot_amount_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    amount = _numeric_series(df, "amount")
    labels = _numeric_series(df, TARGET_COLUMN).astype(int)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, name, color in [(0, "Non-fraud", "#4472c4"), (1, "Fraud", "#c00000")]:
        values = amount[labels == label]
        axes[0].hist(values, bins=30, alpha=0.65, label=name, color=color)
        axes[1].hist(
            np.log1p(values),
            bins=30,
            alpha=0.65,
            label=name,
            color=color,
        )
    axes[0].set_title("Raw transaction amounts")
    axes[1].set_title("log1p transaction amounts")
    for axis in axes:
        axis.set_xlabel("Amount")
        axis.set_ylabel("Transactions")
        axis.legend()
    figure.tight_layout()
    return _save_figure(figure, output_dir / "amount_distributions.png")


def _plot_balance_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    labels = _numeric_series(df, TARGET_COLUMN).astype(int)
    for axis, column in zip(axes.flat, BALANCE_COLUMNS, strict=True):
        for label, name, color in [
            (0, "Non-fraud", "#4472c4"),
            (1, "Fraud", "#c00000"),
        ]:
            axis.hist(
                _numeric_series(df.loc[labels == label], column),
                bins=30,
                alpha=0.65,
                label=name,
                color=color,
            )
        axis.set_title(column)
        axis.set_xlabel("Balance")
        axis.set_ylabel("Transactions")
    axes[0, 0].legend()
    figure.suptitle("Balance distributions by target class")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "balance_distributions.png")


def _plot_benchmark_confusion_matrix(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = benchmark_summary(df)
    matrix = [
        [summary["true_negatives"], summary["false_positives"]],
        [summary["false_negatives"], summary["true_positives"]],
    ]
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], labels=["Predicted non-fraud", "Predicted fraud"])
    axis.set_yticks([0, 1], labels=["Actual non-fraud", "Actual fraud"])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
            )
    axis.set_title("isFlaggedFraud benchmark")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "benchmark_confusion_matrix.png")


def _save_figure(figure: Figure, path: Path) -> Path:
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
