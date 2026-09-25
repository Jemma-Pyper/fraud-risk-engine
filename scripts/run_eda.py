"""Run Phase 2 EDA against an explicitly supplied CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_engine.data import load_raw_transactions
from fraud_engine.eda import (
    amount_summary,
    balance_summary,
    benchmark_summary,
    dataset_overview,
    fraud_by_transaction_type,
    identifier_summary,
    save_eda_figures,
    target_summary,
)
from fraud_engine.validation import assert_valid_raw_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to the PaySim CSV file")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/eda/figures"),
        help="Directory for generated figures",
    )
    args = parser.parse_args()

    transactions = load_raw_transactions(args.data_path)
    validation_result = assert_valid_raw_transactions(transactions)

    print("Dataset overview:")
    print(dataset_overview(transactions, validation_result))
    print("\nTarget summary:")
    print(target_summary(transactions))
    print("\nFraud by transaction type:")
    print(fraud_by_transaction_type(transactions).to_string())
    print("\nAmount summary:")
    print(amount_summary(transactions).to_string())
    print("\nBalance summary:")
    print(balance_summary(transactions).to_string())
    print("\nBenchmark summary:")
    print(benchmark_summary(transactions))
    print("\nIdentifier summary:")
    print(identifier_summary(transactions))
    print("\nFigures:")
    for figure_path in save_eda_figures(transactions, args.figures_dir):
        print(figure_path)


if __name__ == "__main__":
    main()
