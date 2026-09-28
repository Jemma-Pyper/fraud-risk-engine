"""Train-only preprocessing and baseline model construction."""

from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONTINUOUS_FEATURES: Final = (
    "amount",
    "log1p_amount",
    "oldbalanceOrg",
    "amount_to_origin_balance",
)
BINARY_FEATURES: Final = (
    "origin_zero_balance",
    "amount_exceeds_origin_balance",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
)
MODEL_FEATURES: Final = CONTINUOUS_FEATURES + BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the explicit scaler/passthrough preprocessing definition."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), list(CONTINUOUS_FEATURES)),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def build_logistic_pipeline(
    class_weight: str | None = None,
) -> Pipeline:
    """Build an unweighted or balanced logistic-regression pipeline."""
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    tol=1e-4,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy_prior() -> DummyClassifier:
    """Build the no-skill prior-probability classifier."""
    return DummyClassifier(strategy="prior", random_state=42)


def positive_class_scores(model: object, features: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities as an indexed Series."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must provide predict_proba")

    probabilities = model.predict_proba(features)  # type: ignore[attr-defined]
    return pd.Series(probabilities[:, 1], index=features.index, name="score")
