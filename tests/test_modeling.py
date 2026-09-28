import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from fraud_engine.features import FEATURE_COLUMNS, build_feature_matrix, build_target
from fraud_engine.modeling import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    MODEL_FEATURES,
    build_dummy_prior,
    build_logistic_pipeline,
    build_preprocessor,
    positive_class_scores,
)


def make_model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [10.0, 20.0, 100.0, 200.0],
            "oldbalanceOrg": [100.0, 80.0, 100.0, 0.0],
            "type": ["PAYMENT", "PAYMENT", "TRANSFER", "CASH_OUT"],
            "isFraud": [0, 0, 1, 1],
            "isFlaggedFraud": [0, 0, 0, 1],
            "step": [1, 2, 3, 4],
            "nameOrig": ["O1", "O2", "O3", "O4"],
            "nameDest": ["D1", "D2", "D3", "D4"],
            "oldbalanceDest": [0.0, 0.0, 0.0, 0.0],
            "newbalanceOrig": [90.0, 60.0, 0.0, 0.0],
            "newbalanceDest": [10.0, 20.0, 100.0, 200.0],
        }
    )


def test_preprocessor_has_explicit_feature_groups() -> None:
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]

    assert names == ["continuous", "binary"]
    assert list(CONTINUOUS_FEATURES) == [
        "amount",
        "log1p_amount",
        "oldbalanceOrg",
        "amount_to_origin_balance",
    ]
    assert len(BINARY_FEATURES) == 7
    assert len(MODEL_FEATURES) == 11


def test_preprocessor_scales_continuous_and_passes_binary() -> None:
    df = make_model_frame()
    features = build_feature_matrix(df)
    train_features = features.iloc[:3]
    validation_features = features.iloc[3:]
    preprocessor = build_preprocessor()

    transformed_train = preprocessor.fit_transform(train_features)
    transformed_validation = preprocessor.transform(validation_features)

    assert isinstance(transformed_train, pd.DataFrame)
    assert isinstance(transformed_validation, pd.DataFrame)
    assert transformed_train.shape == (3, 11)
    assert transformed_validation.shape == (1, 11)
    assert list(transformed_train.columns) == list(
        CONTINUOUS_FEATURES + BINARY_FEATURES
    )
    assert transformed_validation.iloc[0]["type_CASH_OUT"] == 1


def test_logistic_pipeline_contracts_are_explicit() -> None:
    unweighted = build_logistic_pipeline()
    balanced = build_logistic_pipeline(class_weight="balanced")

    assert isinstance(unweighted.named_steps["model"], LogisticRegression)
    assert isinstance(balanced.named_steps["model"], LogisticRegression)
    assert unweighted.named_steps["model"].solver == "lbfgs"
    assert unweighted.named_steps["model"].max_iter == 1000
    assert unweighted.named_steps["model"].class_weight is None
    assert balanced.named_steps["model"].class_weight == "balanced"


def test_pipeline_fits_without_target_or_forbidden_fields() -> None:
    df = make_model_frame()
    features = build_feature_matrix(df)
    target = build_target(df)
    pipeline = build_logistic_pipeline()

    pipeline.fit(features, target)
    scores = positive_class_scores(pipeline, features)

    assert len(scores) == len(df)
    assert np.isfinite(scores.to_numpy()).all()
    forbidden_fields = {
        "step",
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceOrig",
        "newbalanceDest",
    }
    assert set(features.columns) == set(FEATURE_COLUMNS)
    assert forbidden_fields.isdisjoint(features.columns)


def test_dummy_prior_is_explicit_and_returns_probabilities() -> None:
    model = build_dummy_prior()
    assert isinstance(model, DummyClassifier)
    model.fit(np.zeros((4, 1)), [0, 0, 1, 0])

    probabilities = np.asarray(model.predict_proba(np.zeros((2, 1))))

    assert probabilities.shape == (2, 2)
    assert probabilities[:, 1].tolist() == [0.25, 0.25]


def test_positive_class_scores_requires_probability_model() -> None:
    try:
        positive_class_scores(object(), pd.DataFrame({"amount": [1.0]}))
    except TypeError as exc:
        assert "predict_proba" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
