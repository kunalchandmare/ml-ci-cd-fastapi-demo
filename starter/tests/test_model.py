import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model,compute_model_metrics,save_model,load_model,inference


@pytest.fixture
def binary_classification_data():
    """Small synthetic binary classification dataset"""
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 5.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def trained_model(binary_classification_data):
    X_train, y_train = binary_classification_data
    return train_model(X_train, y_train)


# ────────────────────────────────────────────────
# Tests

def test_train_model_returns_fitted_classifier(binary_classification_data):
    X_train, y_train = binary_classification_data

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert model.n_estimators == 100
    assert model.random_state == 42

    # smoke test: can predict on training data
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics_correct_values():
    # Perfect prediction
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

    # All wrong
    y_pred_wrong = np.array([1, 1, 1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred_wrong)

    assert precision == 0.0
    assert recall == 0.0
    assert fbeta == 0.0  # with zero_division=1 it returns 0 when no positives


def test_save_and_load_model_roundtrip(binary_classification_data):
    X_train, y_train = binary_classification_data
    model = train_model(X_train, y_train)

    # Fake encoder & binarizer (they can be None in simple cases)
    fake_encoder = None
    fake_lb = None

    artifact_dict = {
        "classifier": model,
        "encoder": fake_encoder,
        "label_binarizer": fake_lb
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        save_model(artifact_dict, model_dir, file_name="test_model.joblib")

        saved_file = model_dir / "test_model.joblib"
        assert saved_file.exists()

        loaded_model, loaded_encoder, loaded_lb = load_model(saved_file)

        assert isinstance(loaded_model, RandomForestClassifier)
        assert loaded_encoder is None
        assert loaded_lb is None

        # Check predictions are the same
        original_preds = model.predict(X_train)
        loaded_preds = loaded_model.predict(X_train)
        np.testing.assert_array_equal(original_preds, loaded_preds)


# Bonus test – inference consistency
def test_inference_gives_same_predictions_as_model_predict(trained_model, binary_classification_data):
    X, _ = binary_classification_data

    direct_preds = trained_model.predict(X)
    inference_preds = inference(trained_model, X)

    np.testing.assert_array_equal(direct_preds, inference_preds)
