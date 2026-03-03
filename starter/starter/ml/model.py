import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from pathlib import Path
import joblib

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(artifact_dict, model_dir, file_name='model.joblib'):
    # Get current working directory as base
    model_path = Path(model_dir)

    # Create safe root folder if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact_dict, model_path / file_name)

def load_model(model_path):
    # Load model
    artifacts = joblib.load(model_path)
    rf_model = artifacts["classifier"]
    encoder = artifacts.get("encoder")
    lb = artifacts.get("label_binarizer")
    return rf_model, encoder, lb


def compute_performance_on_slices(
    X_test: pd.DataFrame,
    model,
    slice_feature: str,
    categorical_features: list,
    label:str,
    encoder=None,
    lb=None,
    output_file: str = None
):
    """
    Compute precision, recall, fbeta for each unique value of slice_feature.
    """
    results = []

    unique_values = X_test[slice_feature].unique()
    unique_values = sorted(unique_values)  # nicer output

    for value in unique_values:
        # Create mask
        mask = X_test[slice_feature] == value

        if mask.sum() == 0:
            continue  # skip empty slices

        # Slice data
        X_slice = X_test[mask].copy()

        # Process data for One hot encoding and binarization
        X_slice_processed, y_slice_processed,_,_ = process_data(X_slice,categorical_features,label=label, encoder=encoder, lb=lb, training=False)

        # Make sure y_test is 1D array of 0/1
        if y_slice_processed.ndim > 1:
            y_slice_processed = y_slice_processed.ravel()

        # Predict on this slice
        preds_slice = model.predict(X_slice_processed)

        # Compute metrics
        precision, recall, fbeta = compute_model_metrics(y_slice_processed, preds_slice)

        results.append({
            "slice_feature": slice_feature,
            "value": value,
            "count": mask.sum(),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fbeta": round(fbeta, 4)
        })

    # Convert to DataFrame for nice printing / saving
    results_df = pd.DataFrame(results)

    print(f"\nPerformance on slices of '{slice_feature}':")
    print(results_df.to_string(index=False))

    # Save to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Model Performance on Slices of Feature: {slice_feature}\n")
        f.write("=" * 60 + "\n\n")

        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total slices: {len(results_df)}\n")

    print(f"Results saved to: {output_file}")

    return results_df