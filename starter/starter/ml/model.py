from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from pathlib import Path
import joblib
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
    model = RandomForestClassifier().fit(X_train, y_train)
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