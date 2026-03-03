# Script to train machine learning model.
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from starter.ml.data import process_data
import starter.ml.model as ml_model

# Add code to load in the data.
def read_one_csv_to_df(artifact_inp_dir):
    """Reads the single CSV file in the directory — fails gracefully if not exactly one."""
    try:
        csv_files = [f for f in os.listdir(artifact_inp_dir)
                     if f.lower().endswith('.csv') and os.path.isfile(os.path.join(artifact_inp_dir, f))]

        if len(csv_files) != 1:
            msg = f"No CSV file found" if not csv_files else f"Found {len(csv_files)} CSVs — expected 1"
            print(msg + (f": {csv_files}" if csv_files else ""))
            return None

        file_path = os.path.join(artifact_inp_dir, csv_files[0])
        df = pd.read_csv(file_path)
        print(f"Loaded: {csv_files[0]} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"Error reading CSV from {artifact_inp_dir}: {e}")
        return None


data = read_one_csv_to_df("./data")
print(data.describe())
print(data.isnull().sum())
# Clean "?" values
data = data.replace("?", np.nan)
print(data.head())
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test,_,_ = process_data(
    test, categorical_features=cat_features, label="salary",encoder=encoder, training=False,lb=lb
)
# Train and save a model.
rf_model = ml_model.train_model(
    X_train,
    y_train)

# Compute r2 and MAE
pred = ml_model.inference(rf_model, X_test)

precision, recall, f_beta = ml_model.compute_model_metrics(y_test, pred)
print(f"\nPerformance without slices:")
print("Precision:", precision)
print("Recall:", recall)
print("FBeta:", f_beta)

# Save model package in the MLFlow sklearn format
artifacts = {
    "classifier": rf_model,
    "encoder": encoder,
    "label_binarizer": lb
}
model_dir = "../model"

ml_model.save_model(artifacts,model_dir)

ml_model.compute_performance_on_slices(test,rf_model,"education",
                                       categorical_features=cat_features,
                                       label="salary",
                                       encoder=encoder,
                                       lb=lb,
                                       output_file="./slice_output.txt")
