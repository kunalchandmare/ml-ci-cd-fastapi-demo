# Import Union since our Item object will have tags that can be strings or a list.
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

from starter.ml.model import load_model,inference
from starter.ml.data import process_data

# Declare the data object with its components and their type.
class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

app = FastAPI()

# Global variables (loaded once at startup)
MODEL = None
ENCODER = None
LB = None

def get_model_components() -> Tuple:
    """Dependency to get loaded model artifacts (lazy load)"""
    global MODEL, ENCODER, LB
    if MODEL is None:
        try:
            cwd = Path.cwd()
            model_dir = Path(cwd) / "model"

            saved_file = model_dir / "model.joblib"
            if not saved_file.exists():
                raise FileNotFoundError(f"No model found at {saved_file}")
            else:
                print(f"Found model at {saved_file}")
            MODEL, ENCODER, LB = load_model(saved_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return MODEL, ENCODER, LB


@app.get("/")
async def root():
    return {"message": "Hello! This is API for to connect our Model which is "
                       " a supervised binary classification model to predict whether an adult earns more than $50,000"
                       "per year based on demographic and employment features from the 1994 US Census data."}

@app.post("/prediction/")
async def predict(data: PredictionRequest,
                  model_components: Tuple = Depends(get_model_components)):

    model, encoder, lb = model_components

    # Convert request to DataFrame (single row)
    input_data = pd.DataFrame([data.model_dump(by_alias=True)])

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

    try:
        # Process input (transform categoricals, same as training)
        X_processed, _, _, _ = process_data(
            input_data,
            categorical_features=cat_features,
            label=None,  # no label in inference
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Run inference
        predictions = inference(model, X_processed)  # single value

        # Convert back to original label
        prediction_labels = lb.inverse_transform(predictions.reshape(-1, 1))
        # For single sample:
        prediction_label = prediction_labels[0]

        # Optional: get probability
        prob = model.predict_proba(X_processed)[0][1]  # prob of >50K
        return {
            "prediction": prediction_label,
            "probability_over_50k": round(float(prob), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

