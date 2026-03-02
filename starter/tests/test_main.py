from fastapi.testclient import TestClient

# Import our app from main.py.
from starter.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_root_main():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello! This is API for to connect our Model which is "
                       " a supervised binary classification model to predict whether an adult earns more than $50,000"
                       "per year based on demographic and employment features from the 1994 US Census data."}

def test_train_data_less_50k_prediction():
    pred_request = {
      "age": 39,
      "workclass": "State-gov",
      "fnlgt": 77516,
      "education": "Bachelors",
      "education-num": 13,
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital-gain": 2174,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "United-States"
    }
    r = client.post(
        "/prediction",json=pred_request)
    assert r.status_code == 200
    pred_r = r.json()
    assert pred_r["prediction"] == "<=50K"
    assert pred_r["probability_over_50k"] < 0.1

def test_train_data_more_50k_prediction():
    pred_request = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    r = client.post(
        "/prediction", json=pred_request)
    assert r.status_code == 200
    pred_r = r.json()
    assert pred_r["prediction"] == ">50K"
    assert pred_r["probability_over_50k"] > 0.70