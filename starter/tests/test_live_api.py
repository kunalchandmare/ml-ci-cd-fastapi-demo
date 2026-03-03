import requests

url = "https://ml-ci-cd-fastapi-demo.onrender.com/prediction/"


def test_train_data_live_api():

    data = {
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

    r = requests.post(url, json=data)
    assert r.status_code == 200
    pred_r = r.json()
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.json()}")
    assert pred_r["prediction"] == ">50K"
    assert pred_r["probability_over_50k"] > 0.70