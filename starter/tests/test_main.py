from fastapi.testclient import TestClient

# Import our app from main.py.
from starter.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_root_main():
    r = client.get("/")
    assert r.status_code == 200
