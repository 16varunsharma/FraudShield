
from fastapi.testclient import TestClient
#from fraudshield.api import app

from src.api import app
from src.data import make_synthetic

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_score_and_explain():
    row = make_synthetic(n=2000).iloc[0].drop("Class").to_dict()
    r = client.post("/score", json={"features": row})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0 and len(body["top_drivers"]) == 5
    e = client.post("/explain", json={"features": row})
    assert e.status_code == 200 and e.json()["sources"]


def test_missing_features_rejected():
    assert client.post("/score", json={"features": {"Amount": 10}}).status_code == 422


