import json

import pytest
from fastapi.testclient import TestClient

from churn import config


@pytest.fixture
def client(champion_dir, tmp_path, monkeypatch):
    from churn.serving.api import app

    log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(app, "PREDICTION_LOG", log)
    with TestClient(app.app) as c:
        c.log_path = log
        yield c


@pytest.fixture
def customer(telco) -> dict:
    return json.loads(telco.iloc[0][[config.ID_COL, *config.FEATURES]].to_json())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_exposes_version_and_policy(client):
    body = client.get("/model").json()
    assert body["version"] == 1
    assert body["decision_policy"]["threshold"] == 0.4


def test_predict_returns_a_decision_and_logs_it(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["contact"] == (body["churn_probability"] >= 0.4)
    assert "x-request-id" in response.headers

    logged = json.loads(client.log_path.read_text().splitlines()[-1])
    assert logged["customer_id"] == customer["customer_id"]
    assert logged["model_version"] == 1


def test_invalid_payload_is_rejected_before_the_model(client, customer):
    assert client.post("/predict", json={**customer, "contract": "Weekly", "tenure": -1}).status_code == 422


def test_batch(client, customer):
    response = client.post("/predict/batch", json=[customer, customer])
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert client.post("/predict/batch", json=[]).status_code == 400
