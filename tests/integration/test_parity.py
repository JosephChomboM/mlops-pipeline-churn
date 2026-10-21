"""Post-deployment checks: every scoring path returns the same numbers."""

import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.models.bundle import load_bundle, parity_check, save_bundle
from churn.models.train import fit_calibrated
from churn.serving.batch import score_portfolio


def test_serialized_artifact_scores_like_the_trained_model(tmp_path, telco):
    train = telco.sample(800, random_state=3)
    _, model = fit_calibrated(LogisticRegression(max_iter=1000), train[config.FEATURES], train[config.TARGET])
    save_bundle(tmp_path, model, {"rule": "expected_value", "threshold": None}, {"version": 1}, train)
    assert parity_check(tmp_path, model, telco.sample(200, random_state=4))["passed"]


def test_batch_and_api_return_identical_scores(champion_dir, telco, tmp_path, monkeypatch):
    from churn.serving.api import app

    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(app, "PREDICTION_LOG", tmp_path / "api.jsonl")
    customers = telco.sample(60, random_state=5).drop(columns=[config.TARGET])

    score_portfolio(customers, "2026-01-01")
    batch = pd.read_csv(tmp_path / "predictions" / "scores_2026-01-01.csv").set_index(config.ID_COL)

    payload = json.loads(customers[[config.ID_COL, *config.FEATURES]].to_json(orient="records"))
    with TestClient(app.app) as client:
        api = pd.DataFrame(client.post("/predict/batch", json=payload).json()).set_index("customer_id")

    model = load_bundle(champion_dir)["model"]
    direct = pd.Series(model.predict_proba(customers[config.FEATURES])[:, 1], index=customers[config.ID_COL])

    np.testing.assert_allclose(batch.loc[direct.index, "churn_probability"], direct, atol=1e-12)
    np.testing.assert_allclose(api.loc[direct.index, "churn_probability"], direct.round(4), atol=1e-12)
    assert (batch.loc[direct.index, "contact"] == api.loc[direct.index, "contact"]).all()
