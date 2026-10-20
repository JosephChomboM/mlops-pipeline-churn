import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.data.ingest import load_clean
from churn.models.bundle import save_bundle
from churn.models.train import build_model


@pytest.fixture(scope="session")
def telco() -> pd.DataFrame:
    return load_clean()


@pytest.fixture
def sample(telco) -> pd.DataFrame:
    return telco.sample(400, random_state=0).reset_index(drop=True)


@pytest.fixture
def champion_dir(tmp_path, telco, monkeypatch):
    """A small champion bundle trained in a couple of seconds, isolated in tmp."""
    train = telco.sample(800, random_state=1)
    model = build_model(LogisticRegression(max_iter=1000)).fit(train[config.FEATURES], train[config.TARGET])
    metadata = {
        "version": 1,
        "model": "logistic_regression",
        "run_id": "test",
        "trigger": "test",
        "trained_on_cohorts": [],
    }
    path = tmp_path / "champion"
    save_bundle(path, model, {"rule": "threshold", "threshold": 0.4}, metadata, train)
    monkeypatch.setattr(config, "CHAMPION_DIR", path)
    return path
