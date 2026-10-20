import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn import config
from churn.monitoring.drift import feature_drift, psi_categorical, psi_numeric
from churn.monitoring.performance import bootstrap_auc_ci


def test_psi_is_zero_for_identical_distributions():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi_numeric(s, s) < 1e-6


def test_psi_detects_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < config.PSI_WARN
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 5000))) > config.PSI_RETRAIN


def test_psi_categorical_handles_unseen_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 60)
    assert psi_categorical(ref, cur) > config.PSI_RETRAIN


def test_feature_drift_flags_population_shift(telco):
    reference = telco.sample(2000, random_state=0)
    month_to_month = telco[telco["contract"] == "Month-to-month"].sample(500, random_state=0)
    report = {r["feature"]: r for r in feature_drift(reference, month_to_month)}
    assert report["contract"]["status"] == "drift"
    assert report["gender"]["status"] == "ok"


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = np.clip(0.4 * y + rng.uniform(0, 0.6, 300), 0, 1)
    low, high = bootstrap_auc_ci(y, proba)
    assert low < roc_auc_score(y, proba) < high
