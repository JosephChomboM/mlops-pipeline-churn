import numpy as np
import pytest

from churn import config
from churn.decision.policy import choose_policy, decide, expected_value, net_value, risk_band


def test_net_value_counts_only_contacted_customers():
    y = np.array([1, 0, 1, 0])
    contact = np.array([True, True, False, False])
    value = np.full(4, 1000.0)
    expected = (config.OFFER_SUCCESS_RATE * 1000 - config.RETENTION_OFFER_COST) - config.RETENTION_OFFER_COST
    assert net_value(y, contact, value) == pytest.approx(expected)


def test_nobody_contacted_is_worth_zero():
    assert net_value([1, 0], [False, False], np.array([500.0, 500.0])) == 0


def test_expected_value_rule_contacts_only_profitable_customers():
    proba = np.array([0.1, 0.5, 0.5])
    value = np.array([1000.0, 1000.0, 100.0])
    contact = decide(proba, value, {"rule": "expected_value", "threshold": None})
    assert contact.tolist() == (expected_value(proba, value) > 0).tolist() == [False, True, False]


def test_threshold_rule():
    contact = decide(np.array([0.2, 0.6]), np.array([1.0, 1.0]), {"rule": "threshold", "threshold": 0.5})
    assert contact.tolist() == [False, True]


def test_chosen_policy_is_never_worse_than_default():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.3 * y + rng.uniform(0, 0.7, 2000), 0, 1)
    value = rng.uniform(200, 1500, 2000)
    policy = choose_policy(y, proba, value)
    assert policy["rule"] in {"expected_value", "threshold"}
    assert policy["net_value"] >= policy["net_value_at_0.5"]
    assert policy["net_value"] == max(policy["net_value_expected_value_rule"], policy["net_value_best_threshold"])


@pytest.mark.parametrize(("p", "band"), [(0.9, "high"), (0.5, "medium"), (0.1, "low")])
def test_risk_band(p, band):
    assert risk_band(p) == band
