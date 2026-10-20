"""Turns a churn probability into a contact decision.

For a customer who gets the retention offer:
    actual churner      -> stays with prob. OFFER_SUCCESS_RATE: success_rate * value - offer_cost
    would have stayed   -> the offer is wasted:                  -offer_cost
Not contacting is worth 0.

Two rules compete on the test set and the one that earns more ships with the model:
    expected_value  contact if p * success_rate * value > offer_cost. Optimal per customer,
                    but only if p is calibrated.
    threshold       one global cut-off that maximizes net value. Robust to miscalibration.
"""

import numpy as np
import pandas as pd

from churn import config


def customer_value(X: pd.DataFrame) -> np.ndarray:
    """Revenue saved if the customer stays: monthly charge x retained months."""
    return X["monthly_charges"].to_numpy() * config.RETAINED_MONTHS


def expected_value(proba, value) -> np.ndarray:
    return np.asarray(proba) * config.OFFER_SUCCESS_RATE * np.asarray(value) - config.RETENTION_OFFER_COST


def decide(proba, value, policy: dict) -> np.ndarray:
    if policy["rule"] == "expected_value":
        return expected_value(proba, value) > 0
    return np.asarray(proba) >= policy["threshold"]


def net_value(y_true, contact, value) -> float:
    y_true, contact, value = np.asarray(y_true), np.asarray(contact, dtype=bool), np.asarray(value)
    gain = np.where(
        y_true == 1,
        config.OFFER_SUCCESS_RATE * value - config.RETENTION_OFFER_COST,
        -config.RETENTION_OFFER_COST,
    )
    return float(gain[contact].sum())


def choose_policy(y_true, proba, value) -> dict:
    proba = np.asarray(proba)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    by_threshold = [net_value(y_true, proba >= t, value) for t in grid]
    best_threshold = float(grid[int(np.argmax(by_threshold))])
    threshold_value = max(by_threshold)
    ev_value = net_value(y_true, expected_value(proba, value) > 0, value)

    if ev_value >= threshold_value:
        policy = {"rule": "expected_value", "threshold": None}
    else:
        policy = {"rule": "threshold", "threshold": best_threshold}
    contact = decide(proba, value, policy)
    return policy | {
        "net_value": round(max(ev_value, threshold_value), 2),
        "net_value_expected_value_rule": round(ev_value, 2),
        "net_value_best_threshold": round(threshold_value, 2),
        "best_threshold": best_threshold,
        "net_value_at_0.5": round(net_value(y_true, proba >= 0.5, value), 2),
        "contact_rate": round(float(contact.mean()), 4),
        "n_customers": int(len(contact)),
        "costs": {
            "retention_offer_cost": config.RETENTION_OFFER_COST,
            "offer_success_rate": config.OFFER_SUCCESS_RATE,
            "retained_months": config.RETAINED_MONTHS,
        },
    }


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "medium"
    return "low"
