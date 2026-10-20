import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn import config
from churn.models.evaluate import (
    calibration_table,
    compute_metrics,
    expected_calibration_error,
    ks_statistic,
    run_validation_gates,
    segment_report,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.35 * y + rng.uniform(0, 0.65, 2000), 0, 1)
    return y, proba


def test_gini_is_two_auc_minus_one(scores):
    y, proba = scores
    assert compute_metrics(y, proba, proba >= 0.5)["gini"] == pytest.approx(2 * roc_auc_score(y, proba) - 1)


def test_ks_bounds():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.uniform(size=5000)) < 0.05


def test_ece_detects_inflated_probabilities():
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < proba).astype(int)
    assert expected_calibration_error(y, proba) < 0.02
    assert expected_calibration_error(y, np.clip(proba + 0.2, 0, 1)) > 0.1


def test_calibration_table_has_ten_equal_deciles(scores):
    y, proba = scores
    table = calibration_table(y, proba)
    assert table["decile"].tolist() == list(range(1, 11))
    assert table["n"].sum() == len(y)


def test_segment_report_covers_every_driver(sample):
    proba = np.random.default_rng(0).uniform(size=len(sample))
    report = segment_report(sample[config.FEATURES], sample[config.TARGET], proba, proba > 0.5)
    assert set(report["segment"]) == set(config.SEGMENTS)
    assert report.groupby("segment")["n"].sum().eq(len(sample)).all()


def test_gates_reject_overfit_and_miscalibrated_models():
    good = {"train_gini": 0.75, "test_gini": 0.72, "test_recall": 0.8, "test_ece": 0.02}
    assert run_validation_gates(good)["passed"]
    for metrics, gate in (({**good, "train_gini": 0.95}, "max_gini_gap"), ({**good, "test_ece": 0.15}, "max_ece")):
        failed = [c["name"] for c in run_validation_gates(metrics)["checks"] if not c["passed"]]
        assert failed == [gate]
