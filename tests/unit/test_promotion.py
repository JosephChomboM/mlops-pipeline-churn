import numpy as np

from churn import config
from churn.models.promotion import paired_bootstrap


def test_identical_models_never_win():
    gains = np.random.default_rng(0).normal(size=1000)
    result = paired_bootstrap(gains, gains)
    assert result["observed_diff"] == 0
    assert result["win_probability"] == 0


def test_small_noisy_edge_is_not_enough():
    rng = np.random.default_rng(0)
    champion = rng.normal(40, 200, 1400)
    challenger = champion + rng.normal(0.3, 60, 1400)  # +$420 total, lots of per-customer noise
    result = paired_bootstrap(challenger, champion)
    assert result["ci95"][0] < 0 < result["ci95"][1]
    assert result["win_probability"] < config.PROMOTION_MIN_WIN_PROBABILITY


def test_clear_improvement_wins():
    rng = np.random.default_rng(0)
    champion = rng.normal(40, 200, 1400)
    challenger = champion + rng.normal(10, 20, 1400)
    result = paired_bootstrap(challenger, champion)
    assert result["ci95"][0] > 0
    assert result["win_probability"] >= config.PROMOTION_MIN_WIN_PROBABILITY
