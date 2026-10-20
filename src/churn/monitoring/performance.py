"""Realized performance once labels mature.

Churn is only known LABEL_MATURITY_DAYS after the snapshot, so realized metrics
are computed on matured cohorts only. Everything else is reported as pending.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from churn import config
from churn.decision.policy import customer_value, net_value
from churn.models.evaluate import ks_statistic


def bootstrap_auc_ci(y, proba, n_boot: int = 300, seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """95% CI of the AUC (Gini CI is 2*CI-1). On a few hundred customers the point
    estimate moves +-0.08 Gini from sampling alone; retraining on that is chasing noise."""
    y, proba = np.asarray(y), np.asarray(proba)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], proba[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def load_matured_labels(as_of: str) -> pd.DataFrame:
    """Only labels that exist on `as_of`. Nothing from the future."""
    if not config.LABEL_STORE.exists():
        return pd.DataFrame(columns=[config.ID_COL, "snapshot_date", config.TARGET, "available_from"])
    labels = pd.read_csv(config.LABEL_STORE)
    return labels[labels["available_from"] <= as_of]


def realized_performance(as_of: str) -> list[dict]:
    labels = load_matured_labels(as_of)
    cohorts = []
    for path in sorted(config.PREDICTIONS_DIR.glob("scores_*.csv")):
        scores = pd.read_csv(path)
        date = str(scores["snapshot_date"].iloc[0])
        if date >= as_of:
            continue
        joined = scores.merge(labels[labels["snapshot_date"] == date][[config.ID_COL, config.TARGET]], on=config.ID_COL)
        if joined.empty:
            cohorts.append({"snapshot_date": date, "status": "pending_labels", "rows": len(scores)})
            continue

        y, p = joined[config.TARGET], joined["churn_probability"]
        portfolio = pd.read_csv(config.PORTFOLIO_DIR / f"{date}.csv")
        value = customer_value(joined[[config.ID_COL]].merge(portfolio, on=config.ID_COL))
        auc_low, auc_high = bootstrap_auc_ci(y, p)
        cohorts.append(
            {
                "snapshot_date": date,
                "status": "matured",
                "rows": len(joined),
                "model_version": int(joined["model_version"].iloc[0]),
                "churn_rate": round(float(y.mean()), 4),
                "predicted_churn_rate": round(float(p.mean()), 4),
                "gini": round(2 * float(roc_auc_score(y, p)) - 1, 4),
                "gini_ci95": (round(2 * auc_low - 1, 4), round(2 * auc_high - 1, 4)),
                "ks": round(ks_statistic(y, p), 4),
                "recall": round(float(recall_score(y, joined["contact"])), 4),
                "realized_net_value": round(net_value(y, joined["contact"], value), 2),
            }
        )
    return cohorts
