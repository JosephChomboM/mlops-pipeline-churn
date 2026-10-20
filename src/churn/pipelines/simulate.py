"""Simulated production on the reserved Telco pool.

Telco is a single static snapshot, so time is simulated:
  - each month a cohort is drawn from production_pool without replacement
  - drift is a population shift: sampling is biased toward month-to-month, fiber
    and electronic-check customers. Labels are always Telco's real labels
  - labels go to a label store and only become visible LABEL_MATURITY_DAYS later
  - a delivery can be corrupted (upstream schema change) to exercise validation
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from churn import config
from churn.data.ingest import load_split


def _drift_score(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["contract"] == "Month-to-month").astype(float)
        + (df["internet_service"] == "Fiber optic").astype(float)
        + 0.5 * (df["payment_method"] == "Electronic check").astype(float)
    ).to_numpy()


def _used_ids() -> set[str]:
    if not config.LABEL_STORE.exists():
        return set()
    return set(pd.read_csv(config.LABEL_STORE)[config.ID_COL])


def deliver_cohort(snapshot_date: str, shift: float = 0.0, corrupt: bool = False) -> pd.DataFrame:
    pool = load_split("production_pool")
    pool = pool[~pool[config.ID_COL].isin(_used_ids())]

    weights = np.exp(shift * _drift_score(pool))
    seed = (config.RANDOM_STATE + int(snapshot_date.replace("-", ""))) % 2**32
    cohort = pool.sample(n=config.COHORT_SIZE, weights=weights, random_state=seed)

    features = cohort.drop(columns=[config.TARGET]).copy()
    if corrupt:
        # Upstream starts sending cents and renames a category
        features["monthly_charges"] = features["monthly_charges"] * 100
        features["internet_service"] = features["internet_service"].replace("Fiber optic", "Fiber")

    config.PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.PORTFOLIO_DIR / f"{snapshot_date}.csv", index=False)

    available = date.fromisoformat(snapshot_date) + timedelta(days=config.LABEL_MATURITY_DAYS)
    labels = pd.DataFrame(
        {
            config.ID_COL: cohort[config.ID_COL],
            "snapshot_date": snapshot_date,
            config.TARGET: cohort[config.TARGET],
            "available_from": available.isoformat(),
        }
    )
    labels.to_csv(config.LABEL_STORE, mode="a", header=not config.LABEL_STORE.exists(), index=False)
    return features
