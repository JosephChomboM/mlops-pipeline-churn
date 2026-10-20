"""Deployment bundle: model + decision policy + metadata + reference data.

Kept free of training dependencies so the serving image only needs sklearn.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn import config


def load_bundle(path: Path) -> dict:
    return {
        "model": joblib.load(path / "model.joblib"),
        "policy": json.loads((path / "policy.json").read_text()),
        "metadata": json.loads((path / "metadata.json").read_text()),
    }


def champion_exists() -> bool:
    return (config.CHAMPION_DIR / "model.joblib").exists()


def parity_check(path: Path, model, sample: pd.DataFrame) -> dict:
    """The serialized artifact must score exactly like the in-memory model."""
    X = sample[config.FEATURES]
    trained = model.predict_proba(X)[:, 1]
    deployed = joblib.load(path / "model.joblib").predict_proba(X)[:, 1]
    diff = float(np.max(np.abs(trained - deployed)))
    if diff != 0.0:
        raise RuntimeError(f"Parity check failed: serialized model differs by {diff}")
    return {"rows": len(X), "max_abs_diff": diff, "passed": True}


def save_bundle(path: Path, model, policy: dict, metadata: dict, train_df: pd.DataFrame, parity_sample=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    if parity_sample is not None:
        metadata = metadata | {"parity_check": parity_check(path, model, parity_sample)}
    (path / "policy.json").write_text(json.dumps(policy, indent=2))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Reference distribution for drift monitoring
    train_df[config.FEATURES].to_csv(path / "reference.csv", index=False)
