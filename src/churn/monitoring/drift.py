"""Feature and prediction drift against the champion's training data.

Answers "did the inputs change?" on the day of scoring, without labels.
PSI: < 0.10 stable, 0.10-0.25 watch, >= 0.25 significant shift.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn import config
from churn.models.bundle import load_bundle

EPS = 1e-4


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = config.PSI_BINS) -> float:
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return _psi(np.histogram(ref, edges)[0] / len(ref), np.histogram(cur, edges)[0] / len(cur))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna()) | set(current.dropna()))
    ref_share = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_share = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_share, cur_share)


def _status(psi: float) -> str:
    if psi >= config.PSI_RETRAIN:
        return "drift"
    if psi >= config.PSI_WARN:
        return "warn"
    return "ok"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    rows = []
    for col in config.FEATURES:
        if col in config.NUMERIC_FEATURES:
            psi = psi_numeric(reference[col], current[col])
            ks_p = float(ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue)
        else:
            psi, ks_p = psi_categorical(reference[col], current[col]), None
        rows.append({"feature": col, "psi": round(psi, 4), "ks_pvalue": ks_p, "status": _status(psi)})
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


def drift_report(current: pd.DataFrame) -> dict:
    bundle = load_bundle(config.CHAMPION_DIR)
    reference = pd.read_csv(config.CHAMPION_DIR / "reference.csv")

    features = feature_drift(reference, current)
    ref_scores = bundle["model"].predict_proba(reference[config.FEATURES])[:, 1]
    cur_scores = bundle["model"].predict_proba(current[config.FEATURES])[:, 1]
    prediction_psi = psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores))
    drifted = [f["feature"] for f in features if f["status"] == "drift"]

    return {
        "model_version": bundle["metadata"]["version"],
        "max_feature_psi": features[0]["psi"],
        "top_feature": features[0]["feature"],
        "drifted_features": drifted,
        "prediction_psi": round(prediction_psi, 4),
        "prediction_status": _status(prediction_psi),
        "mean_score_reference": round(float(ref_scores.mean()), 4),
        "mean_score_current": round(float(cur_scores.mean()), 4),
        "features": features,
        "retrain_signal": bool(drifted) or prediction_psi >= config.PSI_RETRAIN,
    }
