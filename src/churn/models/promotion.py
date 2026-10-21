"""Champion vs challenger on the frozen holdout.

A challenger replaces the champion only if, on rows no model has trained on,
  1. it earns more net retention value in at least PROMOTION_MIN_WIN_PROBABILITY of
     paired bootstrap resamples of the holdout, and
  2. it loses no more than PROMOTION_GINI_TOLERANCE Gini.
Comparing the two totals directly isn't enough: differences of a few hundred dollars on
1,400 customers are inside sampling noise, and promoting on them is promoting on noise.

Promoting moves the "champion" alias in the registry and copies the bundle that
the batch job and the API serve. Rolling back is moving the alias back.
"""

import json
import shutil
from datetime import UTC, datetime

import mlflow
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score

from churn import config
from churn.data.ingest import X_y, dataframe_hash, load_split
from churn.decision.policy import customer_gains, customer_value, decide
from churn.models.bundle import champion_exists, load_bundle
from churn.models.evaluate import expected_calibration_error, ks_statistic
from churn.models.train import setup_mlflow


def score_holdout(bundle: dict, holdout) -> tuple[dict, np.ndarray]:
    """Holdout summary plus the realized gain of every customer, used for the paired bootstrap."""
    X, y = X_y(holdout)
    proba = bundle["model"].predict_proba(X)[:, 1]
    value = customer_value(X)
    contact = decide(proba, value, bundle["policy"])
    gains = customer_gains(y, contact, value)
    summary = {
        "version": bundle["metadata"]["version"],
        "gini": round(2 * float(roc_auc_score(y, proba)) - 1, 4),
        "ks": round(ks_statistic(y, proba), 4),
        "ece": round(expected_calibration_error(y, proba), 4),
        "recall": round(float(recall_score(y, contact)), 4),
        "contact_rate": round(float(contact.mean()), 4),
        "net_value": round(float(gains.sum()), 2),
    }
    return summary, gains


def evaluate_on_holdout(bundle: dict, holdout) -> dict:
    return score_holdout(bundle, holdout)[0]


def _money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def paired_bootstrap(challenger_gains: np.ndarray, champion_gains: np.ndarray) -> dict:
    """Resample customers (the same ones for both models) and count how often the challenger wins."""
    diff = challenger_gains - champion_gains
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.integers(0, len(diff), size=(config.PROMOTION_BOOTSTRAP_SAMPLES, len(diff)))
    totals = diff[idx].sum(axis=1)
    return {
        "observed_diff": round(float(diff.sum()), 2),
        "ci95": [round(float(np.percentile(totals, 2.5)), 2), round(float(np.percentile(totals, 97.5)), 2)],
        "win_probability": round(float((totals > 0).mean()), 4),
    }


def promote_if_better(challenger_version: int) -> dict:
    setup_mlflow()
    holdout = load_split("holdout")
    ch, ch_gains = score_holdout(load_bundle(config.MODELS_DIR / f"v{challenger_version}"), holdout)

    if not champion_exists():
        promoted, reason, cp, bootstrap = True, "no champion yet", None, None
    else:
        cp, cp_gains = score_holdout(load_bundle(config.CHAMPION_DIR), holdout)
        bootstrap = paired_bootstrap(ch_gains, cp_gains)
        wins = bootstrap["win_probability"] >= config.PROMOTION_MIN_WIN_PROBABILITY
        keeps_quality = ch["gini"] >= cp["gini"] - config.PROMOTION_GINI_TOLERANCE
        promoted = wins and keeps_quality
        p = bootstrap["win_probability"]
        diff = _money(bootstrap["observed_diff"])
        if promoted:
            reason = f"wins {p:.0%} of resamples ({diff}), Gini within tolerance"
        elif not wins:
            reason = f"wins only {p:.0%} of resamples ({diff}, needs {config.PROMOTION_MIN_WIN_PROBABILITY:.0%})"
        else:
            reason = f"Gini {ch['gini']} drops more than {config.PROMOTION_GINI_TOLERANCE} vs {cp['gini']}"

    decision = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "challenger": ch,
        "champion": cp,
        "bootstrap": bootstrap,
        "promoted": promoted,
        "reason": reason,
        "holdout_hash": dataframe_hash(holdout),
    }

    client = mlflow.MlflowClient()
    name = config.REGISTERED_MODEL_NAME
    with mlflow.start_run(run_name=f"promotion-v{challenger_version}", tags={"stage": "promotion"}):
        mlflow.log_params({"challenger_version": challenger_version, "holdout_hash": decision["holdout_hash"]})
        mlflow.log_metrics({f"challenger_{k}": v for k, v in ch.items() if k != "version"})
        if cp:
            mlflow.log_metrics({f"champion_{k}": v for k, v in cp.items() if k != "version"})
            mlflow.log_metric("win_probability", bootstrap["win_probability"])
        mlflow.log_dict(decision, "promotion_decision.json")
        mlflow.set_tag("promoted", str(promoted))

    client.set_model_version_tag(name, challenger_version, "promotion", "promoted" if promoted else "rejected")
    client.set_model_version_tag(name, challenger_version, "holdout_net_value", str(ch["net_value"]))
    if bootstrap:
        client.set_model_version_tag(name, challenger_version, "win_probability", str(bootstrap["win_probability"]))

    if promoted:
        client.set_registered_model_alias(name, "champion", challenger_version)
        if config.CHAMPION_DIR.exists():
            shutil.rmtree(config.CHAMPION_DIR)
        shutil.copytree(config.MODELS_DIR / f"v{challenger_version}", config.CHAMPION_DIR)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "promotions.jsonl", "a") as f:
        f.write(json.dumps(decision) + "\n")

    print(f"  Promotion v{challenger_version}: {'PROMOTED' if promoted else 'REJECTED'} ({reason})")
    return decision
