"""Champion vs challenger on the frozen holdout.

A challenger replaces the champion only if, on rows no model has trained on, it
  1. earns more net retention value, and
  2. loses no more than PROMOTION_GINI_TOLERANCE Gini.
Promoting moves the "champion" alias in the registry and copies the bundle that
the batch job and the API serve. Rolling back is moving the alias back.
"""

import json
import shutil
from datetime import UTC, datetime

import mlflow
from sklearn.metrics import recall_score, roc_auc_score

from churn import config
from churn.data.ingest import X_y, dataframe_hash, load_split
from churn.decision.policy import customer_value, decide, net_value
from churn.models.bundle import champion_exists, load_bundle
from churn.models.evaluate import expected_calibration_error, ks_statistic
from churn.models.train import setup_mlflow


def evaluate_on_holdout(bundle: dict, holdout) -> dict:
    X, y = X_y(holdout)
    proba = bundle["model"].predict_proba(X)[:, 1]
    value = customer_value(X)
    contact = decide(proba, value, bundle["policy"])
    return {
        "version": bundle["metadata"]["version"],
        "gini": round(2 * float(roc_auc_score(y, proba)) - 1, 4),
        "ks": round(ks_statistic(y, proba), 4),
        "ece": round(expected_calibration_error(y, proba), 4),
        "recall": round(float(recall_score(y, contact)), 4),
        "contact_rate": round(float(contact.mean()), 4),
        "net_value": round(net_value(y, contact, value), 2),
    }


def promote_if_better(challenger_version: int) -> dict:
    setup_mlflow()
    holdout = load_split("holdout")
    ch = evaluate_on_holdout(load_bundle(config.MODELS_DIR / f"v{challenger_version}"), holdout)

    if not champion_exists():
        promoted, reason, cp = True, "no champion yet", None
    else:
        cp = evaluate_on_holdout(load_bundle(config.CHAMPION_DIR), holdout)
        beats_value = ch["net_value"] > cp["net_value"]
        keeps_quality = ch["gini"] >= cp["gini"] - config.PROMOTION_GINI_TOLERANCE
        promoted = beats_value and keeps_quality
        if promoted:
            reason = f"value ${ch['net_value']:,.0f} > ${cp['net_value']:,.0f}, Gini within tolerance"
        elif not beats_value:
            reason = f"value ${ch['net_value']:,.0f} <= champion ${cp['net_value']:,.0f}"
        else:
            reason = f"Gini {ch['gini']} drops more than {config.PROMOTION_GINI_TOLERANCE} vs {cp['gini']}"

    decision = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "challenger": ch,
        "champion": cp,
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
        mlflow.log_dict(decision, "promotion_decision.json")
        mlflow.set_tag("promoted", str(promoted))

    client.set_model_version_tag(name, challenger_version, "promotion", "promoted" if promoted else "rejected")
    client.set_model_version_tag(name, challenger_version, "holdout_net_value", str(ch["net_value"]))

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
