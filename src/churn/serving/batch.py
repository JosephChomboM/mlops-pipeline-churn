"""Batch scoring of the customer base: the primary serving path.

Validates the snapshot, scores it with the champion, applies the decision policy
and writes the prediction log (with the model version) that is later joined with
the delayed labels.
"""

import argparse
import json

import pandas as pd

from churn import config
from churn.data.validation import validate_scoring
from churn.decision.policy import customer_value, decide, expected_value, risk_band
from churn.models.bundle import load_bundle


def score_portfolio(portfolio: pd.DataFrame, snapshot_date: str) -> dict:
    valid = validate_scoring(portfolio)
    bundle = load_bundle(config.CHAMPION_DIR)
    policy = bundle["policy"]

    proba = bundle["model"].predict_proba(valid[config.FEATURES])[:, 1]
    value = customer_value(valid)
    scores = pd.DataFrame(
        {
            config.ID_COL: valid[config.ID_COL],
            "snapshot_date": snapshot_date,
            "churn_probability": proba,
            "risk_band": [risk_band(p) for p in proba],
            "contact": decide(proba, value, policy),
            "expected_value": expected_value(proba, value).round(2),
            "decision_rule": policy["rule"],
            "model_version": bundle["metadata"]["version"],
        }
    ).sort_values("expected_value", ascending=False)

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PREDICTIONS_DIR / f"scores_{snapshot_date}.csv"
    scores.to_csv(out, index=False)

    return {
        "snapshot_date": snapshot_date,
        "rows": len(scores),
        "contacted": int(scores["contact"].sum()),
        "mean_probability": round(float(proba.mean()), 4),
        "expected_campaign_value": round(float(scores.loc[scores["contact"], "expected_value"].sum()), 2),
        "model_version": bundle["metadata"]["version"],
        "output": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a portfolio snapshot")
    parser.add_argument("portfolio_csv")
    parser.add_argument("--date", required=True, help="snapshot date, YYYY-MM-DD")
    args = parser.parse_args()
    print(json.dumps(score_portfolio(pd.read_csv(args.portfolio_csv), args.date), indent=2))


if __name__ == "__main__":
    main()
