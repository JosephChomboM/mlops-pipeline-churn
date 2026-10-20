"""The production loop.

One monthly run:
    matured labels -> realized performance of past cohorts
    new cohort     -> validation -> batch scoring -> drift
    signals        -> retraining (only if there are new matured labels) -> promotion
Each step would be a task in Airflow/Prefect; here they are plain functions.
"""

import json
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config
from churn.data.ingest import build_splits, load_split
from churn.data.quality import to_markdown as data_quality_markdown
from churn.data.validation import DataValidationError
from churn.models.bundle import load_bundle
from churn.models.promotion import promote_if_better
from churn.models.train import train_challenger
from churn.monitoring.drift import drift_report
from churn.monitoring.performance import load_matured_labels, realized_performance
from churn.pipelines.simulate import deliver_cohort
from churn.reporting.model_card import write_model_card
from churn.reporting.summary import write_summary
from churn.serving.batch import score_portfolio

GENERATED = [
    config.MLRUNS_DIR,
    config.ARTIFACTS_DIR,
    config.SPLITS_DIR,
    config.SIM_DIR,
    config.PREDICTIONS_DIR,
    config.REPORTS_DIR,
]


def reset() -> None:
    for path in GENERATED:
        if path.exists():
            shutil.rmtree(path)


def _write_report(name: str, payload: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


def init() -> dict:
    splits = build_splits()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "data_quality.md").write_text(data_quality_markdown(splits))
    challenger = train_challenger(splits["train"], splits["test"], "initial", [])
    if challenger is None:
        raise SystemExit("The initial model did not pass the quality gates.")
    decision = promote_if_better(challenger["version"])
    write_model_card()
    return decision


def labeled_cohorts(as_of: str) -> dict[str, pd.DataFrame]:
    """Cohorts that passed validation, were scored and have matured labels."""
    labels = load_matured_labels(as_of)
    cohorts = {}
    for path in sorted(config.PREDICTIONS_DIR.glob("scores_*.csv")):
        date = path.stem.removeprefix("scores_")
        cohort_labels = labels[labels["snapshot_date"] == date]
        if cohort_labels.empty:
            continue
        features = pd.read_csv(config.PORTFOLIO_DIR / f"{date}.csv")
        cohorts[date] = features.merge(cohort_labels[[config.ID_COL, config.TARGET]], on=config.ID_COL)
    return cohorts


def retrain(as_of: str, reason: str) -> dict:
    champion = load_bundle(config.CHAMPION_DIR)["metadata"]
    cohorts = labeled_cohorts(as_of)
    new = sorted(set(cohorts) - set(champion["trained_on_cohorts"]))
    if not new:
        # Retraining on the same data would just give the same model
        return {"action": "waiting_for_labels", "detail": "signal active but no new matured labels"}

    # New labeled data: 80% to train, 20% to test. The promotion holdout is never touched.
    recent = pd.concat(cohorts.values(), ignore_index=True)
    add_train, add_test = train_test_split(
        recent, test_size=config.TEST_FRACTION, stratify=recent[config.TARGET], random_state=config.RANDOM_STATE
    )
    train_df = pd.concat([load_split("train"), add_train], ignore_index=True)
    test_df = pd.concat([load_split("test"), add_test], ignore_index=True)

    challenger = train_challenger(train_df, test_df, reason, sorted(cohorts))
    if challenger is None:
        return {"action": "failed_gates"}
    decision = promote_if_better(challenger["version"])
    return {
        "action": "promoted" if decision["promoted"] else "rejected",
        "challenger_version": challenger["version"],
        "new_cohorts": new,
        "reason": decision["reason"],
    }


def run_month(as_of: str, shift: float = 0.0, corrupt: bool = False) -> dict:
    print(f"\n=== {as_of} (shift={shift}{', corrupted delivery' if corrupt else ''}) ===")
    champion_version = load_bundle(config.CHAMPION_DIR)["metadata"]["version"]
    record: dict = {"as_of": as_of, "shift": shift, "champion_before": champion_version}

    matured = [c for c in realized_performance(as_of) if c["status"] == "matured"]
    latest = matured[-1] if matured else None
    record["realized"] = latest
    if latest:
        print(
            f"  Realized {latest['snapshot_date']}: gini={latest['gini']} ci95={latest['gini_ci95']} "
            f"churn predicted/actual={latest['predicted_churn_rate']}/{latest['churn_rate']} "
            f"value=${latest['realized_net_value']:,.0f}"
        )

    portfolio = deliver_cohort(as_of, shift=shift, corrupt=corrupt)
    try:
        record["scoring"] = score_portfolio(portfolio, as_of)
        print(
            f"  Batch: {record['scoring']['rows']} customers, {record['scoring']['contacted']} to contact "
            f"(expected value ${record['scoring']['expected_campaign_value']:,.0f})"
        )
    except DataValidationError as err:
        record["validation"] = {"status": "quarantined", "failures": err.summary}
        _write_report(f"validation_{as_of}.json", record["validation"])
        print(f"  VALIDATION FAILED, cohort quarantined: {err.summary}")

    drift = None
    if "scoring" in record:
        drift = drift_report(portfolio)
        _write_report(f"drift_{as_of}.json", drift)
        record["drift"] = {
            k: drift[k] for k in ("max_feature_psi", "top_feature", "drifted_features", "prediction_psi")
        }
        print(
            f"  Drift: max psi={drift['max_feature_psi']} ({drift['top_feature']}), "
            f"prediction psi={drift['prediction_psi']}"
        )

    signals = []
    if drift and drift["retrain_signal"]:
        signals.append("drift")
    # Real degradation: the whole Gini CI below the gate, measured on the current champion
    if latest and latest["model_version"] == champion_version and latest["gini_ci95"][1] < config.MIN_GINI:
        signals.append("performance")
    record["signals"] = signals

    if signals:
        print(f"  Signals: {signals} -> retraining")
        record["retraining"] = retrain(as_of, "+".join(signals))
    else:
        record["retraining"] = {"action": "none"}

    record["champion_after"] = load_bundle(config.CHAMPION_DIR)["metadata"]["version"]
    with open(config.REPORTS_DIR / "timeline.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def simulate() -> None:
    reset()
    init()
    records = [run_month(d, shift, corrupt) for d, shift, corrupt in config.SCHEDULE]
    write_summary(records)
    write_model_card()
    print(f"\nSummary written to {config.REPORTS_DIR / 'simulation.md'}")
