"""Train calibrated candidates, track them in MLflow and register the best as challenger.

Whether the challenger goes to production is decided in promotion.py.
"""

import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn import config
from churn.data.ingest import X_y, dataframe_hash
from churn.data.quality import profile, table_checks
from churn.data.validation import validate_training
from churn.decision.policy import choose_policy, customer_value, decide
from churn.features.build import build_preprocessor
from churn.models.bundle import save_bundle
from churn.models.evaluate import (
    calibration_table,
    compute_metrics,
    expected_calibration_error,
    run_validation_gates,
    segment_report,
)
from churn.models.interpret import shap_analysis, stress_test

warnings.filterwarnings("ignore")

# MLflow serializes with skops, which needs the custom types declared up front
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "churn.features.build.add_derived_features",
    "sklearn.frozen._frozen.FrozenEstimator",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]


def candidates() -> dict:
    seed = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed
        ),
    }


def build_model(estimator) -> Pipeline:
    return Pipeline([("features", build_preprocessor()), ("clf", estimator)])


def fit_calibrated(estimator, X, y):
    """Base model on 80% of train, Platt calibrator on the remaining 20%. No row is reused."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=config.CALIBRATION_FRACTION, stratify=y, random_state=config.RANDOM_STATE
    )
    base = build_model(estimator).fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X_cal, y_cal)
    return base, calibrated, X_fit, y_fit


def setup_mlflow() -> None:
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def _fit_candidate(name, estimator, train_df, test_df, run_tags) -> dict:
    X_train, y_train = X_y(train_df)
    X_test, y_test = X_y(test_df)
    value_test = customer_value(X_test)

    with mlflow.start_run(run_name=name, tags=run_tags) as run:
        base, model, X_fit, y_fit = fit_calibrated(estimator, X_train, y_train)

        proba_fit = model.predict_proba(X_fit)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        policy = choose_policy(y_test, proba_test, value_test)
        contact_fit = decide(proba_fit, customer_value(X_fit), policy)
        contact_test = decide(proba_test, value_test, policy)

        cv_auc = cross_val_score(build_model(estimator), X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        metrics = {
            **compute_metrics(y_fit, proba_fit, contact_fit, prefix="train_"),
            **compute_metrics(y_test, proba_test, contact_test, prefix="test_"),
            "test_ece_uncalibrated": expected_calibration_error(y_test, base.predict_proba(X_test)[:, 1]),
            "cv_gini_mean": float(2 * cv_auc.mean() - 1),
            "cv_gini_std": float(2 * cv_auc.std()),
            "test_net_value": policy["net_value"],
            "test_net_value_at_0.5": policy["net_value_at_0.5"],
            "test_contact_rate": policy["contact_rate"],
        }
        gates = run_validation_gates(metrics)

        mlflow.log_input(mlflow.data.from_pandas(train_df, name="train", targets=config.TARGET), context="training")
        mlflow.log_input(mlflow.data.from_pandas(test_df, name="test", targets=config.TARGET), context="evaluation")
        mlflow.log_params(
            {
                "model_type": name,
                "train_rows": len(train_df),
                "train_data_hash": dataframe_hash(train_df),
                "test_data_hash": dataframe_hash(test_df),
                "calibration": "sigmoid",
                "decision_rule": policy["rule"],
                **{f"clf__{k}": v for k, v in estimator.get_params().items()},
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(policy, "decision_policy.json")
        mlflow.log_dict(gates, "validation_report.json")
        mlflow.log_text(calibration_table(y_test, proba_test).round(4).to_csv(index=False), "calibration_test.csv")
        mlflow.log_text(
            segment_report(X_test, y_test, proba_test, contact_test).to_csv(index=False), "segments_test.csv"
        )
        mlflow.set_tag("validation_status", "PASS" if gates["passed"] else "FAIL")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=mlflow.models.infer_signature(X_test, proba_test),
            input_example=X_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        failed = [c["name"] for c in gates["checks"] if not c["passed"]]
        print(
            f"  {name:<20} gini={metrics['test_gini']:.3f} ks={metrics['test_ks']:.3f} "
            f"ece={metrics['test_ece']:.3f} (raw {metrics['test_ece_uncalibrated']:.3f}) "
            f"recall={metrics['test_recall']:.3f} rule={policy['rule']} "
            f"value=${policy['net_value']:>8,.0f} (@0.5 ${policy['net_value_at_0.5']:,.0f}) "
            f"gates={'PASS' if gates['passed'] else 'FAIL ' + ','.join(failed)}"
        )
        return {
            "name": name,
            "model": model,
            "metrics": metrics,
            "policy": policy,
            "gates": gates,
            "run_id": run.info.run_id,
        }


def train_challenger(train_df: pd.DataFrame, test_df: pd.DataFrame, reason: str, trained_on_cohorts: list[str]):
    """Returns the challenger's metadata, or None if no candidate passes the gates."""
    setup_mlflow()
    train_df = validate_training(train_df)
    test_df = validate_training(test_df)
    print(
        f"Training ({reason}): train={len(train_df)} test={len(test_df)} "
        f"churn_rate={train_df[config.TARGET].mean():.3f}"
    )

    tags = {"trigger": reason, "trained_on_cohorts": ",".join(trained_on_cohorts) or "none"}
    results = [_fit_candidate(name, est, train_df, test_df, tags) for name, est in candidates().items()]

    eligible = [r for r in results if r["gates"]["passed"]]
    if not eligible:
        print("  No candidate passed the gates, nothing registered.")
        return None
    best = max(eligible, key=lambda r: r["metrics"]["test_net_value"])

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", config.REGISTERED_MODEL_NAME).version
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "challenger", version)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, version, "trigger", reason)

    metadata = {
        "version": int(version),
        "model": best["name"],
        "run_id": best["run_id"],
        "trigger": reason,
        "trained_on_cohorts": trained_on_cohorts,
        "train_rows": len(train_df),
        "train_data_hash": dataframe_hash(train_df),
        "test_data_hash": dataframe_hash(test_df),
        "test_metrics": best["metrics"],
    }
    bundle = config.MODELS_DIR / f"v{version}"
    save_bundle(bundle, best["model"], best["policy"], metadata, train_df, parity_sample=test_df)
    explain(best, bundle, test_df, train_df)
    print(f"  Challenger: {best['name']} -> {config.REGISTERED_MODEL_NAME} v{version}")
    return metadata


def explain(best: dict, bundle: Path, test_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    """Interpretability and reports for the winner, saved in its bundle and its MLflow run."""
    X_test, y_test = X_y(test_df)
    proba = best["model"].predict_proba(X_test)[:, 1]
    contact = decide(proba, customer_value(X_test), best["policy"])

    shap_analysis(best["model"], X_test, bundle)
    stress_test(best["model"], X_test, best["policy"], bundle)
    calibration_table(y_test, proba).round(4).to_csv(bundle / "calibration_test.csv", index=False)
    segment_report(X_test, y_test, proba, contact).to_csv(bundle / "segments_test.csv", index=False)
    profile(train_df).to_csv(bundle / "data_quality_train.csv", index=False)
    (bundle / "data_quality_checks.json").write_text(json.dumps(table_checks(train_df), indent=2))

    client = mlflow.MlflowClient()
    for name in (
        "shap_importance.png",
        "shap_beeswarm.png",
        "shap_importance.csv",
        "stress_test.csv",
        "data_quality_train.csv",
        "data_quality_checks.json",
    ):
        client.log_artifact(best["run_id"], str(bundle / name), artifact_path="interpretation")
