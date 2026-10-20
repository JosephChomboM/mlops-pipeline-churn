"""Discrimination, calibration, performance by driver and the absolute quality gates."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn import config


def ks_statistic(y_true, proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def calibration_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed churn rate per score decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["decile"] = (pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1).astype(int)
    table = df.groupby("decile").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    table["abs_error"] = (table["mean_predicted"] - table["observed_rate"]).abs()
    return table.reset_index()


def expected_calibration_error(y_true, proba) -> float:
    table = calibration_table(y_true, proba)
    return float((table["abs_error"] * table["n"]).sum() / table["n"].sum())


def compute_metrics(y_true, proba, contact, prefix: str = "") -> dict:
    auc = roc_auc_score(y_true, proba)
    return {
        f"{prefix}roc_auc": float(auc),
        f"{prefix}gini": float(2 * auc - 1),
        f"{prefix}ks": ks_statistic(y_true, proba),
        f"{prefix}pr_auc": float(average_precision_score(y_true, proba)),
        f"{prefix}brier": float(brier_score_loss(y_true, proba)),
        f"{prefix}ece": expected_calibration_error(y_true, proba),
        f"{prefix}precision": float(precision_score(y_true, contact, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, contact, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, contact, zero_division=0)),
    }


def add_segments(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["tenure_band"] = pd.cut(X["tenure"], config.TENURE_BANDS, labels=config.TENURE_LABELS).astype(str)
    return X


def segment_report(X: pd.DataFrame, y_true, proba, contact) -> pd.DataFrame:
    """Same metrics per driver value. Small segments or segments with one class get no Gini."""
    df = add_segments(X).assign(y=np.asarray(y_true), p=np.asarray(proba), contact=np.asarray(contact))
    rows = []
    for segment in config.SEGMENTS:
        for value, part in df.groupby(segment):
            enough = len(part) >= config.MIN_SEGMENT_ROWS and part["y"].nunique() == 2
            rows.append(
                {
                    "segment": segment,
                    "value": value,
                    "n": len(part),
                    "churn_rate": part["y"].mean(),
                    "mean_predicted": part["p"].mean(),
                    "gini": 2 * roc_auc_score(part["y"], part["p"]) - 1 if enough else np.nan,
                    "recall": recall_score(part["y"], part["contact"], zero_division=0) if part["y"].any() else np.nan,
                    "contact_rate": part["contact"].mean(),
                }
            )
    return pd.DataFrame(rows).round(4)


def run_validation_gates(metrics: dict) -> dict:
    gap = metrics["train_gini"] - metrics["test_gini"]
    checks = [
        ("min_gini", metrics["test_gini"] >= config.MIN_GINI, metrics["test_gini"], config.MIN_GINI),
        ("min_recall", metrics["test_recall"] >= config.MIN_RECALL, metrics["test_recall"], config.MIN_RECALL),
        ("max_gini_gap", gap <= config.MAX_TRAIN_TEST_GINI_GAP, gap, config.MAX_TRAIN_TEST_GINI_GAP),
        ("max_ece", metrics["test_ece"] <= config.MAX_ECE, metrics["test_ece"], config.MAX_ECE),
    ]
    checks = [
        {"name": name, "passed": bool(ok), "value": round(float(value), 4), "threshold": limit}
        for name, ok, value, limit in checks
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
