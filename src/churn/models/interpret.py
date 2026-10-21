"""SHAP drivers and a +-1 std stress test on the numeric features.

SHAP runs on the base model (log-odds). Platt scaling is monotonic, so the
driver ranking is the same as for the calibrated probability. One-hot columns
are summed back into their original variable.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from churn import config
from churn.decision.policy import customer_value, decide


def base_pipeline(model):
    """The pipeline inside CalibratedClassifierCV(ensemble=False), refit on all of train."""
    return model.calibrated_classifiers_[0].estimator


def _original_feature(transformed_name: str) -> str:
    kind, name = transformed_name.split("__", 1)
    if kind == "num":
        return name
    return max((c for c in config.CATEGORICAL_FEATURES if name.startswith(c + "_")), key=len)


def shap_analysis(model, X: pd.DataFrame, out_dir) -> pd.DataFrame:
    pipe = base_pipeline(model)
    features, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    sample = X.sample(min(config.SHAP_SAMPLE, len(X)), random_state=config.RANDOM_STATE)
    Xt = features.transform(sample)
    names = list(features.named_steps["prep"].get_feature_names_out())

    if hasattr(clf, "estimators_"):
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.LinearExplainer(clf, Xt[:100])
    explanation = explainer(Xt)
    explanation.feature_names = names
    if explanation.values.ndim == 3:  # sklearn forests return one output per class
        explanation = explanation[:, :, 1]

    importance = (
        pd.DataFrame(np.abs(explanation.values), columns=names)
        .T.mean(axis=1)
        .groupby(_original_feature)
        .sum()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .rename_axis("feature")
        .reset_index()
    )
    importance["share"] = (importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()).round(4)

    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Churn drivers")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_importance.png", dpi=150)
    plt.close(fig)

    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.gcf().set_size_inches(8, 5.5)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150)
    plt.close("all")

    importance.round(4).to_csv(out_dir / "shap_importance.csv", index=False)
    return importance


def stress_test(model, X: pd.DataFrame, policy: dict, out_dir) -> pd.DataFrame:
    base_proba = model.predict_proba(X)[:, 1]
    base_contact = decide(base_proba, customer_value(X), policy).mean()
    rows = []
    for feature in config.NUMERIC_FEATURES:
        std = X[feature].std()
        for direction in (-1, 1):
            stressed = X.copy()
            stressed[feature] = (stressed[feature] + direction * config.STRESS_STD * std).clip(lower=0)
            if feature == "tenure":
                stressed[feature] = stressed[feature].round().astype(int)
            proba = model.predict_proba(stressed)[:, 1]
            rows.append(
                {
                    "feature": feature,
                    "shift": f"{direction * config.STRESS_STD:+.0f} std ({direction * std:+.1f})",
                    "mean_score_base": base_proba.mean(),
                    "mean_score_stressed": proba.mean(),
                    "delta_score": proba.mean() - base_proba.mean(),
                    "contact_rate_base": base_contact,
                    "contact_rate_stressed": decide(proba, customer_value(stressed), policy).mean(),
                }
            )
    table = pd.DataFrame(rows).round(4)
    table.to_csv(out_dir / "stress_test.csv", index=False)
    return table
