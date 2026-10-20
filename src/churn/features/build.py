"""Feature engineering.

Everything lives inside one sklearn Pipeline, fitted on train only. The batch
job and the API load the same object, so there is no training/serving skew.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config

SERVICE_COLUMNS = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
DERIVED_FEATURES = ["avg_monthly_spend", "num_services", "is_new_customer"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].clip(lower=1)
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    columns = ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + DERIVED_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,  # dense output; SHAP needs it and there are only ~50 columns
    )
    return Pipeline([("derive", FunctionTransformer(add_derived_features)), ("prep", columns)])
