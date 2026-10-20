"""Pandera schemas, checked before training and before every scoring batch.

A batch that fails is quarantined: it is not scored and never used for training.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from churn import config

YES_NO = ["Yes", "No"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]
MAX_NULL_FRACTION = 0.02
MIN_ROWS = 50


class DataValidationError(Exception):
    def __init__(self, stage: str, failures: pd.DataFrame):
        self.stage = stage
        self.failures = failures
        summary = failures.groupby(["column", "check"], dropna=False).size().reset_index(name="n_failures")
        self.summary = summary.to_dict(orient="records")
        super().__init__(f"[{stage}] data validation failed: {self.summary}")


def _category(values: list[str]) -> pa.Column:
    return pa.Column(str, pa.Check.isin(values))


def _feature_columns() -> dict[str, pa.Column]:
    return {
        config.ID_COL: pa.Column(str, unique=True),
        "tenure": pa.Column(int, pa.Check.in_range(0, 100)),
        "monthly_charges": pa.Column(float, pa.Check.in_range(0, 200)),
        "total_charges": pa.Column(float, pa.Check.ge(0), nullable=True),
        "gender": _category(["Male", "Female"]),
        "senior_citizen": _category(YES_NO),
        "partner": _category(YES_NO),
        "dependents": _category(YES_NO),
        "phone_service": _category(YES_NO),
        "multiple_lines": _category(["Yes", "No", "No phone service"]),
        "internet_service": _category(["DSL", "Fiber optic", "No"]),
        "online_security": _category(YES_NO_INTERNET),
        "online_backup": _category(YES_NO_INTERNET),
        "device_protection": _category(YES_NO_INTERNET),
        "tech_support": _category(YES_NO_INTERNET),
        "streaming_tv": _category(YES_NO_INTERNET),
        "streaming_movies": _category(YES_NO_INTERNET),
        "contract": _category(["Month-to-month", "One year", "Two year"]),
        "paperless_billing": _category(YES_NO),
        "payment_method": _category(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        ),
    }


_table_checks = [
    pa.Check(lambda df: len(df) >= MIN_ROWS, error=f"batch has fewer than {MIN_ROWS} rows"),
    pa.Check(
        lambda df: df["total_charges"].isna().mean() <= MAX_NULL_FRACTION,
        error=f"total_charges has more than {MAX_NULL_FRACTION:.0%} nulls",
    ),
]

SCORING_SCHEMA = pa.DataFrameSchema(_feature_columns(), checks=_table_checks, coerce=True, strict="filter")
TRAINING_SCHEMA = pa.DataFrameSchema(
    {**_feature_columns(), config.TARGET: pa.Column(int, pa.Check.isin([0, 1]))},
    checks=_table_checks,
    coerce=True,
    strict="filter",
)


def _validate(schema: pa.DataFrameSchema, df: pd.DataFrame, stage: str) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as err:
        raise DataValidationError(stage, err.failure_cases) from None


def validate_training(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(TRAINING_SCHEMA, df, "training")


def validate_scoring(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(SCORING_SCHEMA, df, "scoring")
