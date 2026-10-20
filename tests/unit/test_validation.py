import numpy as np
import pytest

from churn.data.validation import DataValidationError, validate_scoring, validate_training


def test_clean_data_passes(sample):
    assert len(validate_training(sample)) == len(sample)


def test_extra_columns_are_dropped(sample):
    out = validate_scoring(sample.assign(snapshot_date="2026-01-01"))
    assert "snapshot_date" not in out.columns


def test_upstream_schema_change_is_rejected(sample):
    corrupt = sample.copy()
    corrupt["monthly_charges"] *= 100
    corrupt["internet_service"] = corrupt["internet_service"].replace("Fiber optic", "Fiber")
    with pytest.raises(DataValidationError) as err:
        validate_scoring(corrupt)
    assert {"monthly_charges", "internet_service"} <= {row["column"] for row in err.value.summary}


def test_too_many_nulls_is_rejected(sample):
    corrupt = sample.copy()
    corrupt.loc[corrupt.index[:40], "total_charges"] = np.nan  # 10%, limit is 2%
    with pytest.raises(DataValidationError):
        validate_scoring(corrupt)


def test_duplicate_ids_are_rejected(sample):
    corrupt = sample.copy()
    corrupt.loc[1, "customer_id"] = corrupt.loc[0, "customer_id"]
    with pytest.raises(DataValidationError):
        validate_scoring(corrupt)


def test_invalid_target_is_rejected(sample):
    with pytest.raises(DataValidationError):
        validate_training(sample.assign(churn=2))
