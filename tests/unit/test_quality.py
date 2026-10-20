from churn.data.quality import fill_rate_stability, profile, table_checks


def test_profile(sample):
    prof = profile(sample).set_index("variable")
    assert prof.loc["tenure", "fill_rate"] == 1.0
    assert prof.loc["contract", "n_unique"] == 3


def test_table_checks(sample):
    checks = table_checks(sample)
    assert checks["rows"] == len(sample)
    assert checks["duplicate_ids"] == 0
    assert checks["empty_records"] == 0


def test_fill_rate_stability(sample):
    corrupted = sample.copy()
    corrupted.loc[: len(corrupted) // 2, "total_charges"] = None
    table = fill_rate_stability({"a": sample, "b": corrupted}).set_index("variable")
    assert table.loc["tenure", "max_diff"] == 0
    assert table.loc["total_charges", "max_diff"] > 0.4
