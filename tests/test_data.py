import pytest

from salary import config as C
from salary.data import DataValidationError, split, validate


def test_raw_data_matches_schema(df):
    assert len(df) == 250_000
    assert set(C.FEATURES + [C.TARGET]) <= set(df.columns)


@pytest.mark.parametrize("col,value", [("job_title", "Astronaut"), ("experience_years", 99),
                                       ("company_size", "Huge")])
def test_validate_rejects_bad_values(profile, col, value):
    bad = profile.copy()
    bad[col] = value
    with pytest.raises(DataValidationError):
        validate(bad, require_target=False)


def test_validate_rejects_missing_column(profile):
    with pytest.raises(DataValidationError):
        validate(profile.drop(columns="location"), require_target=False)


def test_split_is_deterministic_and_stratified(df):
    tr1, te1 = split(df)
    tr2, te2 = split(df)
    assert te1.index.equals(te2.index)
    assert len(te1) == int(len(df) * C.TEST_SIZE)
    share_tr = tr1.job_title.value_counts(normalize=True)
    share_te = te1.job_title.value_counts(normalize=True)
    assert (share_tr - share_te).abs().max() < 0.002
