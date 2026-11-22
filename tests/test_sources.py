"""Real-data connectors, tested offline on hand-made survey rows."""

import json

import numpy as np
import pandas as pd
import pytest

from salary.sources import stackoverflow as so
from salary.sources import worldbank as wb


def _raw(year: int) -> pd.DataFrame:
    """Two respondents in the column names of the given survey year."""
    role, edu, remote = (("Role", "Education", "WorkSetting") if year == 2026
                         else ("DevType", "EdLevel", "RemoteWork"))
    return pd.DataFrame({
        "Employment": ["Employed", "Employed", "Student"],
        "ConvertedCompYearly": [120_000.0, 30_000.0, 5_000.0],
        "Country": ["United States of America", "India", "France"],
        role: ["Data scientist", "Developer, back-end", "Developer, full-stack"],
        "WorkExp": [8, 3, 1],
        edu: ["Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
              "Bachelor's degree (B.A., B.S., B.Eng., etc.)", "Some college/university study"],
        "OrgSize": ["10,000 or more employees", "20 to 99 employees", "Less than 20 employees"],
        remote: ["Remote", "Hybrid (some remote, leans heavy to in-person)", "In-person"],
        "Age": ["25-34 years old", "18-24 years old", "18-24 years old"],
    })


@pytest.mark.parametrize("year", [2025, 2026])
def test_harmonise_maps_both_questionnaires(year):
    df = so.harmonise(_raw(year), year)
    assert list(df.columns) == ["year", "country", "job_family", "experience_years", "education",
                                "company_size", "remote", "age", "salary_usd"]
    assert len(df) == 2  # the student is dropped
    us = df.iloc[0]
    assert (us.country, us.job_family, us.education, us.company_size, us.remote, us.age) == (
        "États-Unis", "Data scientist / IA", "Master", "10 000 +", "Télétravail", "25-34")
    assert df.iloc[1].education == "Licence"  # straight or curly apostrophe
    assert df.iloc[1].remote == "Hybride"


def test_levels_are_consistent():
    for col, levels in so.LEVELS.items():
        assert len(levels) == len(set(levels)), col
    assert set(so.ROLE_TO_FAMILY.values()) == set(so.LEVELS["job_family"])
    assert len(so.ISO3) == len(so.COUNTRIES)


def test_trim_outliers_is_per_country():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"country": ["A"] * 100 + ["B"] * 100,
                       "salary_usd": np.r_[rng.uniform(10_000, 20_000, 100), rng.uniform(100_000, 200_000, 100)]})
    df.loc[0, "salary_usd"] = 500  # below the floor
    out = so.trim_outliers(df)
    assert out["salary_usd"].min() >= 1_000
    assert 180 <= len(out) < 200
    assert out.groupby("country").size().min() > 90


def test_ppp_conversion(tmp_path, monkeypatch):
    rows = {"PA.NUS.PPP": [("USA", 1.0), ("IND", 20.0)], "PA.NUS.FCRF": [("USA", 1.0), ("IND", 80.0)]}
    for code, vals in rows.items():
        (tmp_path / f"{code}_2018_2026.json").write_text(json.dumps(
            [{"countryiso3code": c, "date": "2025", "value": v} for c, v in vals]))
    monkeypatch.setattr(wb, "CACHE", tmp_path)
    pl = wb.price_level(["USA", "IND"], 2025).set_index("iso3")["price_level"]
    assert pl["USA"] == 1.0 and pl["IND"] == pytest.approx(0.25)
    df = pd.DataFrame({"iso3": ["USA", "IND"], "salary_usd": [100_000.0, 20_000.0]})
    assert wb.to_ppp(df).tolist() == pytest.approx([100_000.0, 80_000.0])
