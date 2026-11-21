import pandas as pd
import pytest

from salary import config as C
from salary.data import load_raw


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def small(df) -> pd.DataFrame:
    return df.sample(3000, random_state=C.SEED)


@pytest.fixture
def profile() -> pd.DataFrame:
    return pd.DataFrame([{
        "job_title": "Data Scientist", "experience_years": 5, "education_level": "Master",
        "skills_count": 8, "industry": "Technology", "company_size": "Medium",
        "location": "Germany", "remote_work": "Hybrid", "certifications": 1,
    }])[C.FEATURES]
