"""Single source of truth for paths, columns, category levels and seeds."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("SALARY_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw" / "job_salary_prediction_dataset.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.2
TARGET = "salary"

NUMERIC = ["experience_years", "skills_count", "certifications"]
ORDINAL_ORDERS: dict[str, list[str]] = {
    "education_level": ["High School", "Diploma", "Bachelor", "Master", "PhD"],
    "company_size": ["Startup", "Small", "Medium", "Large", "Enterprise"],
}
NOMINAL_LEVELS: dict[str, list[str]] = {
    "job_title": [
        "AI Engineer", "Backend Developer", "Business Analyst", "Cloud Engineer",
        "Cybersecurity Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer",
        "Frontend Developer", "Machine Learning Engineer", "Product Manager",
        "Software Engineer",
    ],
    "industry": [
        "Consulting", "Education", "Finance", "Government", "Healthcare",
        "Manufacturing", "Media", "Retail", "Technology", "Telecom",
    ],
    "location": [
        "Australia", "Canada", "Germany", "India", "Netherlands", "Remote",
        "Singapore", "Sweden", "UK", "USA",
    ],
    "remote_work": ["No", "Hybrid", "Yes"],
}
NOMINAL = list(NOMINAL_LEVELS)
ORDINAL = list(ORDINAL_ORDERS)
CATEGORICAL = NOMINAL + ORDINAL
FEATURES = NUMERIC + CATEGORICAL
CATEGORY_LEVELS = {**NOMINAL_LEVELS, **ORDINAL_ORDERS}
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "experience_years": (0, 20),
    "skills_count": (1, 19),
    "certifications": (0, 5),
}

# French labels used by the notebook and the app.
LABELS_FR = {
    "job_title": "Poste",
    "experience_years": "Années d'expérience",
    "education_level": "Niveau d'études",
    "skills_count": "Nombre de compétences",
    "industry": "Secteur",
    "company_size": "Taille d'entreprise",
    "location": "Localisation",
    "remote_work": "Télétravail",
    "certifications": "Certifications",
    "salary": "Salaire annuel",
}
