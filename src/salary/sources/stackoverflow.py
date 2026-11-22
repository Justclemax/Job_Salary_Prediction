"""Stack Overflow Developer Survey: download, cache and harmonise.

Source: https://survey.stackoverflow.co/ (files on github.com/StackExchange/Survey).
Licence: Open Database License (ODbL) 1.0, contents under DbCL 1.0. Attribution is
required wherever the derived data is shown.

The questionnaire changes between years (`DevType` -> `Role`, `EdLevel` ->
`Education`, `RemoteWork` -> `WorkSetting`), so `harmonise` reads both spellings
and maps answers onto short French levels. Only salaried employees with a
reported annual compensation are kept.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from salary import config as cfg

BASE = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive"
FILES = {2026: "2026/survey_results_public.csv", 2025: "2025/results.csv"}
CACHE = cfg.ROOT / "data" / "external" / "stackoverflow"
ATTRIBUTION = ("Stack Overflow Developer Survey 2025 et 2026, sous licence Open Database "
               "License (ODbL) 1.0")

# Column name per year for each concept.
COLUMNS = {
    "role": ("Role", "DevType"),
    "education": ("Education", "EdLevel"),
    "remote": ("WorkSetting", "RemoteWork"),
}

JOB_FAMILIES: dict[str, list[str]] = {
    "Dév. full-stack": ["Developer, full-stack"],
    "Dév. back-end": ["Developer, back-end"],
    "Dév. front-end": ["Developer, front-end"],
    "Dév. mobile": ["Developer, mobile"],
    "Dév. desktop / entreprise": ["Developer, desktop or enterprise applications"],
    "Dév. embarqué": ["Developer, embedded applications or devices"],
    "Dév. jeux / graphisme": ["Developer, game or graphics"],
    "QA / test": ["Developer, QA or test"],
    "Architecte": ["Architect, software or solutions"],
    "Manager technique": ["Engineering manager"],
    "DevOps / Cloud / SRE": ["DevOps engineer or professional", "Cloud infrastructure engineer",
                             "Site reliability engineer"],
    "Administrateur système": ["System administrator"],
    "Data scientist / IA": ["Data scientist", "AI/ML engineer", "Applied scientist",
                            "Developer, AI apps or physical AI"],
    "Data engineer": ["Data engineer"],
    "Analyste data / BI": ["Data or business analyst"],
    "Cybersécurité": ["Cybersecurity or InfoSec professional"],
    "Chef de projet / produit": ["Project manager", "Product manager"],
    "Chercheur": ["Academic researcher"],
}
ROLE_TO_FAMILY = {r: fam for fam, roles in JOB_FAMILIES.items() for r in roles}

# Ordered levels (ordinal features).
EDUCATION_LEVELS = ["Secondaire", "Supérieur sans diplôme", "Bac+2", "Licence", "Master",
                    "Doctorat / prof."]
_EDUCATION_PREFIX = {
    "Primary/elementary": "Secondaire", "Secondary school": "Secondaire",
    "Some college": "Supérieur sans diplôme", "Associate": "Bac+2", "Bachelor": "Licence",
    "Master": "Master", "Professional degree": "Doctorat / prof.",
}
COMPANY_LEVELS = ["< 20", "20–99", "100–499", "500–999", "1 000–4 999", "5 000–9 999", "10 000 +"]
_COMPANY_PREFIX = {
    "Just me": "< 20", "Less than 20": "< 20", "20 to 99": "20–99", "100 to 499": "100–499",
    "500 to 999": "500–999", "1,000 to 4,999": "1 000–4 999", "5,000 to 9,999": "5 000–9 999",
    "10,000 or more": "10 000 +",
}
REMOTE_LEVELS = ["Présentiel", "Hybride", "Télétravail"]
_REMOTE_PREFIX = {"In-person": "Présentiel", "Hybrid": "Hybride", "Your choice": "Hybride",
                  "Remote": "Télétravail"}
AGE_LEVELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# Survey country name -> (French label, ISO3). Countries outside this list are dropped.
COUNTRIES: dict[str, tuple[str, str]] = {
    "United States of America": ("États-Unis", "USA"),
    "Germany": ("Allemagne", "DEU"),
    "United Kingdom of Great Britain and Northern Ireland": ("Royaume-Uni", "GBR"),
    "France": ("France", "FRA"),
    "Canada": ("Canada", "CAN"),
    "India": ("Inde", "IND"),
    "Netherlands": ("Pays-Bas", "NLD"),
    "Spain": ("Espagne", "ESP"),
    "Australia": ("Australie", "AUS"),
    "Poland": ("Pologne", "POL"),
    "Italy": ("Italie", "ITA"),
    "Ukraine": ("Ukraine", "UKR"),
    "Brazil": ("Brésil", "BRA"),
    "Sweden": ("Suède", "SWE"),
    "Switzerland": ("Suisse", "CHE"),
    "Austria": ("Autriche", "AUT"),
    "Czech Republic": ("Tchéquie", "CZE"),
    "Portugal": ("Portugal", "PRT"),
    "Denmark": ("Danemark", "DNK"),
    "Belgium": ("Belgique", "BEL"),
    "Norway": ("Norvège", "NOR"),
    "Finland": ("Finlande", "FIN"),
    "Romania": ("Roumanie", "ROU"),
    "Israel": ("Israël", "ISR"),
    "Mexico": ("Mexique", "MEX"),
    "Singapore": ("Singapour", "SGP"),
    "South Africa": ("Afrique du Sud", "ZAF"),
    "Turkey": ("Turquie", "TUR"),
    "New Zealand": ("Nouvelle-Zélande", "NZL"),
    "Ireland": ("Irlande", "IRL"),
}
ISO3 = {fr: iso for fr, iso in COUNTRIES.values()}

FEATURES_NUM = ["experience_years"]
FEATURES_CAT = ["country", "job_family", "education", "company_size", "remote", "age", "year"]
FEATURES = FEATURES_NUM + FEATURES_CAT
LEVELS = {"education": EDUCATION_LEVELS, "company_size": COMPANY_LEVELS,
          "remote": REMOTE_LEVELS, "age": AGE_LEVELS, "job_family": list(JOB_FAMILIES)}


def download(year: int, force: bool = False) -> Path:
    """Download one survey year to the cache (≈ 120-140 MB) and return its path."""
    path = CACHE / f"so_{year}.csv"
    if force or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE}/{FILES[year]}", tmp)  # noqa: S310 (fixed https URL)
        tmp.replace(path)
    return path


def _by_prefix(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=object)
    for prefix, label in mapping.items():
        out[s.fillna("").str.startswith(prefix)] = label
    return out


def _pick(raw: pd.DataFrame, concept: str) -> pd.Series:
    for name in COLUMNS[concept]:
        if name in raw.columns:
            return raw[name]
    raise KeyError(f"Aucune colonne pour {concept} parmi {COLUMNS[concept]}")


def _age(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    out = pd.Series(np.nan, index=s.index, dtype=object)
    for lvl in AGE_LEVELS[:-1]:
        out[s.str.startswith(lvl)] = lvl
    out[s.str.startswith("65")] = "65+"
    return out


def harmonise(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Map one survey year onto the harmonised schema. Drops incomplete rows."""
    raw = raw[raw["Employment"].eq("Employed") & raw["ConvertedCompYearly"].gt(0)]
    countries = raw["Country"].map(lambda c: COUNTRIES.get(c, (np.nan,))[0])
    df = pd.DataFrame({
        "year": str(year),
        "country": countries,
        "job_family": _pick(raw, "role").map(ROLE_TO_FAMILY),
        "experience_years": pd.to_numeric(raw["WorkExp"], errors="coerce"),
        "education": _by_prefix(_pick(raw, "education"), _EDUCATION_PREFIX),
        "company_size": _by_prefix(raw["OrgSize"], _COMPANY_PREFIX),
        "remote": _by_prefix(_pick(raw, "remote"), _REMOTE_PREFIX),
        "age": _age(raw["Age"]),
        "salary_usd": raw["ConvertedCompYearly"].astype(float),
    }, index=raw.index)
    df = df.dropna()
    df = df[df["experience_years"].between(0, 45)]
    return df.reset_index(drop=True)


def trim_outliers(df: pd.DataFrame, low: float = 0.02, high: float = 0.98,
                  floor: float = 1_000) -> pd.DataFrame:
    """Drop implausible salaries: below `floor` USD, or outside the country's [low, high] quantiles.

    Self-reported compensation has typos (monthly vs yearly, extra zeros), and they sit
    in the tails of each country's distribution.
    """
    df = df[df["salary_usd"] >= floor]
    q = df.groupby("country")["salary_usd"].quantile([low, high]).unstack()
    lo, hi = df["country"].map(q[low]), df["country"].map(q[high])
    return df[df["salary_usd"].between(lo, hi)].reset_index(drop=True)


def load(years: tuple[int, ...] = (2025, 2026), min_per_country: int = 100,
         force: bool = False) -> pd.DataFrame:
    """Download (if needed), harmonise, stack and clean the requested survey years."""
    frames = []
    for y in years:
        raw = pd.read_csv(download(y, force), low_memory=False)
        frames.append(harmonise(raw, y))
    df = trim_outliers(pd.concat(frames, ignore_index=True))
    counts = df["country"].value_counts()
    df = df[df["country"].isin(counts[counts >= min_per_country].index)]
    df["experience_years"] = df["experience_years"].astype(int)
    return df.reset_index(drop=True)
